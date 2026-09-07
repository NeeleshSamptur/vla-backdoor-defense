#!/usr/bin/env python
"""Attention-sink-calibration backdoor detector.

Idea (branch: experiments/attention-sink-detection): instead of asking "does
this episode's own attention rows disagree with each other" (the existing
FTT statistic), ask "does this episode's overall attention pattern disagree
with what CLEAN episodes normally look like for this task." Motivated by
VisAttnSink (arxiv 2503.03321): LMMs have a strong, content-irrelevant
tendency to dump attention on a few fixed "sink" patches -- visible directly
in this project's own GoBA heatmaps as fixed bright columns present in BOTH
clean and trigger conditions. If a backdoor trigger manufactures a NEW,
anomalous sink at the trigger's location, an episode's attention profile
should look unusually far from the normal (clean-calibrated) sink pattern,
even if the episode's own internal rows are numerically well-behaved.

Method:
  1. Row-normalize each episode's (desc_tokens x image_patches) attention
     matrix, then mean over desc-token rows -> one [256] "profile" vector
     per episode (this is the same quantity FTT calls the "assimilated
     pattern", but here it's compared across episodes, not within one).
  2. Calibrate a "clean reference profile" by averaging the per-episode
     profiles of a CALIBRATION SUBSET of clean episodes only (half the
     seeds, held out from scoring) -- never seen at test time.
  3. Score every held-out clean and every trigger episode by its L2
     distance from that clean reference. Higher distance = more anomalous
     relative to the calibrated clean norm = predicted triggered.
  4. AUROC (label 1 = trigger, score = distance, NOT negated -- polarity is
     "far from clean norm", opposite convention from ftt_score's "low FTT").

No GPU, no model -- reads already-extracted results/goba_layeravg_causal_fixed/
(layer-averaged text2img matrices, causal-fixed attention). Pure numpy.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

DEFENSE_REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, DEFENSE_REPO)

from detectors.ftt import row_normalize  # reuse, do not reimplement


def manual_auroc(neg_scores, pos_scores):
    """label 0 = neg (clean), label 1 = pos (trigger). Higher score -> predicted positive."""
    neg = np.asarray(neg_scores, dtype=np.float64)
    pos = np.asarray(pos_scores, dtype=np.float64)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    order = np.argsort(s)
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i, r = 0, 1
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + (r + (j - i))) / 2.0
        r += (j - i + 1)
        i = j + 1
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def episode_profile(attn_text_image: np.ndarray) -> np.ndarray:
    """Row-normalize, then mean over query (desc-token) rows -> [n_patches]."""
    P = row_normalize(attn_text_image)
    return P.mean(axis=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="attack", choices=["attack", "clean_baseline"],
                     help="filter to one checkpoint's episodes -- this directory can hold "
                          "both the real attack checkpoint's and a clean-baseline checkpoint's "
                          "episodes side by side; mixing them silently pools two different "
                          "models' data into one run, which is wrong.")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{DEFENSE_REPO}/results/goba_layeravg_causal_fixed/*.npz"))
    assert files, "no episodes found -- run adapters/goba/extract_img2img_layeravg_ftt.py first"

    episodes = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        if meta["extra"]["role"] != args.role:
            continue
        episodes.append(dict(
            label=meta["label"], task_id=meta["task_id"], seed=meta["seed"],
            profile=episode_profile(z["attn_text_image"]),
        ))
    print(f"[*] role={args.role}")

    clean = [e for e in episodes if e["label"] == 0]
    trig = [e for e in episodes if e["label"] == 1]
    print(f"[*] loaded {len(episodes)} episodes ({len(clean)} clean, {len(trig)} trigger)")

    # Calibration split: even seeds -> calibration (never scored), odd seeds -> held-out test.
    # (seed here is the within-condition episode index used when saving, 0..9 per task/condition)
    calib_clean = [e for e in clean if e["seed"] % 2 == 0]
    test_clean = [e for e in clean if e["seed"] % 2 == 1]
    print(f"[*] calibration: {len(calib_clean)} clean episodes (even seeds)")
    print(f"[*] held-out test: {len(test_clean)} clean episodes (odd seeds) + {len(trig)} trigger episodes")

    clean_reference = np.mean([e["profile"] for e in calib_clean], axis=0)

    def dist(e):
        return float(np.linalg.norm(e["profile"] - clean_reference))

    test_clean_scores = [dist(e) for e in test_clean]
    trig_scores = [dist(e) for e in trig]

    auroc = manual_auroc(test_clean_scores, trig_scores)
    print()
    print(f"[*] clean (held-out) distance-from-clean-norm: mean={np.mean(test_clean_scores):.5f} "
          f"std={np.std(test_clean_scores):.5f}")
    print(f"[*] trigger distance-from-clean-norm:            mean={np.mean(trig_scores):.5f} "
          f"std={np.std(trig_scores):.5f}")
    print(f"[*] AUROC (higher distance = predicted trigger): {auroc:.4f}")

    # Sanity/leakage check: also score the CALIBRATION clean episodes themselves
    # against their own reference (should be small/near-zero average distance,
    # confirming the reference is a real central tendency, not noise).
    calib_self_scores = [dist(e) for e in calib_clean]
    print(f"[*] (sanity) calibration clean episodes' own distance from their own reference: "
          f"mean={np.mean(calib_self_scores):.5f} (should be < held-out clean mean)")


if __name__ == "__main__":
    main()
