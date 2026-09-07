#!/usr/bin/env python
"""Per-token (per-patch) refinement of attention_sink_calibration.py.

Instead of scoring an episode by its AGGREGATE L2 distance from a clean
reference profile (diluted across 256 patches), score it by the MAX
per-patch z-score: for each patch i, (episode_value[i] - clean_mean[i]) /
clean_std[i], using per-patch mean/std estimated from calibration clean
episodes only, then take the max over i. This targets the actual mechanism
a small, localized trigger produces -- one new, unexplained hotspot at a
specific patch -- rather than diluting it across the whole profile, and
z-scoring (not raw magnitude) accounts for patches that are naturally noisy
across clean episodes (open floor, background) vs. patches that are normally
rock-stable (fixed sinks, the robot base) so a stable patch spiking is
weighted more than a naturally-variable one doing the same.

Same calibration split as attention_sink_calibration.py (even seeds =
calibration, odd seeds + all trigger = held-out test) -- reused exactly, not
re-derived, so the two scripts are directly comparable.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

DEFENSE_REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, DEFENSE_REPO)

from detectors.ftt import row_normalize
from attention_sink_calibration import episode_profile, manual_auroc

EPS = 1e-9


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="attack", choices=["attack", "clean_baseline"],
                     help="see attention_sink_calibration.py's identical flag docstring -- "
                          "this directory holds both roles' episodes side by side.")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{DEFENSE_REPO}/results/goba_layeravg_causal_fixed/*.npz"))
    assert files, "no episodes found"

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
    calib_clean = [e for e in clean if e["seed"] % 2 == 0]
    test_clean = [e for e in clean if e["seed"] % 2 == 1]
    print(f"[*] {len(episodes)} episodes ({len(clean)} clean, {len(trig)} trigger)")
    print(f"[*] calibration: {len(calib_clean)} clean episodes (even seeds)")
    print(f"[*] held-out test: {len(test_clean)} clean episodes (odd seeds) + {len(trig)} trigger")

    calib_profiles = np.stack([e["profile"] for e in calib_clean])  # [n_calib, 256]
    clean_mean = calib_profiles.mean(axis=0)
    clean_std = calib_profiles.std(axis=0)
    print(f"[*] per-patch clean std: min={clean_std.min():.6f} max={clean_std.max():.6f} "
          f"mean={clean_std.mean():.6f}")

    def max_zscore(e):
        z = (e["profile"] - clean_mean) / (clean_std + EPS)
        return float(z.max())  # only positive-direction spikes count as "unusually high attention"

    def argmax_patch(e):
        z = (e["profile"] - clean_mean) / (clean_std + EPS)
        return int(np.argmax(z)), float(z.max())

    test_clean_scores = [max_zscore(e) for e in test_clean]
    trig_scores = [max_zscore(e) for e in trig]
    calib_self_scores = [max_zscore(e) for e in calib_clean]

    auroc = manual_auroc(test_clean_scores, trig_scores)
    print()
    print(f"[*] held-out clean max z-score:  mean={np.mean(test_clean_scores):.3f} "
          f"std={np.std(test_clean_scores):.3f} max={np.max(test_clean_scores):.3f}")
    print(f"[*] trigger max z-score:         mean={np.mean(trig_scores):.3f} "
          f"std={np.std(trig_scores):.3f} min={np.min(trig_scores):.3f}")
    print(f"[*] AUROC (higher max z-score = predicted trigger): {auroc:.4f}")
    print(f"[*] (sanity) calibration clean episodes' own max z-score: "
          f"mean={np.mean(calib_self_scores):.3f} (should be small/near held-out clean mean)")

    print()
    print("[*] which patch fires for a sample of trigger episodes (localization check):")
    patch_hits = {}
    for e in trig:
        p, _ = argmax_patch(e)
        patch_hits[p] = patch_hits.get(p, 0) + 1
    top_patches = sorted(patch_hits.items(), key=lambda kv: -kv[1])[:5]
    for patch, count in top_patches:
        row, col = divmod(patch, 16)
        print(f"    patch {patch:3d} (row {row}, col {col}): argmax in {count}/{len(trig)} trigger episodes")


if __name__ == "__main__":
    main()
