#!/usr/bin/env python
"""Two orderings of the 'shared reference, per-layer rows' FTT construction
on GoBA text2img attention -- the mirror of
analysis/score_goba_flatten_normalize_order.py, but for the OTHER
construction: never flatten the 32 layers into one map. Instead build ONE
shared reference vector (average over both layer and token jointly:
row(avg, avg)), and measure every individual (layer, token) row -- kept
un-flattened -- against that one reference, averaging distances over
tokens then layers.

  (1) normalize-then-average: row-normalize each layer's rows first, take
      each layer's own row-mean (32 vectors), average those into ONE shared
      reference, then measure the (already normalized) individual rows
      against it. Reproduces detectors/ftt.py's ftt_score_layerwise exactly
      (also analysis/score_goba_meanofmeans_layerwise_ftt.py's number) --
      kept here for direct side-by-side comparison, not a re-derivation.
  (2) average-then-normalize: average the RAW rows jointly over (layer,
      token) into one raw reference first, THEN normalize that single
      reference vector once. Individual rows are still normalized
      independently (row-normalization is a per-row operation, needed for
      the reference and the rows to live in the same space before a
      distance is meaningful) -- only the REFERENCE's own construction
      differs in when normalization happens relative to averaging.

Both orderings, per the earlier "mean of means" finding, come out INVERTED
relative to the fixed "low FTT -> triggered" polarity convention used
everywhere else -- reported as-is, not flipped.

Reads results/goba_text2img_layerwise_causal_fixed/*.npz, role=='attack'
only. New scoring file -- does not modify detectors/ftt.py or any other
scoring script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "goba_text2img_layerwise_causal_fixed"
BALANCED_63V63_JSON = REPO / "results" / "ftt_goba_text2img_layerwise_63v63_balanced.json"
EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def score_normalize_then_average(layers: np.ndarray) -> float:
    P = row_normalize(layers)              # [L, T, P]
    per_layer_mean = P.mean(axis=1)        # [L, P]
    ref = per_layer_mean.mean(axis=0)      # [P] -- shared reference
    per_token = np.linalg.norm(P - ref[None, None, :], axis=-1)  # [L, T]
    return float(per_token.mean(axis=1).mean())


def score_average_then_normalize(layers: np.ndarray) -> float:
    ref_raw = layers.mean(axis=(0, 1))     # [P] -- average of RAW rows, jointly over (layer, token)
    ref = ref_raw / max(ref_raw.sum(), EPS)  # normalize the reference once, at the end
    P = row_normalize(layers)              # rows still normalized individually -- needed for a
                                            # meaningful distance in the same (probability) space
    per_token = np.linalg.norm(P - ref[None, None, :], axis=-1)
    return float(per_token.mean(axis=1).mean())


def rank_auroc(clean_scores, trig_scores) -> float:
    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trig_scores, dtype=np.float64)
    s = np.concatenate([-c, -t])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_c, n_t = len(c), len(t)
    return float((ranks[n_c:].sum() - n_t * (n_t + 1) / 2) / (n_c * n_t))


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")

    with open(BALANCED_63V63_JSON) as f:
        balanced = json.load(f)
    clean_63_keys = {(tid, seed) for tid, seed in balanced["chosen_clean_pairs"]}
    trig_63_keys = {(tid, seed) for tid, seed in balanced["trigger_pairs"]}

    variants = ["normalize_then_average", "average_then_normalize"]
    clean_100 = {v: {} for v in variants}
    trig_100 = {v: {} for v in variants}

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        if meta["extra"]["role"] != "attack":
            continue
        layers = d["attn_text_image_layers"].astype(np.float64)
        key = (meta["task_id"], meta["seed"])
        bucket = clean_100 if meta["label"] == 0 else trig_100
        bucket["normalize_then_average"][key] = score_normalize_then_average(layers)
        bucket["average_then_normalize"][key] = score_average_then_normalize(layers)

    for v in variants:
        assert len(clean_100[v]) == 100 and len(trig_100[v]) == 100

    results = {}
    print()
    for v in variants:
        c100 = list(clean_100[v].values())
        t100 = list(trig_100[v].values())
        c63 = [clean_100[v][k] for k in sorted(clean_63_keys)]
        t63 = [trig_100[v][k] for k in sorted(trig_63_keys)]
        auroc_100 = rank_auroc(c100, t100)
        auroc_63 = rank_auroc(c63, t63)
        results[v] = {"auroc_100v100": auroc_100, "auroc_63v63": auroc_63,
                      "clean_mean": float(np.mean(c100)), "trig_mean": float(np.mean(t100))}
        print(f"[*] {v:24s}  100v100={auroc_100:.4f}  63v63={auroc_63:.4f}  "
              f"(clean_mean={np.mean(c100):.4f} trig_mean={np.mean(t100):.4f})")

    out_path = REPO / "results" / "ftt_goba_text2img_sharedref_normalize_order.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
