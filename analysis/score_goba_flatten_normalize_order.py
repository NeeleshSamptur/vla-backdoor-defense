#!/usr/bin/env python
"""The 'flatten, normalize-then-average' FTT construction on GoBA text2img
desc_only attention: row-normalize each of the 32 layers' maps separately,
average those already-normalized maps into one map, then run FTT
(Frobenius-norm distance from each row to the map's own row-mean) on that
one map. Scored against AUROC.

Reads results/goba_text2img_layerwise_causal_fixed/*.npz, role=='attack'
only.
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
    normed = row_normalize(layers)   # [L, T, P] -- each row already sums to 1
    avg_map = normed.mean(axis=0)    # [T, P] -- average of already-normalized rows
    ref = avg_map.mean(axis=0)       # [P]
    return float(np.linalg.norm(avg_map - ref[None, :], axis=1).mean())


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

    clean_100, trig_100 = {}, {}

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        if meta["extra"]["role"] != "attack":
            continue
        layers = d["attn_text_image_layers"].astype(np.float64)
        key = (meta["task_id"], meta["seed"])
        bucket = clean_100 if meta["label"] == 0 else trig_100
        bucket[key] = score_normalize_then_average(layers)

    assert len(clean_100) == 100 and len(trig_100) == 100

    c100 = list(clean_100.values())
    t100 = list(trig_100.values())
    c63 = [clean_100[k] for k in sorted(clean_63_keys)]
    t63 = [trig_100[k] for k in sorted(trig_63_keys)]
    auroc_100 = rank_auroc(c100, t100)
    auroc_63 = rank_auroc(c63, t63)
    results = {"normalize_then_average": {"auroc_100v100": auroc_100, "auroc_63v63": auroc_63}}
    print(f"[*] normalize_then_average  100v100={auroc_100:.4f}  63v63={auroc_63:.4f}")

    out_path = REPO / "results" / "ftt_goba_text2img_flatten_normalize_order.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
