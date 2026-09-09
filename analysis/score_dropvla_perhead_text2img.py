#!/usr/bin/env python
"""Score per-(layer,head) text2img FTT AUROC for DropVLA, 100 clean vs 100
trigger (the same n=100/100, libero_spatial, trigger_mode=vision episode
set as the existing head-averaged run in
results/dropvla_text2img_alllayers_causal_fixed/).

Reads results/dropvla_perhead_text2img/*.npz (written by the sibling
extraction script extract_perhead_text2img_ftt.py). Standalone script --
does not import or modify any existing scoring/extraction file.

Reports: per-head summary (each head's own best-over-layers AUROC, i.e. the
same "mean of all heads" summary computed for GoBA), ranked, plus the mean
across all heads of that per-head best score.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "dropvla_perhead_text2img"


def rank_auroc(clean_scores, trig_scores) -> float:
    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trig_scores, dtype=np.float64)
    if len(c) == 0 or len(t) == 0:
        return float("nan")
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
    rank_sum_trig = ranks[n_c:].sum()
    return float((rank_sum_trig - n_t * (n_t + 1) / 2) / (n_c * n_t))


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")

    clean_cells, trig_cells = [], []
    n_layers = n_heads = None
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        extra = meta["extra"]
        cell = np.asarray(extra["text2img_perhead_ftt"], dtype=np.float64)  # [L,H]
        if n_layers is None:
            n_layers, n_heads = cell.shape
        (clean_cells if meta["label"] == 0 else trig_cells).append(cell)

    n_clean, n_trig = len(clean_cells), len(trig_cells)
    print(f"[*] n_clean={n_clean}  n_trig={n_trig}")

    clean_stack = np.stack(clean_cells)
    trig_stack = np.stack(trig_cells)

    auroc = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            auroc[l, h] = rank_auroc(clean_stack[:, l, h], trig_stack[:, l, h])

    out = {
        "n_clean": n_clean, "n_trig": n_trig,
        "n_layers": n_layers, "n_heads": n_heads,
        "auroc": auroc.tolist(),
    }
    out_path = REPO / "results" / "dropvla_perhead_text2img_auroc.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[*] saved -> {out_path}")

    best_per_head = auroc.max(axis=0)
    best_layer_per_head = auroc.argmax(axis=0)
    order = np.argsort(best_per_head)[::-1]
    print(f"\n[*] {n_clean} clean vs {n_trig} trigger -- heads ranked by their OWN best-over-layers AUROC:")
    for h in order:
        print(f"    head={h:2d}  best_AUROC={best_per_head[h]:.4f}  (at layer {best_layer_per_head[h]})")
    print(f"\n    MEAN over all {n_heads} heads of each head's own best-over-layers AUROC: {best_per_head.mean():.4f}")

    # also report the pure head-averaged AUROC per layer, for direct comparison
    # to the existing head-averaged report (same statistic, same data source)
    head_avg_per_layer = np.zeros(n_layers)
    for l in range(n_layers):
        head_avg_per_layer[l] = rank_auroc(clean_stack[:, l].mean(axis=1), trig_stack[:, l].mean(axis=1))
    best_l = int(np.argmax(head_avg_per_layer))
    print(f"\n    [comparison] head-AVERAGED AUROC, best layer {best_l}: {head_avg_per_layer[best_l]:.4f}")
    print(f"    [comparison] head-AVERAGED AUROC, mean over all layers: {head_avg_per_layer.mean():.4f}")


if __name__ == "__main__":
    main()
