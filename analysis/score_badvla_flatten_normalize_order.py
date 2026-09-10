#!/usr/bin/env python
"""The 'flatten, normalize-then-average' FTT construction on BadVLA
(white_patch attack) text2img desc_only attention: row-normalize each of the
32 layers' maps separately, average those already-normalized maps into one
map, then run FTT (Frobenius-norm distance from each row to the map's own
row-mean) on that one map. Scored against AUROC.

Reads results/badvla_text2img_alllayers_causal_fixed/*.npz (written by
adapters/badvla_white_patch/extract_text2img_alllayers_driver.py), role==
'attack' only. attn_text_image_layers is already desc_only (query = task
description tokens only, extra["text_scope"]=="desc_only"), primary camera,
shape [32 layers, T query tokens, 256 image patches].

Unlike GoBA, there is no pre-existing balanced-63v63 pairing file for
BadVLA, so this script reports the full 100-clean-vs-100-trigger AUROC only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "badvla_text2img_alllayers_causal_fixed"
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

    clean, trig = {}, {}

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        if meta["extra"]["role"] != "attack":
            continue
        layers = d["attn_text_image_layers"].astype(np.float64)
        key = (meta["task_id"], meta["seed"])
        bucket = clean if meta["label"] == 0 else trig
        bucket[key] = score_normalize_then_average(layers)

    print(f"[*] n_clean={len(clean)} n_trigger={len(trig)}")

    c = list(clean.values())
    t = list(trig.values())
    auroc = rank_auroc(c, t)
    results = {
        "n_clean": len(c),
        "n_trigger": len(t),
        "clean_mean": float(np.mean(c)),
        "trig_mean": float(np.mean(t)),
        "auroc_100v100": auroc,
    }
    print(f"[*] normalize_then_average  n_clean={len(c)} n_trigger={len(t)}  "
          f"clean_mean={np.mean(c):.5f}  trig_mean={np.mean(t):.5f}  auroc={auroc:.4f}")

    out_path = REPO / "results" / "ftt_badvla_text2img_flatten_normalize_order.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
