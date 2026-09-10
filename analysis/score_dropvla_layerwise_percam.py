#!/usr/bin/env python
"""The 'flatten, normalize-then-average' FTT construction on DropVLA
text2img desc_only attention: row-normalize each layer's map separately,
average those already-normalized maps into one map, then run FTT
(Frobenius-norm distance from each row to the map's own row-mean) on that
one map. Scored against AUROC, run separately for the primary and wrist
cameras.

Reads results/dropvla_text2img_layerwise_percam/*.npz (written by the
sibling extraction script
extract_text2img_layerwise_percam_ftt.py). 100 clean vs 100 trigger only
(no ASR-success-only subset exists yet for DropVLA).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "dropvla_text2img_layerwise_percam"
EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def ftt_on_rows(rows: np.ndarray) -> float:
    ref = rows.mean(axis=0)
    return float(np.linalg.norm(rows - ref[None, :], axis=1).mean())


def flatten_normalize_then_average(layers_raw: np.ndarray) -> float:
    normed = row_normalize(layers_raw)
    avg_map = normed.mean(axis=0)
    return ftt_on_rows(avg_map)


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


def score_camera(cam_key: str, files):
    clean_layers, trig_layers = {}, {}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if str(d["role"]) != "attack":
            continue
        layers = d[cam_key].astype(np.float64)  # [L, n_desc, num_patches]
        key = (int(d["task_id"]), int(d["seed"]))
        (clean_layers if int(d["label"]) == 0 else trig_layers)[key] = layers

    n_clean, n_trig = len(clean_layers), len(trig_layers)

    c_na = [flatten_normalize_then_average(v) for v in clean_layers.values()]
    t_na = [flatten_normalize_then_average(v) for v in trig_layers.values()]
    auroc_flat_na = rank_auroc(c_na, t_na)

    return {
        "n_clean": n_clean, "n_trig": n_trig,
        "flatten_normalize_then_average": auroc_flat_na,
    }


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")

    out = {}
    for cam_key, label in [("attn_primary_layers", "primary"), ("attn_wrist_layers", "wrist")]:
        print(f"\n[*] === {label} camera ===")
        result = score_camera(cam_key, files)
        out[label] = result
        print(f"    n_clean={result['n_clean']} n_trig={result['n_trig']}")
        print(f"    Flatten, normalize-then-average: {result['flatten_normalize_then_average']:.4f}")

    out_path = REPO / "results" / "ftt_dropvla_text2img_layerwise_percam.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
