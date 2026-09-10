#!/usr/bin/env python
"""The 'flatten, normalize-then-average' FTT construction on Pi0-Fast's
image+text (TI4) trigger variant, text2img desc_only attention: row-normalize
each layer's map separately, average those already-normalized maps into one
map, then run FTT (Frobenius-norm distance from each row to the map's own
row-mean) on that one map. Scored against AUROC.

Reads results/pi0fast_ti4_fullbattery_extracted/*.npz -- written by
adapters/pi0fast_backdoorvla/extract_ti4_fullbattery.py, using the FIXED
attention_prefix() (left_to_right_align() bug fix).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "pi0fast_ti4_fullbattery_extracted"
OUT_PATH = REPO / "results" / "pi0fast_ti4_fullbattery_auroc.json"
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
    return float((ranks[n_c:].sum() - n_t * (n_t + 1) / 2) / (n_c * n_t))


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")
    if not files:
        raise FileNotFoundError(f"no files in {DATA_DIR} -- run extract_ti4_fullbattery.py first")

    clean_t2i, trig_t2i = {}, {}  # attn_text_image_layers [L, n_desc, per_cam]

    for fp in files:
        d = np.load(fp, allow_pickle=False)
        label = int(d["label"])
        key = (int(d["task_id"]), int(d["seed"]))
        t2i = d["attn_text_image_layers"].astype(np.float64)
        bucket_t2i = clean_t2i if label == 0 else trig_t2i
        bucket_t2i[key] = t2i

    n_clean, n_trig = len(clean_t2i), len(trig_t2i)
    print(f"[*] n_clean={n_clean} n_trig={n_trig}")

    c_na = {k: flatten_normalize_then_average(v) for k, v in clean_t2i.items()}
    t_na = {k: flatten_normalize_then_average(v) for k, v in trig_t2i.items()}
    auroc_na = rank_auroc(list(c_na.values()), list(t_na.values()))

    out = {
        "n_clean": n_clean, "n_trig": n_trig,
        "flatten_normalize_then_average_text2img": auroc_na,
        "clean_mean": float(np.mean(list(c_na.values()))),
        "trig_mean": float(np.mean(list(t_na.values()))),
    }
    print(f"\n[*] Flatten, normalize-then-average (text2img desc_only): AUROC={auroc_na:.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
