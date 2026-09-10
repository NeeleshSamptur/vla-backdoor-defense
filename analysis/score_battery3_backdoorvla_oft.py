#!/usr/bin/env python
"""The 'flatten, normalize-then-average' FTT construction on BackdoorVLA-OFT
text2img desc_only attention: row-normalize each layer's head-averaged map
separately, average those already-normalized maps into one map, then run FTT
(Frobenius-norm distance from each row to the map's own row-mean) on that
one map. Scored against AUROC.

Reads results/oft_battery3_fresh/*.npz (extract_battery3_fresh.py) under the
already-fixed eager-attention loader.

This adapter's episode convention (confirmed from extract_text2img_
alllayers_driver.py / extract_img2img_and_merged_ftt.py, both pre-existing):
9 non-target task_ids x 10 seeds = 90 clean + 90 trigger. No ASR-success-only
subset exists for this checkpoint, so this is a single 90-clean-vs-90-trigger
AUROC, matching DropVLA's pattern.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import numpy as np

DATA_DIR = REPO / "results" / "oft_battery3_fresh"
OUT_DIR = REPO / "results"
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


def load_all():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")
    episodes = []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        episodes.append(dict(
            fp=fp,
            label=int(d["label"]), task_id=int(d["task_id"]), seed=int(d["seed"]),
            t2i_perhead=d["t2i_perhead"].astype(np.float64),   # [L,H,1+n_txt,P]
            txt_rel_desc=d["txt_rel_desc"].tolist(),
            n_txt=int(d["n_txt"]), task_description=str(d["task_description"]),
        ))
    n_clean = sum(1 for e in episodes if e["label"] == 0)
    n_trig = sum(1 for e in episodes if e["label"] == 1)
    print(f"[*] n_clean={n_clean} n_trig={n_trig}")
    assert n_clean == 90 and n_trig == 90, (n_clean, n_trig)
    return episodes


def desc_rows(ep):
    """[L, n_desc, P] head-averaged, desc_only rows, from t2i_perhead (row0=BOS)."""
    t2i_avg = ep["t2i_perhead"].mean(axis=1)  # [L, 1+n_txt, P]
    idx = [1 + r for r in ep["txt_rel_desc"]]
    return t2i_avg[:, idx, :]


def main():
    episodes = load_all()
    clean = [e for e in episodes if e["label"] == 0]
    trig = [e for e in episodes if e["label"] == 1]

    c_na = [flatten_normalize_then_average(desc_rows(e)) for e in clean]
    t_na = [flatten_normalize_then_average(desc_rows(e)) for e in trig]
    auroc_na = rank_auroc(c_na, t_na)

    out = {
        "n_clean": 90, "n_trig": 90,
        "flatten_normalize_then_average_text2img_desc_only": auroc_na,
        "clean_mean": float(np.mean(c_na)),
        "trig_mean": float(np.mean(t_na)),
    }
    print(f"\n[*] Flatten, normalize-then-average (desc-only text2img): AUROC={auroc_na:.4f}")

    out_path = OUT_DIR / "ftt_battery3_backdoorvla_oft_full.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
