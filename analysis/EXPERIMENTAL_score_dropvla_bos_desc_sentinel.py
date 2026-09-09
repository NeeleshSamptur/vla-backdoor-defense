#!/usr/bin/env python
"""ONE-OFF, ISOLATED experiment -- DropVLA equivalent of
analysis/EXPERIMENTAL_score_goba_bos_desc_sentinel.py: {BOS, desc_only
tokens, sentinel 29871} combined as one query set, scored with the same
"flatten to one map, two normalization orders" construction used throughout
this project -- run separately per camera (primary, wrist).

Merges two already-extracted, independently-produced datasets by
(task_id, seed) key:
  - results/dropvla_text2img_layerwise_percam/*.npz -- the standard
    desc_only rows (attn_primary_layers / attn_wrist_layers, each
    [32, n_desc, P]), role=='attack'
  - results/EXPERIMENTAL_dropvla_bos_sentinel/*.npz -- BOS's and the
    sentinel token's (29871) rows per camera (bos_primary, bos_wrist,
    sentinel_primary, sentinel_wrist, each [32, P])

Combined per-layer, per-camera query becomes [BOS_row; desc_row_0; ...;
desc_row_{n-1}; sentinel_row] -- BOS first (its real sequence position),
sentinel last (its real sequence position), desc tokens in their existing
order in between -- matching GoBA's ordering convention exactly.

Comparison convention: DropVLA's existing report has no 63v63
successful-attack-only subset (confirmed: results/ftt_dropvla_text2img_layerwise_percam.json
carries only n_clean=100/n_trig=100 fields, no balanced-63 variant) --
so this script reports 100v100 only, matching what
analysis/score_dropvla_layerwise_percam.py already established as the
DropVLA convention.

Isolated: new file, does not modify detectors/ftt.py or any other scoring
script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DESC_DATA_DIR = REPO / "results" / "dropvla_text2img_layerwise_percam"
BOS_DATA_DIR = REPO / "results" / "EXPERIMENTAL_dropvla_bos_sentinel"
EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def ftt_on_rows(rows: np.ndarray) -> float:
    ref = rows.mean(axis=0)
    return float(np.linalg.norm(rows - ref[None, :], axis=1).mean())


def score_normalize_then_average(layers: np.ndarray) -> float:
    normed = row_normalize(layers)
    avg_map = normed.mean(axis=0)
    return ftt_on_rows(avg_map)


def score_average_then_normalize(layers: np.ndarray) -> float:
    avg_map = layers.mean(axis=0)
    avg_map = row_normalize(avg_map)
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


def score_camera(desc_key: str, bos_key: str, sentinel_key: str, desc_files, bos_lookup):
    clean_na, trig_na = {}, {}
    clean_an, trig_an = {}, {}
    n_matched = 0

    for fp in desc_files:
        d = np.load(fp, allow_pickle=True)
        if str(d["role"]) != "attack":
            continue
        key = (int(d["task_id"]), int(d["seed"]))
        if key not in bos_lookup:
            continue
        n_matched += 1
        desc_layers = d[desc_key].astype(np.float64)  # [32, n_desc, P]
        bos_row, sentinel_row = bos_lookup[key][bos_key], bos_lookup[key][sentinel_key]  # each [32, P]

        combined = np.concatenate(
            [bos_row[:, None, :], desc_layers, sentinel_row[:, None, :]], axis=1)  # [32, n_desc+2, P]

        label = int(d["label"])
        bucket_na = clean_na if label == 0 else trig_na
        bucket_an = clean_an if label == 0 else trig_an
        bucket_na[key] = score_normalize_then_average(combined)
        bucket_an[key] = score_average_then_normalize(combined)

    return clean_na, trig_na, clean_an, trig_an, n_matched


def main():
    desc_files = sorted(DESC_DATA_DIR.glob("*.npz"))
    bos_files = sorted(BOS_DATA_DIR.glob("*.npz"))
    print(f"[*] {len(desc_files)} desc files, {len(bos_files)} bos/sentinel files")

    bos_lookup = {}
    for fp in bos_files:
        d = np.load(fp, allow_pickle=True)
        if str(d["role"]) != "attack":
            continue
        key = (int(d["task_id"]), int(d["seed"]))
        bos_lookup[key] = {
            "bos_primary": d["bos_primary"].astype(np.float64),
            "bos_wrist": d["bos_wrist"].astype(np.float64),
            "sentinel_primary": d["sentinel_primary"].astype(np.float64),
            "sentinel_wrist": d["sentinel_wrist"].astype(np.float64),
        }

    out = {}
    for cam_label, desc_key, bos_key, sentinel_key in (
        ("primary", "attn_primary_layers", "bos_primary", "sentinel_primary"),
        ("wrist", "attn_wrist_layers", "bos_wrist", "sentinel_wrist"),
    ):
        print(f"\n[*] === {cam_label} camera ===")
        clean_na, trig_na, clean_an, trig_an, n_matched = score_camera(
            desc_key, bos_key, sentinel_key, desc_files, bos_lookup)
        print(f"[*] {n_matched} episodes matched across both datasets")
        assert len(clean_na) == 100 and len(trig_na) == 100, (len(clean_na), len(trig_na))

        c100_na, t100_na = list(clean_na.values()), list(trig_na.values())
        c100_an, t100_an = list(clean_an.values()), list(trig_an.values())

        auroc_na_100 = rank_auroc(c100_na, t100_na)
        auroc_an_100 = rank_auroc(c100_an, t100_an)

        print(f"    BOS+desc+sentinel, normalize-then-average: 100v100={auroc_na_100:.4f}")
        print(f"    BOS+desc+sentinel, average-then-normalize: 100v100={auroc_an_100:.4f}")

        out[cam_label] = {
            "n_clean": len(clean_na), "n_trig": len(trig_na), "n_matched": n_matched,
            "normalize_then_average": {"auroc_100v100": auroc_na_100},
            "average_then_normalize": {"auroc_100v100": auroc_an_100},
        }

    out_path = REPO / "results" / "ftt_dropvla_bos_desc_sentinel.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
