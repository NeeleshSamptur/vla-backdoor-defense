#!/usr/bin/env python
"""ONE-OFF, ISOLATED experiment -- {BOS, desc_only tokens, sentinel 29871}
combined as one query set, scored with the same "flatten to one map, two
normalization orders" construction as section 2 of the main report.

Merges two already-extracted, independently-produced datasets by
(task_id, seed) key:
  - results/goba_text2img_layerwise_causal_fixed/*.npz -- the standard
    desc_only rows (attn_text_image_layers, [32, n_desc, P]), role=='attack'
  - results/EXPERIMENTAL_goba_bos_sentinel/*.npz -- BOS's row and the
    sentinel token's (29871) row (bos_row, sentinel_row, each [32, P])

Combined per-layer query becomes [BOS_row; desc_row_0; ...; desc_row_{n-1};
sentinel_row] -- BOS first (its real sequence position), sentinel last
(its real sequence position), desc tokens in their existing order in
between -- so token ORDER in the combined query set matches their real
order in the actual sequence.

Isolated: new file, does not modify detectors/ftt.py or any other scoring
script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DESC_DATA_DIR = REPO / "results" / "goba_text2img_layerwise_causal_fixed"
BOS_DATA_DIR = REPO / "results" / "EXPERIMENTAL_goba_bos_sentinel"
BALANCED_63V63_JSON = REPO / "results" / "ftt_goba_text2img_layerwise_63v63_balanced.json"
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
        bos_lookup[key] = (d["bos_row"].astype(np.float64), d["sentinel_row"].astype(np.float64))

    with open(BALANCED_63V63_JSON) as f:
        balanced = json.load(f)
    clean_63_keys = {(tid, seed) for tid, seed in balanced["chosen_clean_pairs"]}
    trig_63_keys = {(tid, seed) for tid, seed in balanced["trigger_pairs"]}

    clean_na, trig_na = {}, {}
    clean_an, trig_an = {}, {}
    n_matched = 0

    for fp in desc_files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        if meta["extra"]["role"] != "attack":
            continue
        key = (meta["task_id"], meta["seed"])
        if key not in bos_lookup:
            continue
        n_matched += 1
        desc_layers = d["attn_text_image_layers"].astype(np.float64)  # [32, n_desc, P]
        bos_row, sentinel_row = bos_lookup[key]                       # each [32, P]

        combined = np.concatenate(
            [bos_row[:, None, :], desc_layers, sentinel_row[:, None, :]], axis=1)  # [32, n_desc+2, P]

        label = meta["label"]
        bucket_na = clean_na if label == 0 else trig_na
        bucket_an = clean_an if label == 0 else trig_an
        bucket_na[key] = score_normalize_then_average(combined)
        bucket_an[key] = score_average_then_normalize(combined)

    print(f"[*] {n_matched} episodes matched across both datasets")
    assert len(clean_na) == 100 and len(trig_na) == 100, (len(clean_na), len(trig_na))

    c100_na, t100_na = list(clean_na.values()), list(trig_na.values())
    c100_an, t100_an = list(clean_an.values()), list(trig_an.values())
    c63_na = [clean_na[k] for k in sorted(clean_63_keys)]
    t63_na = [trig_na[k] for k in sorted(trig_63_keys)]
    c63_an = [clean_an[k] for k in sorted(clean_63_keys)]
    t63_an = [trig_an[k] for k in sorted(trig_63_keys)]

    auroc_na_100 = rank_auroc(c100_na, t100_na)
    auroc_na_63 = rank_auroc(c63_na, t63_na)
    auroc_an_100 = rank_auroc(c100_an, t100_an)
    auroc_an_63 = rank_auroc(c63_an, t63_an)

    print(f"\n[*] BOS+desc+sentinel, normalize-then-average: 100v100={auroc_na_100:.4f}  63v63={auroc_na_63:.4f}")
    print(f"[*] BOS+desc+sentinel, average-then-normalize: 100v100={auroc_an_100:.4f}  63v63={auroc_an_63:.4f}")

    out = {
        "normalize_then_average": {"auroc_100v100": auroc_na_100, "auroc_63v63": auroc_na_63},
        "average_then_normalize": {"auroc_100v100": auroc_an_100, "auroc_63v63": auroc_an_63},
    }
    out_path = REPO / "results" / "EXPERIMENTAL_ftt_goba_bos_desc_sentinel_auroc.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
