#!/usr/bin/env python
"""DropVLA equivalent of the GoBA report's sections 6/7: img2img (each
camera scored separately) and the merged single-combined-query block.

Reads results/dropvla_img2img_merged_percam/*.npz (written by the sibling
extraction script extract_img2img_merged_percam_ftt.py). 100 clean vs 100
trigger.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "dropvla_img2img_merged_percam"
EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def ftt_on_rows(rows: np.ndarray) -> float:
    ref = rows.mean(axis=0)
    return float(np.linalg.norm(rows - ref[None, :], axis=1).mean())


def per_layer_ftt(layers_raw: np.ndarray) -> np.ndarray:
    normed = row_normalize(layers_raw)
    return np.array([ftt_on_rows(normed[l]) for l in range(normed.shape[0])])


def flatten_normalize_then_average(layers_raw: np.ndarray) -> float:
    normed = row_normalize(layers_raw)
    avg_map = normed.mean(axis=0)
    return ftt_on_rows(avg_map)


def flatten_average_then_normalize(layers_raw: np.ndarray) -> float:
    avg_map = layers_raw.mean(axis=0)
    avg_map = row_normalize(avg_map)
    return ftt_on_rows(avg_map)


def sharedref_normalize_then_average(layers_raw: np.ndarray) -> float:
    P = row_normalize(layers_raw)
    per_layer_mean = P.mean(axis=1)
    ref = per_layer_mean.mean(axis=0)
    per_row = np.linalg.norm(P - ref[None, None, :], axis=-1)
    return float(per_row.mean(axis=1).mean())


def sharedref_average_then_normalize(layers_raw: np.ndarray) -> float:
    ref_raw = layers_raw.mean(axis=(0, 1))
    ref = ref_raw / max(ref_raw.sum(), EPS)
    P = row_normalize(layers_raw)
    per_row = np.linalg.norm(P - ref[None, None, :], axis=-1)
    return float(per_row.mean(axis=1).mean())


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


def score_key(key_name: str, files):
    clean_layers, trig_layers = {}, {}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if str(d["role"]) != "attack":
            continue
        layers = d[key_name].astype(np.float64)
        key = (int(d["task_id"]), int(d["seed"]))
        (clean_layers if int(d["label"]) == 0 else trig_layers)[key] = layers

    assert len(clean_layers) == 100 and len(trig_layers) == 100, (key_name, len(clean_layers), len(trig_layers))

    clean_pl = np.stack([per_layer_ftt(v) for v in clean_layers.values()])
    trig_pl = np.stack([per_layer_ftt(v) for v in trig_layers.values()])
    n_layers = clean_pl.shape[1]
    auroc_pl = np.array([rank_auroc(clean_pl[:, l], trig_pl[:, l]) for l in range(n_layers)])
    best_l = int(np.argmax(auroc_pl))

    c_na = [flatten_normalize_then_average(v) for v in clean_layers.values()]
    t_na = [flatten_normalize_then_average(v) for v in trig_layers.values()]
    c_an = [flatten_average_then_normalize(v) for v in clean_layers.values()]
    t_an = [flatten_average_then_normalize(v) for v in trig_layers.values()]
    auroc_flat_na = rank_auroc(c_na, t_na)
    auroc_flat_an = rank_auroc(c_an, t_an)

    c_sna = [sharedref_normalize_then_average(v) for v in clean_layers.values()]
    t_sna = [sharedref_normalize_then_average(v) for v in trig_layers.values()]
    c_san = [sharedref_average_then_normalize(v) for v in clean_layers.values()]
    t_san = [sharedref_average_then_normalize(v) for v in trig_layers.values()]
    auroc_shared_na = rank_auroc(c_sna, t_sna)
    auroc_shared_an = rank_auroc(c_san, t_san)

    return {
        "n_layers": n_layers,
        "per_layer_auroc": auroc_pl.tolist(),
        "best_layer": best_l, "best_layer_auroc": float(auroc_pl[best_l]),
        "flatten_normalize_then_average": auroc_flat_na,
        "flatten_average_then_normalize": auroc_flat_an,
        "sharedref_normalize_then_average": auroc_shared_na,
        "sharedref_average_then_normalize": auroc_shared_an,
    }


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")

    out = {}
    for key_name, label in [("primary_primary", "primary_img2img"),
                             ("wrist_wrist", "wrist_img2img"),
                             ("combined", "merged_combined")]:
        print(f"\n[*] === {label} ===")
        result = score_key(key_name, files)
        out[label] = result
        print(f"    best layer {result['best_layer']}: AUROC={result['best_layer_auroc']:.4f}")
        print(f"    Flatten, normalize-then-average: {result['flatten_normalize_then_average']:.4f}")
        print(f"    Flatten, average-then-normalize: {result['flatten_average_then_normalize']:.4f}")
        print(f"    Shared-ref, normalize-then-average: {result['sharedref_normalize_then_average']:.4f}")
        print(f"    Shared-ref, average-then-normalize: {result['sharedref_average_then_normalize']:.4f}")

    out_path = REPO / "results" / "ftt_dropvla_img2img_merged_percam.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
