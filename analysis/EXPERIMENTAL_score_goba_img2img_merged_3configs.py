#!/usr/bin/env python
"""ONE-OFF, ISOLATED experiment -- run this project's 3 standard configs
(per-layer AUROC, Flatten method, Shared-reference method -- the exact ones
already reported for text2img) on img2img and on merged
(image+text)->(image+text) attention for GoBA.

Reads results/EXPERIMENTAL_goba_img2img_merged_alllayers/*.npz (written by
the sibling extraction script
EXPERIMENTAL_extract_img2img_merged_alllayers.py). Isolated: does not
modify detectors/ftt.py or any other scoring script.

Merged convention (matches this project's own established
extract_img2img_merged_ftt.py exactly): "merged" is TWO separate FTT
statistics -- text-group (query=desc tokens, key=image+desc columns
combined) and image-group (query=image patches, key=image+desc columns
combined) -- reported and scored SEPARATELY, since they have different
query sets. combined = the plain average of the two, same as the existing
merged_combined_ftt convention.

IMPORTANT CAVEAT carried through every merged number here: image patches
occur BEFORE text in the sequence, so under real causal masking the
image-group's key columns that fall in the text-token range are
STRUCTURALLY GUARANTEED TO BE EXACTLY ZERO (a later token cannot be
attended to by an earlier one). That's not a bug in this extraction --
it's the same causal fact that motivated the eager-vs-sdpa fix earlier this
project -- but it means image-group's key space is really only half real
information (the image columns) and half a constant-zero block (the text
columns), diluting whatever text-group's key space is also getting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "EXPERIMENTAL_goba_img2img_merged_alllayers"
BALANCED_63V63_JSON = REPO / "results" / "ftt_goba_text2img_layerwise_63v63_balanced.json"
EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def ftt_on_rows(rows: np.ndarray) -> float:
    ref = rows.mean(axis=0)
    return float(np.linalg.norm(rows - ref[None, :], axis=1).mean())


def per_layer_ftt(layers_normed: np.ndarray) -> np.ndarray:
    """layers_normed: [L, rows, cols], ALREADY row-normalized. Returns [L]."""
    return np.array([ftt_on_rows(layers_normed[l]) for l in range(layers_normed.shape[0])])


def flatten_normalize_then_average(layers_raw: np.ndarray) -> float:
    normed = row_normalize(layers_raw)
    avg_map = normed.mean(axis=0)
    return ftt_on_rows(avg_map)


def flatten_average_then_normalize(layers_raw: np.ndarray) -> float:
    avg_map = layers_raw.mean(axis=0)
    avg_map = row_normalize(avg_map)
    return ftt_on_rows(avg_map)


def sharedref_normalize_then_average(layers_raw: np.ndarray) -> float:
    P = row_normalize(layers_raw)          # [L, rows, cols]
    per_layer_mean = P.mean(axis=1)        # [L, cols]
    ref = per_layer_mean.mean(axis=0)      # [cols]
    per_row = np.linalg.norm(P - ref[None, None, :], axis=-1)  # [L, rows]
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


def score_key(key_name: str, files, clean_63_keys, trig_63_keys):
    """key_name: 'img2img_raw', 'merged_image_raw', or 'merged_text_raw'."""
    clean_layers, trig_layers = {}, {}  # key -> [L, rows, cols] raw

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if str(d["role"]) != "attack":
            continue
        layers_raw = d[key_name].astype(np.float64)
        key = (int(d["task_id"]), int(d["seed"]))
        (clean_layers if int(d["label"]) == 0 else trig_layers)[key] = layers_raw

    assert len(clean_layers) == 100 and len(trig_layers) == 100, (key_name, len(clean_layers), len(trig_layers))

    # --- (1) per-layer AUROC ---
    clean_pl = np.stack([per_layer_ftt(row_normalize(v)) for v in clean_layers.values()])  # [100, L]
    trig_pl = np.stack([per_layer_ftt(row_normalize(v)) for v in trig_layers.values()])
    n_layers = clean_pl.shape[1]
    auroc_pl_100 = np.array([rank_auroc(clean_pl[:, l], trig_pl[:, l]) for l in range(n_layers)])

    keys_sorted = sorted(clean_layers.keys())
    idx_by_key = {k: i for i, k in enumerate(keys_sorted)}
    c63_idx = [idx_by_key[k] for k in sorted(clean_63_keys) if k in idx_by_key]
    keys_sorted_t = sorted(trig_layers.keys())
    idx_by_key_t = {k: i for i, k in enumerate(keys_sorted_t)}
    t63_idx = [idx_by_key_t[k] for k in sorted(trig_63_keys) if k in idx_by_key_t]
    clean_pl_ordered = np.stack([per_layer_ftt(row_normalize(clean_layers[k])) for k in keys_sorted])
    trig_pl_ordered = np.stack([per_layer_ftt(row_normalize(trig_layers[k])) for k in keys_sorted_t])
    auroc_pl_63 = np.array([
        rank_auroc(clean_pl_ordered[c63_idx, l], trig_pl_ordered[t63_idx, l]) for l in range(n_layers)])

    # --- (2) Flatten method ---
    c_na = {k: flatten_normalize_then_average(v) for k, v in clean_layers.items()}
    t_na = {k: flatten_normalize_then_average(v) for k, v in trig_layers.items()}
    c_an = {k: flatten_average_then_normalize(v) for k, v in clean_layers.items()}
    t_an = {k: flatten_average_then_normalize(v) for k, v in trig_layers.items()}

    auroc_flat_na_100 = rank_auroc(list(c_na.values()), list(t_na.values()))
    auroc_flat_na_63 = rank_auroc([c_na[k] for k in sorted(clean_63_keys)], [t_na[k] for k in sorted(trig_63_keys)])
    auroc_flat_an_100 = rank_auroc(list(c_an.values()), list(t_an.values()))
    auroc_flat_an_63 = rank_auroc([c_an[k] for k in sorted(clean_63_keys)], [t_an[k] for k in sorted(trig_63_keys)])

    # --- (3) Shared-reference method ---
    c_sna = {k: sharedref_normalize_then_average(v) for k, v in clean_layers.items()}
    t_sna = {k: sharedref_normalize_then_average(v) for k, v in trig_layers.items()}
    c_san = {k: sharedref_average_then_normalize(v) for k, v in clean_layers.items()}
    t_san = {k: sharedref_average_then_normalize(v) for k, v in trig_layers.items()}

    auroc_shared_na_100 = rank_auroc(list(c_sna.values()), list(t_sna.values()))
    auroc_shared_na_63 = rank_auroc([c_sna[k] for k in sorted(clean_63_keys)], [t_sna[k] for k in sorted(trig_63_keys)])
    auroc_shared_an_100 = rank_auroc(list(c_san.values()), list(t_san.values()))
    auroc_shared_an_63 = rank_auroc([c_san[k] for k in sorted(clean_63_keys)], [t_san[k] for k in sorted(trig_63_keys)])

    return {
        "per_layer_100v100": auroc_pl_100.tolist(),
        "per_layer_63v63": auroc_pl_63.tolist(),
        "flatten_normalize_then_average": {"auroc_100v100": auroc_flat_na_100, "auroc_63v63": auroc_flat_na_63},
        "flatten_average_then_normalize": {"auroc_100v100": auroc_flat_an_100, "auroc_63v63": auroc_flat_an_63},
        "sharedref_normalize_then_average": {"auroc_100v100": auroc_shared_na_100, "auroc_63v63": auroc_shared_na_63},
        "sharedref_average_then_normalize": {"auroc_100v100": auroc_shared_an_100, "auroc_63v63": auroc_shared_an_63},
    }


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")

    with open(BALANCED_63V63_JSON) as f:
        balanced = json.load(f)
    clean_63_keys = {(tid, seed) for tid, seed in balanced["chosen_clean_pairs"]}
    trig_63_keys = {(tid, seed) for tid, seed in balanced["trigger_pairs"]}

    out = {}
    for key_name, label in [("img2img_raw", "img2img"), ("merged_image_raw", "merged_image_group"),
                             ("merged_text_raw", "merged_text_group")]:
        print(f"\n[*] === {label} ===")
        result = score_key(key_name, files, clean_63_keys, trig_63_keys)
        out[label] = result
        best_l = int(np.argmax(result["per_layer_100v100"]))
        print(f"    best per-layer (layer {best_l}): "
              f"100v100={result['per_layer_100v100'][best_l]:.4f}  "
              f"63v63={result['per_layer_63v63'][best_l]:.4f}")
        print(f"    Flatten, normalize-then-average: 100v100={result['flatten_normalize_then_average']['auroc_100v100']:.4f}  "
              f"63v63={result['flatten_normalize_then_average']['auroc_63v63']:.4f}")
        print(f"    Flatten, average-then-normalize: 100v100={result['flatten_average_then_normalize']['auroc_100v100']:.4f}  "
              f"63v63={result['flatten_average_then_normalize']['auroc_63v63']:.4f}")
        print(f"    Shared-ref, normalize-then-average: 100v100={result['sharedref_normalize_then_average']['auroc_100v100']:.4f}  "
              f"63v63={result['sharedref_normalize_then_average']['auroc_63v63']:.4f}")
        print(f"    Shared-ref, average-then-normalize: 100v100={result['sharedref_average_then_normalize']['auroc_100v100']:.4f}  "
              f"63v63={result['sharedref_average_then_normalize']['auroc_63v63']:.4f}")

    out_path = REPO / "results" / "EXPERIMENTAL_ftt_goba_img2img_merged_3configs.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
