#!/usr/bin/env python
"""ONE-OFF, ISOLATED experiment -- the LITERAL single (image+text)->(image+
text) query block: unlike section 7's image-group/text-group (scored
SEPARATELY, each with its own query set but a shared key space), this
stacks ALL rows -- every image-patch row AND every desc-token row -- into
ONE combined query set, sharing the same combined key columns
(image patches + desc tokens), and runs a SINGLE FTT computation across all
of them together: one reference, one set of row-to-reference distances
spanning both image rows and text rows at once.

Reuses the already-extracted results/EXPERIMENTAL_goba_img2img_merged_alllayers/*.npz
(merged_image_raw [L, 256, K] and merged_text_raw [L, n_desc, K], same K key
columns per episode) -- no new GPU extraction needed, just concatenate the
two row-blocks per episode along the row axis.

CAUSAL-MASKING CAVEAT (also noted in the artifact): image patches occur
BEFORE text in the sequence. So every image-row's key-columns that fall in
the text range are EXACTLY ZERO (a later token cannot be attended to by an
earlier one) -- real, not a bug, same fact behind the eager-vs-sdpa fix
earlier this project. Roughly the top ~256/(256+n_desc) of this combined
query block therefore carries a constant zero sub-vector that the bottom
~n_desc/(256+n_desc) (text rows) does not; this experiment quantifies what
happens when both are forced into ONE shared reference/statistic anyway.

Isolated: new file, does not modify detectors/ftt.py or any other scoring
script.
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
    return np.array([ftt_on_rows(layers_normed[l]) for l in range(layers_normed.shape[0])])


def flatten_normalize_then_average(layers: np.ndarray) -> float:
    normed = row_normalize(layers)
    avg_map = normed.mean(axis=0)
    return ftt_on_rows(avg_map)


def flatten_average_then_normalize(layers: np.ndarray) -> float:
    avg_map = layers.mean(axis=0)
    avg_map = row_normalize(avg_map)
    return ftt_on_rows(avg_map)


def sharedref_normalize_then_average(layers: np.ndarray) -> float:
    P = row_normalize(layers)
    per_layer_mean = P.mean(axis=1)
    ref = per_layer_mean.mean(axis=0)
    per_row = np.linalg.norm(P - ref[None, None, :], axis=-1)
    return float(per_row.mean(axis=1).mean())


def sharedref_average_then_normalize(layers: np.ndarray) -> float:
    ref_raw = layers.mean(axis=(0, 1))
    ref = ref_raw / max(ref_raw.sum(), EPS)
    P = row_normalize(layers)
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


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")

    with open(BALANCED_63V63_JSON) as f:
        balanced = json.load(f)
    clean_63_keys = {(tid, seed) for tid, seed in balanced["chosen_clean_pairs"]}
    trig_63_keys = {(tid, seed) for tid, seed in balanced["trigger_pairs"]}

    clean_layers, trig_layers = {}, {}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if str(d["role"]) != "attack":
            continue
        combined = np.concatenate(
            [d["merged_image_raw"].astype(np.float64), d["merged_text_raw"].astype(np.float64)], axis=1)
        key = (int(d["task_id"]), int(d["seed"]))
        (clean_layers if int(d["label"]) == 0 else trig_layers)[key] = combined

    assert len(clean_layers) == 100 and len(trig_layers) == 100, (len(clean_layers), len(trig_layers))
    sample_shape = next(iter(clean_layers.values())).shape
    print(f"[*] combined query block shape (one episode): {sample_shape}  "
          f"(256 image rows + {sample_shape[1]-256} desc rows = {sample_shape[1]} total)")

    # --- per-layer AUROC ---
    keys_c = sorted(clean_layers.keys()); keys_t = sorted(trig_layers.keys())
    clean_pl = np.stack([per_layer_ftt(row_normalize(clean_layers[k])) for k in keys_c])
    trig_pl = np.stack([per_layer_ftt(row_normalize(trig_layers[k])) for k in keys_t])
    n_layers = clean_pl.shape[1]
    auroc_pl_100 = np.array([rank_auroc(clean_pl[:, l], trig_pl[:, l]) for l in range(n_layers)])

    idx_c = {k: i for i, k in enumerate(keys_c)}
    idx_t = {k: i for i, k in enumerate(keys_t)}
    c63_idx = [idx_c[k] for k in sorted(clean_63_keys)]
    t63_idx = [idx_t[k] for k in sorted(trig_63_keys)]
    auroc_pl_63 = np.array([rank_auroc(clean_pl[c63_idx, l], trig_pl[t63_idx, l]) for l in range(n_layers)])

    best_l = int(np.argmax(auroc_pl_100))
    print(f"\n[*] Per-layer AUROC (combined query), best layer {best_l}: "
          f"100v100={auroc_pl_100[best_l]:.4f}  63v63={auroc_pl_63[best_l]:.4f}")

    # --- Flatten method ---
    c_na = {k: flatten_normalize_then_average(v) for k, v in clean_layers.items()}
    t_na = {k: flatten_normalize_then_average(v) for k, v in trig_layers.items()}
    c_an = {k: flatten_average_then_normalize(v) for k, v in clean_layers.items()}
    t_an = {k: flatten_average_then_normalize(v) for k, v in trig_layers.items()}
    auroc_flat_na_100 = rank_auroc(list(c_na.values()), list(t_na.values()))
    auroc_flat_na_63 = rank_auroc([c_na[k] for k in sorted(clean_63_keys)], [t_na[k] for k in sorted(trig_63_keys)])
    auroc_flat_an_100 = rank_auroc(list(c_an.values()), list(t_an.values()))
    auroc_flat_an_63 = rank_auroc([c_an[k] for k in sorted(clean_63_keys)], [t_an[k] for k in sorted(trig_63_keys)])
    print(f"[*] Flatten, normalize-then-average: 100v100={auroc_flat_na_100:.4f}  63v63={auroc_flat_na_63:.4f}")
    print(f"[*] Flatten, average-then-normalize: 100v100={auroc_flat_an_100:.4f}  63v63={auroc_flat_an_63:.4f}")

    # --- Shared-reference method ---
    c_sna = {k: sharedref_normalize_then_average(v) for k, v in clean_layers.items()}
    t_sna = {k: sharedref_normalize_then_average(v) for k, v in trig_layers.items()}
    c_san = {k: sharedref_average_then_normalize(v) for k, v in clean_layers.items()}
    t_san = {k: sharedref_average_then_normalize(v) for k, v in trig_layers.items()}
    auroc_sh_na_100 = rank_auroc(list(c_sna.values()), list(t_sna.values()))
    auroc_sh_na_63 = rank_auroc([c_sna[k] for k in sorted(clean_63_keys)], [t_sna[k] for k in sorted(trig_63_keys)])
    auroc_sh_an_100 = rank_auroc(list(c_san.values()), list(t_san.values()))
    auroc_sh_an_63 = rank_auroc([c_san[k] for k in sorted(clean_63_keys)], [t_san[k] for k in sorted(trig_63_keys)])
    print(f"[*] Shared-ref, normalize-then-average: 100v100={auroc_sh_na_100:.4f}  63v63={auroc_sh_na_63:.4f}")
    print(f"[*] Shared-ref, average-then-normalize: 100v100={auroc_sh_an_100:.4f}  63v63={auroc_sh_an_63:.4f}")

    out = {
        "combined_row_count": int(sample_shape[1]),
        "per_layer_100v100": auroc_pl_100.tolist(),
        "per_layer_63v63": auroc_pl_63.tolist(),
        "flatten_normalize_then_average": {"auroc_100v100": auroc_flat_na_100, "auroc_63v63": auroc_flat_na_63},
        "flatten_average_then_normalize": {"auroc_100v100": auroc_flat_an_100, "auroc_63v63": auroc_flat_an_63},
        "sharedref_normalize_then_average": {"auroc_100v100": auroc_sh_na_100, "auroc_63v63": auroc_sh_na_63},
        "sharedref_average_then_normalize": {"auroc_100v100": auroc_sh_an_100, "auroc_63v63": auroc_sh_an_63},
    }
    out_path = REPO / "results" / "EXPERIMENTAL_ftt_goba_merged_combined_single.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
