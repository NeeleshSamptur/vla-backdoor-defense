#!/usr/bin/env python
"""Scoring for adapters/goba/extract_img2img_merged_alllayers_fixed.py's
output (results/goba_img2img_merged_alllayers_fixed/*.npz) -- the
consistent, single-run, eager-attention (post SDPA-bug-fix) scalar+raw
img2img/merged data.

Reuses row_normalize/ftt_on_rows/flatten_*/sharedref_*/rank_auroc VERBATIM
from analysis/EXPERIMENTAL_score_goba_img2img_merged_3configs.py and
analysis/EXPERIMENTAL_score_goba_merged_combined_single.py, per direct
instruction -- do not reinvent this math.

Produces, for img2img and for the "merged combined" single-query-group
statistic (image-group rows concatenated with text-group rows, following
EXPERIMENTAL_score_goba_merged_combined_single.py's exact convention):
  - full 32-layer per-layer AUROC, both 100v100 (all attack-role episodes)
    and 63v63 (successful-attack-only balanced subset, using the exact
    (task_id, seed) pairs from results/ftt_goba_text2img_layerwise_63v63_
    balanced.json -- NOT a freshly recomputed balanced subset)
  - the 4 Flatten/Shared-reference combined-32-layer scores (100v100, 63v63
    each)

Also cross-validates, per episode, the scalar img2img_ftt_per_layer saved
directly by the extractor against the scalar recomputed here from that same
episode's raw img2img_raw tensor -- these should match to float32 precision
since both come from the identical A_np slice within the same forward pass;
any mismatch would indicate a genuine bug (not the SDPA/eager issue, which
this data was already extracted after fixing).

Saves results/ftt_goba_img2img_merged_alllayers_FIXED.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "goba_img2img_merged_alllayers_fixed"
BALANCED_63V63_JSON = REPO / "results" / "ftt_goba_text2img_layerwise_63v63_balanced.json"
EPS = 1e-12


# ---- verbatim from EXPERIMENTAL_score_goba_img2img_merged_3configs.py /
#      EXPERIMENTAL_score_goba_merged_combined_single.py ----

def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def ftt_on_rows(rows: np.ndarray) -> float:
    ref = rows.mean(axis=0)
    return float(np.linalg.norm(rows - ref[None, :], axis=1).mean())


def per_layer_ftt(layers_normed: np.ndarray) -> np.ndarray:
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


# ---- new: reads the FIXED data (scalar + raw + cross-check) ----

def score_key(key_name, files, clean_63_keys, trig_63_keys):
    clean_layers, trig_layers = {}, {}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if str(d["role"]) != "attack":
            continue
        layers_raw = d[key_name].astype(np.float64)
        key = (int(d["task_id"]), int(d["seed"]))
        (clean_layers if int(d["label"]) == 0 else trig_layers)[key] = layers_raw

    assert len(clean_layers) == 100 and len(trig_layers) == 100, (key_name, len(clean_layers), len(trig_layers))

    clean_pl = np.stack([per_layer_ftt(row_normalize(v)) for v in clean_layers.values()])
    trig_pl = np.stack([per_layer_ftt(row_normalize(v)) for v in trig_layers.values()])
    n_layers = clean_pl.shape[1]
    keys_sorted_c = sorted(clean_layers.keys())
    keys_sorted_t = sorted(trig_layers.keys())
    clean_pl_ordered = np.stack([per_layer_ftt(row_normalize(clean_layers[k])) for k in keys_sorted_c])
    trig_pl_ordered = np.stack([per_layer_ftt(row_normalize(trig_layers[k])) for k in keys_sorted_t])
    auroc_pl_100 = np.array([rank_auroc(clean_pl_ordered[:, l], trig_pl_ordered[:, l]) for l in range(n_layers)])

    idx_c = {k: i for i, k in enumerate(keys_sorted_c)}
    idx_t = {k: i for i, k in enumerate(keys_sorted_t)}
    c63_idx = [idx_c[k] for k in sorted(clean_63_keys) if k in idx_c]
    t63_idx = [idx_t[k] for k in sorted(trig_63_keys) if k in idx_t]
    auroc_pl_63 = np.array([
        rank_auroc(clean_pl_ordered[c63_idx, l], trig_pl_ordered[t63_idx, l]) for l in range(n_layers)])

    c_na = {k: flatten_normalize_then_average(v) for k, v in clean_layers.items()}
    t_na = {k: flatten_normalize_then_average(v) for k, v in trig_layers.items()}
    c_an = {k: flatten_average_then_normalize(v) for k, v in clean_layers.items()}
    t_an = {k: flatten_average_then_normalize(v) for k, v in trig_layers.items()}
    auroc_flat_na_100 = rank_auroc(list(c_na.values()), list(t_na.values()))
    auroc_flat_na_63 = rank_auroc([c_na[k] for k in sorted(clean_63_keys)], [t_na[k] for k in sorted(trig_63_keys)])
    auroc_flat_an_100 = rank_auroc(list(c_an.values()), list(t_an.values()))
    auroc_flat_an_63 = rank_auroc([c_an[k] for k in sorted(clean_63_keys)], [t_an[k] for k in sorted(trig_63_keys)])

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


def cross_check_scalar_vs_raw(files):
    """Per-episode: saved img2img_ftt_per_layer (computed directly from A_np)
    vs FTT recomputed here from the saved img2img_raw tensor. Should match to
    float32 precision -- both came from the identical A_np slice in the same
    forward pass."""
    import sys
    sys.path.insert(0, str(REPO))
    from detectors.ftt import ftt_score

    max_diff = 0.0
    n_checked = 0
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        saved = np.asarray(d["img2img_ftt_per_layer"], dtype=np.float64)
        raw = d["img2img_raw"].astype(np.float64)
        recomputed = np.array([ftt_score(raw[l]) for l in range(raw.shape[0])])
        diff = np.abs(saved - recomputed).max()
        max_diff = max(max_diff, diff)
        n_checked += 1
    return {"n_episodes_checked": n_checked, "max_abs_diff_scalar_vs_recomputed_from_raw": float(max_diff)}


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")
    if not files:
        print("[!] no files found -- extraction has not produced output yet")
        return

    with open(BALANCED_63V63_JSON) as f:
        balanced = json.load(f)
    clean_63_keys = {(tid, seed) for tid, seed in balanced["chosen_clean_pairs"]}
    trig_63_keys = {(tid, seed) for tid, seed in balanced["trigger_pairs"]}

    print("[*] cross-checking saved scalar vs FTT recomputed from raw (internal consistency)...")
    cross_check = cross_check_scalar_vs_raw(files)
    print(f"    {cross_check}")

    print("\n[*] === img2img ===")
    img2img_result = score_key("img2img_raw", files, clean_63_keys, trig_63_keys)
    best_l = int(np.argmax(img2img_result["per_layer_100v100"]))
    print(f"    best layer {best_l}: 100v100={img2img_result['per_layer_100v100'][best_l]:.6f} "
          f"63v63={img2img_result['per_layer_63v63'][best_l]:.6f}")

    print("\n[*] === merged combined (image-group rows + text-group rows, single query block) ===")
    clean_layers, trig_layers = {}, {}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if str(d["role"]) != "attack":
            continue
        combined = np.concatenate(
            [d["merged_image_raw"].astype(np.float64), d["merged_text_raw"].astype(np.float64)], axis=1)
        key = (int(d["task_id"]), int(d["seed"]))
        (clean_layers if int(d["label"]) == 0 else trig_layers)[key] = combined
    assert len(clean_layers) == 100 and len(trig_layers) == 100

    keys_c = sorted(clean_layers.keys())
    keys_t = sorted(trig_layers.keys())
    clean_pl = np.stack([per_layer_ftt(row_normalize(clean_layers[k])) for k in keys_c])
    trig_pl = np.stack([per_layer_ftt(row_normalize(trig_layers[k])) for k in keys_t])
    n_layers = clean_pl.shape[1]
    mc_auroc_pl_100 = np.array([rank_auroc(clean_pl[:, l], trig_pl[:, l]) for l in range(n_layers)])
    idx_c = {k: i for i, k in enumerate(keys_c)}
    idx_t = {k: i for i, k in enumerate(keys_t)}
    c63_idx = [idx_c[k] for k in sorted(clean_63_keys)]
    t63_idx = [idx_t[k] for k in sorted(trig_63_keys)]
    mc_auroc_pl_63 = np.array([rank_auroc(clean_pl[c63_idx, l], trig_pl[t63_idx, l]) for l in range(n_layers)])

    c_na = {k: flatten_normalize_then_average(v) for k, v in clean_layers.items()}
    t_na = {k: flatten_normalize_then_average(v) for k, v in trig_layers.items()}
    c_an = {k: flatten_average_then_normalize(v) for k, v in clean_layers.items()}
    t_an = {k: flatten_average_then_normalize(v) for k, v in trig_layers.items()}
    mc_flat_na_100 = rank_auroc(list(c_na.values()), list(t_na.values()))
    mc_flat_na_63 = rank_auroc([c_na[k] for k in sorted(clean_63_keys)], [t_na[k] for k in sorted(trig_63_keys)])
    mc_flat_an_100 = rank_auroc(list(c_an.values()), list(t_an.values()))
    mc_flat_an_63 = rank_auroc([c_an[k] for k in sorted(clean_63_keys)], [t_an[k] for k in sorted(trig_63_keys)])

    c_sna = {k: sharedref_normalize_then_average(v) for k, v in clean_layers.items()}
    t_sna = {k: sharedref_normalize_then_average(v) for k, v in trig_layers.items()}
    c_san = {k: sharedref_average_then_normalize(v) for k, v in clean_layers.items()}
    t_san = {k: sharedref_average_then_normalize(v) for k, v in trig_layers.items()}
    mc_shared_na_100 = rank_auroc(list(c_sna.values()), list(t_sna.values()))
    mc_shared_na_63 = rank_auroc([c_sna[k] for k in sorted(clean_63_keys)], [t_sna[k] for k in sorted(trig_63_keys)])
    mc_shared_an_100 = rank_auroc(list(c_san.values()), list(t_san.values()))
    mc_shared_an_63 = rank_auroc([c_san[k] for k in sorted(clean_63_keys)], [t_san[k] for k in sorted(trig_63_keys)])

    merged_combined_result = {
        "per_layer_100v100": mc_auroc_pl_100.tolist(),
        "per_layer_63v63": mc_auroc_pl_63.tolist(),
        "flatten_normalize_then_average": {"auroc_100v100": mc_flat_na_100, "auroc_63v63": mc_flat_na_63},
        "flatten_average_then_normalize": {"auroc_100v100": mc_flat_an_100, "auroc_63v63": mc_flat_an_63},
        "sharedref_normalize_then_average": {"auroc_100v100": mc_shared_na_100, "auroc_63v63": mc_shared_na_63},
        "sharedref_average_then_normalize": {"auroc_100v100": mc_shared_an_100, "auroc_63v63": mc_shared_an_63},
    }
    best_l_mc = int(np.argmax(merged_combined_result["per_layer_100v100"]))
    print(f"    best layer {best_l_mc}: 100v100={merged_combined_result['per_layer_100v100'][best_l_mc]:.6f} "
          f"63v63={merged_combined_result['per_layer_63v63'][best_l_mc]:.6f}")

    out = {
        "cross_check_scalar_vs_raw": cross_check,
        "img2img": img2img_result,
        "merged_combined": merged_combined_result,
    }
    out_path = REPO / "results" / "ftt_goba_img2img_merged_alllayers_FIXED.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
