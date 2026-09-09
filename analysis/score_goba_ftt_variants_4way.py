#!/usr/bin/env python
"""Four FTT variants on GoBA text2img attention, crossing two independent
choices:

  Construction axis:
    (A) FLATTEN  -- average the 32 layers' attention maps into ONE map,
        then compute one reference and one set of row-distances from that
        single flattened map.
    (B) SHARED-REF -- average each layer's own row-mean into ONE shared
        reference vector, but measure distances using EVERY layer's own
        individual (unflattened) token rows against that one reference,
        then average distances over tokens, then over layers.

  Normalization axis:
    (raw)    -- row-normalization is skipped entirely; every averaging and
                distance computation runs on the raw post-softmax attention
                values sliced to the image-patch columns.
    (normed) -- each row is row-normalized (divided by its own sum over the
                sliced columns) before anything else happens, exactly
                detectors/ftt.py's convention.

This gives 4 variants:
    A-normed  = analysis/score_goba_layeraveraged_text2img_ftt.py (already run, 0.8351/0.8251)
    B-normed  = analysis/score_goba_meanofmeans_layerwise_ftt.py / detectors/ftt.py's
                ftt_score_layerwise (already run, 0.0890/0.0784)
    A-raw     = NEW here
    B-raw     = NEW here

Reads the same source as both earlier scripts
(results/goba_text2img_layerwise_causal_fixed/*.npz, role=='attack' only).
New scoring file only -- does not modify detectors/ftt.py or either earlier
scoring script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "goba_text2img_layerwise_causal_fixed"
BALANCED_63V63_JSON = REPO / "results" / "ftt_goba_text2img_layerwise_63v63_balanced.json"
EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    """Same convention as detectors/ftt.py: clip negatives (attention is
    already >=0 post-softmax, this is just parity with that file), divide
    each row by its own sum over the LAST axis."""
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def score_A_flatten(layers: np.ndarray, normalize: bool) -> float:
    """Average the 32 layers into ONE map, optionally row-normalize that one
    map, then standard single-map FTT (reference = that map's own row-mean,
    distance = each row to that reference)."""
    avg_map = layers.mean(axis=0)  # [T, P] -- averaging happens on raw values either way
    if normalize:
        avg_map = row_normalize(avg_map)
    ref = avg_map.mean(axis=0)  # [P]
    return float(np.linalg.norm(avg_map - ref[None, :], axis=1).mean())


def score_B_sharedref(layers: np.ndarray, normalize: bool) -> float:
    """Optionally row-normalize each layer separately, take each layer's own
    row-mean (32 vectors), average those into ONE shared reference, then
    measure every layer's own individual token rows against that one
    reference -- never flattening layers into one map."""
    P = row_normalize(layers) if normalize else layers  # [L,T,P]
    per_layer_mean = P.mean(axis=1)      # [L,P] -- mean over tokens, per layer
    ref = per_layer_mean.mean(axis=0)    # [P]   -- mean over layers -> shared reference
    per_token = np.linalg.norm(P - ref[None, None, :], axis=-1)  # [L,T]
    per_layer = per_token.mean(axis=1)   # mean over tokens
    return float(per_layer.mean())       # mean over layers


def rank_auroc(clean_scores, trig_scores) -> float:
    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trig_scores, dtype=np.float64)
    s = np.concatenate([-c, -t])  # fixed convention: low score = triggered
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

    variants = ["A_raw", "A_normed", "B_raw", "B_normed"]
    clean_100 = {v: {} for v in variants}
    trig_100 = {v: {} for v in variants}

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        if meta["extra"]["role"] != "attack":
            continue
        layers = d["attn_text_image_layers"].astype(np.float64)  # [32, n_txt, n_patches]
        key = (meta["task_id"], meta["seed"])
        bucket = clean_100 if meta["label"] == 0 else trig_100

        bucket["A_raw"][key] = score_A_flatten(layers, normalize=False)
        bucket["A_normed"][key] = score_A_flatten(layers, normalize=True)
        bucket["B_raw"][key] = score_B_sharedref(layers, normalize=False)
        bucket["B_normed"][key] = score_B_sharedref(layers, normalize=True)

    for v in variants:
        assert len(clean_100[v]) == 100 and len(trig_100[v]) == 100

    results = {}
    print()
    for v in variants:
        c100 = list(clean_100[v].values())
        t100 = list(trig_100[v].values())
        c63 = [clean_100[v][k] for k in sorted(clean_63_keys)]
        t63 = [trig_100[v][k] for k in sorted(trig_63_keys)]
        auroc_100 = rank_auroc(c100, t100)
        auroc_63 = rank_auroc(c63, t63)
        results[v] = {"auroc_100v100": auroc_100, "auroc_63v63": auroc_63,
                      "clean_mean": float(np.mean(c100)), "trig_mean": float(np.mean(t100))}
        print(f"[*] {v:10s}  100v100={auroc_100:.4f}  63v63={auroc_63:.4f}  "
              f"(clean_mean={np.mean(c100):.4f} trig_mean={np.mean(t100):.4f})")

    out_path = REPO / "results" / "ftt_goba_text2img_4way_variants.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
