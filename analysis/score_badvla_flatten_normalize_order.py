#!/usr/bin/env python
"""Two orderings of the same 'flatten to one map' FTT construction on BadVLA
(white_patch attack) text2img desc_only attention, differing only in WHEN
row-normalization happens relative to averaging the 32 layers together:

  (1) normalize-then-average: row-normalize each layer's map separately
      first, THEN average those 32 already-normalized maps into one map,
      then run FTT on that one map (reference = its row-mean, distance =
      its own rows to that reference).
  (2) average-then-normalize: average the 32 RAW (un-normalized) maps into
      one map first, THEN row-normalize that single combined map, then run
      FTT on it the same way. Kept here side-by-side for direct comparison,
      not a re-derivation.

Both flatten to ONE map before scoring -- the only difference is
normalization order, and because normalization (divide-by-row-sum) is
nonlinear relative to averaging, the two orders are not equivalent.

Mirrors analysis/score_goba_flatten_normalize_order.py exactly (same
score_normalize_then_average / score_average_then_normalize / rank_auroc
logic), adapted to BadVLA's data:

Reads results/badvla_text2img_alllayers_causal_fixed/*.npz (written by
adapters/badvla_white_patch/extract_text2img_alllayers_driver.py), role==
'attack' only. attn_text_image_layers is already desc_only (query = task
description tokens only, extra["text_scope"]=="desc_only"), primary camera,
shape [32 layers, T query tokens, 256 image patches].

Unlike GoBA, there is no pre-existing balanced-63v63 pairing file for
BadVLA, so this script reports the full 100-clean-vs-100-trigger AUROC only.

New scoring file -- does not modify detectors/ftt.py or any other scoring
script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "badvla_text2img_alllayers_causal_fixed"
EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def score_normalize_then_average(layers: np.ndarray) -> float:
    normed = row_normalize(layers)   # [L, T, P] -- each row already sums to 1
    avg_map = normed.mean(axis=0)    # [T, P] -- average of already-normalized rows
    ref = avg_map.mean(axis=0)       # [P]
    return float(np.linalg.norm(avg_map - ref[None, :], axis=1).mean())


def score_average_then_normalize(layers: np.ndarray) -> float:
    avg_map = layers.mean(axis=0)    # [T, P] -- average of RAW rows
    avg_map = row_normalize(avg_map)
    ref = avg_map.mean(axis=0)
    return float(np.linalg.norm(avg_map - ref[None, :], axis=1).mean())


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

    variants = ["normalize_then_average", "average_then_normalize"]
    clean = {v: {} for v in variants}
    trig = {v: {} for v in variants}

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        if meta["extra"]["role"] != "attack":
            continue
        layers = d["attn_text_image_layers"].astype(np.float64)
        key = (meta["task_id"], meta["seed"])
        bucket = clean if meta["label"] == 0 else trig
        bucket["normalize_then_average"][key] = score_normalize_then_average(layers)
        bucket["average_then_normalize"][key] = score_average_then_normalize(layers)

    for v in variants:
        print(f"[*] {v}: n_clean={len(clean[v])} n_trigger={len(trig[v])}")

    results = {}
    print()
    for v in variants:
        c = list(clean[v].values())
        t = list(trig[v].values())
        auroc = rank_auroc(c, t)
        results[v] = {
            "n_clean": len(c),
            "n_trigger": len(t),
            "clean_mean": float(np.mean(c)),
            "trig_mean": float(np.mean(t)),
            "auroc_100v100": auroc,
        }
        print(f"[*] {v:24s}  n_clean={len(c)} n_trigger={len(t)}  "
              f"clean_mean={np.mean(c):.5f}  trig_mean={np.mean(t):.5f}  auroc={auroc:.4f}")

    out_path = REPO / "results" / "ftt_badvla_text2img_flatten_normalize_order.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
