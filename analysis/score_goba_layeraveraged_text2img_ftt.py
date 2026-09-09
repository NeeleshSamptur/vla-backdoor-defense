#!/usr/bin/env python
"""Layer-AVERAGED text2img FTT for GoBA: average the raw attention MAP across
all 32 LLM layers into ONE [n_txt, n_patches] map per episode, THEN compute a
single FTT score on that averaged map -- as opposed to the layerwise reports
(scoring each layer separately and reporting 32 AUROC numbers). This is a
different statistic, not a re-presentation of the layerwise numbers: FTT is
nonlinear (row-normalize, then L2 distance to the row-mean), so
FTT(mean_over_layers(map)) != mean_over_layers(FTT(map)).

Reads results/goba_text2img_layerwise_causal_fixed/*.npz, which already
stores the un-collapsed per-layer attention (`attn_text_image_layers`,
shape [32, n_txt, n_patches]) alongside the already-computed per-layer FTT
scores -- no new extraction needed, this is a new SCORING file only.
Filters role=='attack' (this directory also holds role=='clean_baseline'
episodes from the confound-check run; mixing them was the earlier
contamination bug this project already hit once).

Two comparisons, both size-matched:
  (A) 100 clean vs 100 trigger (all)
  (B) 63 clean vs 63 trigger (ASR-success-only, balanced) -- the exact same
      episodes selected in results/ftt_goba_text2img_layerwise_63v63_balanced.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "goba_text2img_layerwise_causal_fixed"
BALANCED_63V63_JSON = REPO / "results" / "ftt_goba_text2img_layerwise_63v63_balanced.json"


def ftt_score(P: np.ndarray) -> float:
    """Same definition as detectors/ftt.py's ftt_score (not imported to keep
    this script importable with plain numpy only, no torch/repo path setup
    needed) -- row-normalize, then mean L2 distance of each row to the mean
    row. Lower = more assimilated = predicted triggered."""
    P = np.clip(P, 0, None)
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    mean_row = P.mean(axis=0)
    return float(np.linalg.norm(P - mean_row[None, :], axis=1).mean())


def rank_auroc(clean_scores, trig_scores) -> float:
    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trig_scores, dtype=np.float64)
    s = np.concatenate([-c, -t])  # low FTT = triggered, so rank on -score
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
    rank_sum_trig = ranks[n_c:].sum()
    return float((rank_sum_trig - n_t * (n_t + 1) / 2) / (n_c * n_t))


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")

    with open(BALANCED_63V63_JSON) as f:
        balanced = json.load(f)
    clean_63_keys = {(tid, seed) for tid, seed in balanced["chosen_clean_pairs"]}
    trig_63_keys = {(tid, seed) for tid, seed in balanced["trigger_pairs"]}

    clean_100, trig_100 = {}, {}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        if meta["extra"]["role"] != "attack":
            continue
        layers = d["attn_text_image_layers"]  # [32, n_txt, n_patches]
        avg_map = layers.mean(axis=0)          # [n_txt, n_patches] -- ONE map, averaged over layers
        score = ftt_score(avg_map)
        key = (meta["task_id"], meta["seed"])
        (clean_100 if meta["label"] == 0 else trig_100)[key] = score

    assert len(clean_100) == 100 and len(trig_100) == 100, (len(clean_100), len(trig_100))

    clean_63 = [clean_100[k] for k in sorted(clean_63_keys)]
    trig_63 = [trig_100[k] for k in sorted(trig_63_keys)]
    assert len(clean_63) == 63 and len(trig_63) == 63, (len(clean_63), len(trig_63))

    auroc_100 = rank_auroc(list(clean_100.values()), list(trig_100.values()))
    auroc_63 = rank_auroc(clean_63, trig_63)

    print(f"\n[*] Layer-averaged (one map, mean over all 32 layers) text2img FTT AUROC:")
    print(f"    100 clean vs 100 trigger (all):              {auroc_100:.4f}")
    print(f"    63 clean vs 63 trigger (ASR-success, balanced): {auroc_63:.4f}")

    out = {
        "n_clean_100": 100, "n_trig_100": 100, "n_clean_63": 63, "n_trig_63": 63,
        "auroc_100v100_layeraveraged": auroc_100,
        "auroc_63v63_layeraveraged": auroc_63,
    }
    out_path = REPO / "results" / "ftt_goba_text2img_layeraveraged_auroc.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
