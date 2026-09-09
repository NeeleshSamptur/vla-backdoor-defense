#!/usr/bin/env python
""""Mean of means" layerwise FTT for GoBA: build ONE shared reference vector
by averaging each layer's OWN row-mean ("assimilated pattern") across all 32
layers -- detectors/ftt.py's grand_mean_pattern -- then measure every
individual (layer, token) row's distance to that ONE shared reference and
average first over tokens (per layer), then over layers. This is
detectors/ftt.py's ftt_score_layerwise, unused elsewhere in this project's
reports until now.

Different from BOTH earlier statistics on this same data:
  - the per-layer report (results/ftt_goba_text2img_layerwise_*.json):
    32 separate scores, one independent reference per layer.
  - the layer-averaged report (analysis/score_goba_layeraveraged_text2img_ftt.py):
    the raw MAPS are averaged into one map first, then a single reference
    and single set of row-distances come from that one flattened map --
    individual layers' own rows are never separately measured.
  - here: the reference is a simple average (mathematically equivalent to a
    flat mean over all (layer, token) pairs, since token count is constant
    across layers), but the quantities being averaged in the FINAL score are
    each layer's own un-flattened per-token distances to that reference --
    so a layer whose rows are unusually spread out still shows up in the
    final score instead of being smoothed away before any distance is ever
    computed.

Reads results/goba_text2img_layerwise_causal_fixed/*.npz (same source as
the layer-averaged script), filters role=='attack'. New scoring file only,
does not modify detectors/ftt.py or any extraction script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from detectors.ftt import ftt_score_layerwise  # noqa: E402 -- unchanged, reused as-is

DATA_DIR = REPO / "results" / "goba_text2img_layerwise_causal_fixed"
BALANCED_63V63_JSON = REPO / "results" / "ftt_goba_text2img_layerwise_63v63_balanced.json"


def rank_auroc(clean_scores, trig_scores) -> float:
    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trig_scores, dtype=np.float64)
    s = np.concatenate([-c, -t])  # low score = triggered, so rank on -score
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

    clean_100, trig_100 = {}, {}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        if meta["extra"]["role"] != "attack":
            continue
        layers = d["attn_text_image_layers"]  # [32, n_txt, n_patches], raw (not yet row-normalized)
        score = ftt_score_layerwise(layers)
        key = (meta["task_id"], meta["seed"])
        (clean_100 if meta["label"] == 0 else trig_100)[key] = score

    assert len(clean_100) == 100 and len(trig_100) == 100, (len(clean_100), len(trig_100))

    clean_63 = [clean_100[k] for k in sorted(clean_63_keys)]
    trig_63 = [trig_100[k] for k in sorted(trig_63_keys)]
    assert len(clean_63) == 63 and len(trig_63) == 63, (len(clean_63), len(trig_63))

    auroc_100 = rank_auroc(list(clean_100.values()), list(trig_100.values()))
    auroc_63 = rank_auroc(clean_63, trig_63)

    print(f"\n[*] 'Mean of means' layerwise FTT AUROC:")
    print(f"    100 clean vs 100 trigger (all):                 {auroc_100:.4f}")
    print(f"    63 clean vs 63 trigger (ASR-success, balanced): {auroc_63:.4f}")

    out = {
        "n_clean_100": 100, "n_trig_100": 100, "n_clean_63": 63, "n_trig_63": 63,
        "auroc_100v100_meanofmeans": auroc_100,
        "auroc_63v63_meanofmeans": auroc_63,
    }
    out_path = REPO / "results" / "ftt_goba_text2img_meanofmeans_auroc.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
