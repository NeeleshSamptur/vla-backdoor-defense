#!/usr/bin/env python
"""Score per-(layer,head) text2img FTT AUROC for GoBA, two SIZE-MATCHED
comparisons only (no 100-vs-63 mismatch):
  (A) 100 clean vs 100 trigger (all)
  (B) 63 clean vs 63 trigger (ASR-success-only, balanced) -- the EXACT same
      63 clean and 63 trigger (task_id, seed) episodes selected by the
      earlier layerwise balanced run (results/ftt_goba_text2img_layerwise_
      63v63_balanced.json's chosen_clean_pairs / trigger_pairs), so this is
      a like-for-like re-run of that same comparison, just per-head instead
      of head-averaged.

Reads results/goba_perhead_text2img/*.npz (written by the sibling
extraction script extract_perhead_text2img_ftt.py). Standalone script --
does not import or modify any existing scoring/extraction file.

Reporting: ranks by best-per-HEAD (max AUROC over all 32 layers for that
head), not by cherry-picked (layer,head) cells -- i.e. we ask "which heads
carry a strong signal" rather than picking isolated layer/head pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "goba_perhead_text2img"
BALANCED_63V63_JSON = REPO / "results" / "ftt_goba_text2img_layerwise_63v63_balanced.json"


def rank_auroc(clean_scores, trig_scores) -> float:
    """Mann-Whitney rank-based AUROC. label 1 = triggered, LOW score = predicted triggered
    (FTT polarity), so we rank on -score."""
    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trig_scores, dtype=np.float64)
    if len(c) == 0 or len(t) == 0:
        return float("nan")
    s = np.concatenate([-c, -t])  # negate so higher = more "triggered-like"
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # tie correction: average ranks for equal values
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
    auc = (rank_sum_trig - n_t * (n_t + 1) / 2) / (n_c * n_t)
    return float(auc)


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")

    with open(BALANCED_63V63_JSON) as f:
        balanced = json.load(f)
    clean_63_keys = {(tid, seed) for tid, seed in balanced["chosen_clean_pairs"]}
    trig_63_keys = {(tid, seed) for tid, seed in balanced["trigger_pairs"]}

    clean_100, trig_100 = {}, {}
    n_layers = n_heads = None

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        label = meta["label"]
        extra = meta["extra"]
        cell = np.asarray(extra["text2img_perhead_ftt"], dtype=np.float64)  # [L, H]
        if n_layers is None:
            n_layers, n_heads = cell.shape
        key = (meta["task_id"], meta["seed"])
        (clean_100 if label == 0 else trig_100)[key] = cell

    assert len(clean_100) == 100 and len(trig_100) == 100, (len(clean_100), len(trig_100))

    clean_63 = [clean_100[k] for k in sorted(clean_63_keys)]
    trig_63 = [trig_100[k] for k in sorted(trig_63_keys)]
    assert len(clean_63) == 63 and len(trig_63) == 63, (len(clean_63), len(trig_63))
    print(f"[*] n_clean=100 n_trig=100 (all)   |   n_clean=63 n_trig=63 (balanced, same episodes as the layerwise 63v63 run)")

    clean_100_stack = np.stack(list(clean_100.values()))
    trig_100_stack = np.stack(list(trig_100.values()))
    clean_63_stack = np.stack(clean_63)
    trig_63_stack = np.stack(trig_63)

    auroc_100 = np.zeros((n_layers, n_heads))
    auroc_63 = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            auroc_100[l, h] = rank_auroc(clean_100_stack[:, l, h], trig_100_stack[:, l, h])
            auroc_63[l, h] = rank_auroc(clean_63_stack[:, l, h], trig_63_stack[:, l, h])

    out = {
        "n_clean_100": 100, "n_trig_100": 100,
        "n_clean_63": 63, "n_trig_63": 63,
        "n_layers": n_layers, "n_heads": n_heads,
        "auroc_100v100": auroc_100.tolist(),
        "auroc_63v63": auroc_63.tolist(),
    }
    out_path = REPO / "results" / "goba_perhead_text2img_auroc.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[*] saved -> {out_path}")

    for name, arr in [("100 clean vs 100 trigger", auroc_100), ("63 clean vs 63 trigger (balanced)", auroc_63)]:
        best_per_head = arr.max(axis=0)          # [H] -- best AUROC for this head, over any layer
        best_layer_per_head = arr.argmax(axis=0)  # [H]
        order = np.argsort(best_per_head)[::-1]
        print(f"\n[*] {name} -- heads ranked by their OWN best-over-layers AUROC:")
        for h in order[:10]:
            print(f"    head={h:2d}  best_AUROC={best_per_head[h]:.4f}  (at layer {best_layer_per_head[h]})")
        print(f"    mean over all 32 heads of each head's own best-over-layers AUROC: {best_per_head.mean():.4f}")


if __name__ == "__main__":
    main()
