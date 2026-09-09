#!/usr/bin/env python
"""Score AUROC for the BackdoorVLA (openvla-oft, Text_Image_Attack object_TI_4)
img2img-per-layer (per camera: primary + wrist) and merged desc_only
image+text FTT stats, computed by
adapters/backdoorvla_openvla_oft/extract_img2img_and_merged_ftt.py and stored
per-episode in ExtractedSample.extra. No GPU, no attack code -- this just
reads the .npz files and calls detectors.ftt.auroc.

    python runners/score_backdoorvla_oft_img2img_merged.py \
        --samples-dir results/backdoorvla_oft_extracted_img2img_and_merged \
        --out results/ftt_backdoorvla_oft_img2img_and_merged.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.ftt import auroc  # noqa: E402
from detectors.schema import load_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    samples = load_dir(args.samples_dir)
    print(f"[*] loaded {len(samples)} samples from {args.samples_dir}")

    clean = [s for s in samples if s.label == 0]
    trig = [s for s in samples if s.label == 1]
    print(f"[*] n_clean={len(clean)} n_trigger={len(trig)}")

    # ---- token-count confound check -----------------------------------
    clean_tok = [s.extra["n_query_tokens_desc"] for s in clean]
    trig_tok = [s.extra["n_query_tokens_desc"] for s in trig]
    print(f"[*] n_query_tokens_desc: clean min/max/mean = "
          f"{min(clean_tok)}/{max(clean_tok)}/{np.mean(clean_tok):.2f}  "
          f"trigger min/max/mean = {min(trig_tok)}/{max(trig_tok)}/{np.mean(trig_tok):.2f}")

    # ---- 1. img2img per layer, per camera --------------------------------
    def per_layer_sweep(key):
        n_layers = len(samples[0].extra[key])
        aurocs = []
        for L in range(n_layers):
            c = [s.extra[key][L] for s in clean]
            t = [s.extra[key][L] for s in trig]
            aurocs.append(auroc(c, t))
        best = int(np.argmax(aurocs))
        return aurocs, best

    primary_auroc, primary_best = per_layer_sweep("img2img_ftt_per_layer_primary")
    wrist_auroc, wrist_best = per_layer_sweep("img2img_ftt_per_layer_wrist")

    print(f"[*] img2img per-layer AUROC, primary camera ({len(primary_auroc)} layers):")
    for L, a in enumerate(primary_auroc):
        marker = "  <-- best" if L == primary_best else ""
        print(f"      layer {L:2d}: {a:.4f}{marker}")

    print(f"[*] img2img per-layer AUROC, wrist camera ({len(wrist_auroc)} layers):")
    for L, a in enumerate(wrist_auroc):
        marker = "  <-- best" if L == wrist_best else ""
        print(f"      layer {L:2d}: {a:.4f}{marker}")

    # ---- 2. merged desc_only image+text, last layer ----------------------
    def group_auroc(key):
        c = [s.extra[key] for s in clean]
        t = [s.extra[key] for s in trig]
        return auroc(c, t), float(np.mean(c)), float(np.mean(t))

    merged_image_auroc, mi_c, mi_t = group_auroc("merged_ftt_image_group")
    merged_text_auroc, mt_c, mt_t = group_auroc("merged_ftt_text_group")
    merged_combined_auroc, mc_c, mc_t = group_auroc("merged_ftt_combined")

    print(f"[*] merged_ftt_image_group AUROC={merged_image_auroc:.4f}  clean_mean={mi_c:.4f} trig_mean={mi_t:.4f}")
    print(f"[*] merged_ftt_text_group  AUROC={merged_text_auroc:.4f}  clean_mean={mt_c:.4f} trig_mean={mt_t:.4f}")
    print(f"[*] merged_ftt_combined    AUROC={merged_combined_auroc:.4f}  clean_mean={mc_c:.4f} trig_mean={mc_t:.4f}")

    results = {
        "n_clean": len(clean),
        "n_trigger": len(trig),
        "n_query_tokens_desc": {
            "clean_min": min(clean_tok), "clean_max": max(clean_tok), "clean_mean": float(np.mean(clean_tok)),
            "trigger_min": min(trig_tok), "trigger_max": max(trig_tok), "trigger_mean": float(np.mean(trig_tok)),
        },
        "img2img_primary_per_layer_auroc": primary_auroc,
        "img2img_primary_best_layer": primary_best,
        "img2img_primary_best_layer_auroc": primary_auroc[primary_best],
        "img2img_wrist_per_layer_auroc": wrist_auroc,
        "img2img_wrist_best_layer": wrist_best,
        "img2img_wrist_best_layer_auroc": wrist_auroc[wrist_best],
        "merged_ftt_image_group_auroc": merged_image_auroc,
        "merged_ftt_text_group_auroc": merged_text_auroc,
        "merged_ftt_combined_auroc": merged_combined_auroc,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n[*] wrote {args.out}")


if __name__ == "__main__":
    main()
