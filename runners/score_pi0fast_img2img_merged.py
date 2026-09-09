#!/usr/bin/env python
"""Score AUROC for the pi0-FAST TI4 img2img-per-layer and merged desc_only
image+text FTT stats, which the extractor
(adapters/pi0fast_backdoorvla/extract_img2img_merged_ftt.py) already computed
and stored per-episode in ExtractedSample.extra. No GPU, no attack code --
this just reads the .npz files and calls detectors.ftt.auroc.

    python runners/score_pi0fast_img2img_merged.py \
        --samples-dir results/pi0fast_ti4_extracted_img2img_and_merged \
        --out results/ftt_pi0fast_ti4_img2img_and_merged.json
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

    # ---- 1. img2img per layer -------------------------------------------
    n_layers = len(samples[0].extra["img2img_ftt_per_layer"])
    per_layer_auroc = []
    for L in range(n_layers):
        c = [s.extra["img2img_ftt_per_layer"][L] for s in clean]
        t = [s.extra["img2img_ftt_per_layer"][L] for s in trig]
        per_layer_auroc.append(auroc(c, t))
    best_layer = int(np.argmax(per_layer_auroc))
    print(f"[*] img2img per-layer AUROC ({n_layers} layers):")
    for L, a in enumerate(per_layer_auroc):
        marker = "  <-- best" if L == best_layer else ""
        print(f"      layer {L:2d}: {a:.4f}{marker}")

    # bidirectionality check (upper-triangular, last layer)
    upper_means = [s.extra["img2img_upper_tri_mean"] for s in samples]
    upper_fracs = [s.extra["img2img_upper_tri_frac_nonzero"] for s in samples]
    print(f"[*] img2img last-layer upper-triangular (j>i) check: "
          f"mean attn mean={np.mean(upper_means):.6f}, "
          f"frac nonzero mean={np.mean(upper_fracs):.4f} "
          f"(0 would mean pure-causal; nonzero confirms bidirectional prefix)")

    # ---- 2. merged desc_only image+text, last layer ----------------------
    def group_auroc(key):
        c = [s.extra[key] for s in clean]
        t = [s.extra[key] for s in trig]
        return auroc(c, t), float(np.mean(c)), float(np.mean(t))

    merged_pooled_auroc, mp_c, mp_t = group_auroc("merged_pooled_ftt")
    merged_text_auroc, mt_c, mt_t = group_auroc("merged_text_group_ftt")
    merged_image_auroc, mi_c, mi_t = group_auroc("merged_image_group_ftt")
    merged_combined_auroc, mc_c, mc_t = group_auroc("merged_combined_ftt")

    print(f"[*] merged_pooled_ftt      AUROC={merged_pooled_auroc:.4f}  clean_mean={mp_c:.4f} trig_mean={mp_t:.4f}")
    print(f"[*] merged_text_group_ftt  AUROC={merged_text_auroc:.4f}  clean_mean={mt_c:.4f} trig_mean={mt_t:.4f}")
    print(f"[*] merged_image_group_ftt AUROC={merged_image_auroc:.4f}  clean_mean={mi_c:.4f} trig_mean={mi_t:.4f}")
    print(f"[*] merged_combined_ftt    AUROC={merged_combined_auroc:.4f}  clean_mean={mc_c:.4f} trig_mean={mc_t:.4f}")

    results = {
        "n_clean": len(clean),
        "n_trigger": len(trig),
        "n_query_tokens_desc": {
            "clean_min": min(clean_tok), "clean_max": max(clean_tok), "clean_mean": float(np.mean(clean_tok)),
            "trigger_min": min(trig_tok), "trigger_max": max(trig_tok), "trigger_mean": float(np.mean(trig_tok)),
        },
        "img2img_per_layer_auroc": per_layer_auroc,
        "img2img_best_layer": best_layer,
        "img2img_best_layer_auroc": per_layer_auroc[best_layer],
        "img2img_upper_tri_mean_attn": float(np.mean(upper_means)),
        "img2img_upper_tri_frac_nonzero": float(np.mean(upper_fracs)),
        "merged_pooled_ftt_auroc": merged_pooled_auroc,
        "merged_text_group_ftt_auroc": merged_text_auroc,
        "merged_image_group_ftt_auroc": merged_image_auroc,
        "merged_combined_ftt_auroc": merged_combined_auroc,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n[*] wrote {args.out}")


if __name__ == "__main__":
    main()
