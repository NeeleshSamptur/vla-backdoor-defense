#!/usr/bin/env python
"""Score adapters/goba/extract_vision_tower_attn.py's output: per-layer AUROC,
per tower, per role. No GPU, no attack code -- pure numpy/json, reusing
detectors.ftt.auroc unchanged.

The .npz files here are NOT detectors/schema.py's ExtractedSample (see the
header comment in extract_vision_tower_attn.py for why: two independent
towers with different layer counts and prefix-token counts don't fit that
one-attention-map contract naturally), so this is a small standalone reader
rather than detectors.schema.load_dir.

Usage:
    python runners/score_vision_tower_attn.py \
        --samples-dir results/goba_vision_tower_attn \
        --out results/ftt_goba_vision_tower_attn.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from detectors.ftt import auroc


def load_samples(samples_dir: str):
    files = sorted(Path(samples_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files under {samples_dir}")
    samples = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        samples.append(dict(
            dino=z["dino_ftt_per_layer"], siglip=z["siglip_ftt_per_layer"], **meta))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = load_samples(args.samples_dir)
    print(f"[*] loaded {len(samples)} samples from {args.samples_dir}")

    by_role = defaultdict(list)
    for s in samples:
        by_role[s["role"]].append(s)

    result = {}
    for role, group in sorted(by_role.items()):
        n_dino_layers = min(len(s["dino"]) for s in group)
        n_siglip_layers = min(len(s["siglip"]) for s in group)
        clean = [s for s in group if s["label"] == 0]
        trig = [s for s in group if s["label"] == 1]
        print(f"\n[*] role={role}: n={len(group)} (clean={len(clean)}, trig={len(trig)}) "
              f"dino_layers={n_dino_layers} siglip_layers={n_siglip_layers}")

        role_result = {"n_clean": len(clean), "n_trig": len(trig),
                        "n_dino_layers": n_dino_layers, "n_siglip_layers": n_siglip_layers}

        for tower, n_layers in (("dino", n_dino_layers), ("siglip", n_siglip_layers)):
            per_layer = []
            for l in range(n_layers):
                c = [s[tower][l] for s in clean]
                t = [s[tower][l] for s in trig]
                per_layer.append(auroc(c, t))
            best_layer = int(np.nanargmax(per_layer)) if per_layer else -1
            best_auroc = per_layer[best_layer] if per_layer else float("nan")
            role_result[f"{tower}_auroc_per_layer"] = per_layer
            role_result[f"{tower}_best_layer"] = best_layer
            role_result[f"{tower}_best_auroc"] = best_auroc
            print(f"    [{tower}] best layer={best_layer} AUROC={best_auroc:.4f}")
            print(f"    [{tower}] full curve: " +
                  " ".join(f"{i}:{v:.3f}" for i, v in enumerate(per_layer)))

        result[role] = role_result

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[*] saved -> {args.out}")


if __name__ == "__main__":
    main()
