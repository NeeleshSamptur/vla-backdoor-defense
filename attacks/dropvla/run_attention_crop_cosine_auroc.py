#!/usr/bin/env python
"""DropVLA: COSINE DISTANCE between the full-image and cropped-image
ATTENTION MAPS -> AUROC. No GPU: reads the maps
attacks/dropvla/run_crop_ftt_auroc.py already saved.

This is the attention-space counterpart of run_crop_hidden_auroc.py's
score. That one asks how far the last-layer ACTIVATIONS move when the frame
is cropped; this one asks how far the text-to-image ATTENTION moves, on the
identical frames, with the identical crop, from the identical forward
passes -- so the two numbers isolate the readout and nothing else.

Score, per sample and per camera (HIGH = triggered):

    collapsed    -- each layer's map is row-normalized and the layers
                    averaged into one [n_desc_tokens, n_patches] map, exactly
                    as attacks.common.compute_ftt builds its map, for the
                    uncropped and cropped frame separately; the score is the
                    mean over query-token rows of 1 - cos(row_unc, row_crop).
                    Per row, not on the flattened map, matching how
                    attacks.common.cosine_distance scores activations.
    per_layer    -- the same row-wise cosine distance computed inside each
                    layer and then averaged over layers, i.e. layers are
                    never mixed. Distinguishes "the averaged map moved" from
                    "every layer moved".
    flattened    -- one cosine distance over the whole collapsed map treated
                    as a single vector. The crudest of the three, included
                    because it is what "cosine distance between the two maps"
                    most literally means.

The sibling run_crop_ftt_auroc.py already reports `attn_shift`, the
FROBENIUS distance between the same two collapsed maps; these are the cosine
versions of that, which ignore magnitude and compare only direction.

Usage (no GPU, no attack repo):
    python attacks/dropvla/run_attention_crop_cosine_auroc.py \
        --maps-dir results/dropvla_crop_ftt/attention_maps \
        --out results/dropvla_attention_crop_cosine_auroc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFENSE_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEFENSE_REPO))

from attacks.common import compute_auroc_high_is_triggered, cosine_distance, row_normalize

CAMERAS = ("primary", "wrist")
READOUTS = ("collapsed", "per_layer", "flattened")


def collapse(maps: np.ndarray) -> np.ndarray:
    """[n_layers, n_desc, n_patches] -> [n_desc, n_patches], the same
    normalize-then-average construction compute_ftt uses."""
    return row_normalize(maps).mean(axis=0)


def score_pair(unc: np.ndarray, crp: np.ndarray) -> dict:
    a, b = collapse(unc), collapse(crp)
    per_layer = np.mean([cosine_distance(row_normalize(unc)[l], row_normalize(crp)[l])
                         for l in range(unc.shape[0])])
    return {
        "collapsed": cosine_distance(a, b),
        "per_layer": float(per_layer),
        "flattened": cosine_distance(a.reshape(1, -1), b.reshape(1, -1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default=str(DEFENSE_REPO / "results" / "dropvla_crop_ftt" / "attention_maps"))
    ap.add_argument("--out", default=str(DEFENSE_REPO / "results" / "dropvla_attention_crop_cosine_auroc.json"))
    args = ap.parse_args()

    files = sorted(Path(args.maps_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no attention maps under {args.maps_dir}")

    scores = {cam: {r: {"clean": [], "trigger": []} for r in READOUTS} for cam in CAMERAS}
    episodes = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        cond = meta["condition"]
        rec = {"task_id": meta["task_id"], "init_index": meta["init_index"],
               "condition": cond, "label": meta["label"]}
        for cam in CAMERAS:
            s = score_pair(z[f"{cam}_uncropped"], z[f"{cam}_cropped"])
            for r in READOUTS:
                scores[cam][r][cond].append(s[r])
                rec[f"{cam}_{r}"] = s[r]
        episodes.append(rec)

    print(f"[*] {len(files)} maps from {args.maps_dir}")
    results = {
        "attack": "dropvla", "source_maps": str(args.maps_dir),
        "score": "cosine distance between the full-image and cropped-image text-to-image attention maps",
        "polarity": "high = triggered",
        "n_samples": len(files),
    }
    for cam in CAMERAS:
        results[cam] = {}
        for r in READOUTS:
            c, t = scores[cam][r]["clean"], scores[cam][r]["trigger"]
            auroc = compute_auroc_high_is_triggered(c, t)
            results[cam][r] = {"n_clean": len(c), "n_trigger": len(t), "auroc": auroc,
                               "clean_mean": float(np.mean(c)), "trigger_mean": float(np.mean(t))}
            print(f"[*] {cam:7s} {r:10s}: n_clean={len(c)} n_trigger={len(t)} AUROC={auroc:.4f}  "
                  f"mean clean={np.mean(c):.4f} trigger={np.mean(t):.4f}")
    results["episodes"] = episodes

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
