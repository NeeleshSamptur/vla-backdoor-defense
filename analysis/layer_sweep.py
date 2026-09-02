#!/usr/bin/env python
"""Per-layer AUROC sweep across attacks, from --dump-all-layers extractions.

Reads the raw (non-schema) .npz files written by adapters/goba and
adapters/backdoorvla_openvla_oft when run with --dump-all-layers (each holds
attn_all_layers: [n_layers, n_query, n_patches], not the collapsed single/
averaged matrix run_detector.py expects). For every layer independently,
computes the same FTT statistic and AUROC detectors/ftt.py uses, then reports
where the two attacks' per-layer AUROC curves are BOTH high -- the candidate
"generalizes across attacks" layer(s) -- rather than each attack's individual
best layer, which may not be the same layer at all.

Pure numpy + detectors/ftt.py; no torch/model loading needed here, this only
reads what extraction already saved.

Usage:
    python analysis/layer_sweep.py \
        --dir goba=results/goba_layersweep_libero_object \
        --dir backdoorvla_oft=results/backdoorvla_oft_layersweep_libero_object \
        --out results/layer_sweep_libero_object.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

DEFENSE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))

from detectors.ftt import auroc, ftt_score  # noqa: E402


def load_all_layers(d: pathlib.Path, key: str = "attn_all_layers", role: str = "attack"):
    """Returns (per_layer_attn, labels): per_layer_attn is [n_samples, n_layers, Q, P]
    is NOT assumed -- Q varies per episode (different instructions -> different
    token counts), so this returns a list of [n_layers, Q_i, P] arrays plus a
    parallel label array; the per-layer FTT score is computed per-sample before
    any stacking.

    Filters to meta_json["role"] == role (default "attack"): a directory holds
    BOTH the backdoored checkpoint's episodes (role=attack, the real clean-vs-
    trigger comparison) AND the clean_baseline checkpoint's episodes (role=
    clean_baseline, a negative control run on the same clean/trigger scenes
    with a benign model -- expected to look like noise). run_detector.py keeps
    these as separate groups (grouped by checkpoint); mixing them here would
    silently blend the real signal with the negative control and deflate every
    AUROC number for no attack-related reason.
    """
    files = sorted(d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz under {d}")
    attns, labels = [], []
    n_layers = None
    skipped = 0
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        if meta.get("role") != role:
            skipped += 1
            continue
        a = z[key]  # [n_layers, Q, P]
        if n_layers is None:
            n_layers = a.shape[0]
        assert a.shape[0] == n_layers, f"{f}: layer count {a.shape[0]} != {n_layers}"
        attns.append(a)
        labels.append(int(z["label"]))
    if skipped:
        print(f"[*] {d}: skipped {skipped} files with role != {role!r}")
    return attns, np.asarray(labels), n_layers


def per_layer_auroc(attns, labels, n_layers) -> np.ndarray:
    """FTT + AUROC computed independently at every layer."""
    scores = np.full((len(attns), n_layers), np.nan, dtype=np.float64)
    for i, a in enumerate(attns):
        for layer in range(n_layers):
            scores[i, layer] = ftt_score(a[layer])
    aurocs = np.full(n_layers, np.nan)
    for layer in range(n_layers):
        clean = scores[labels == 0, layer]
        trig = scores[labels == 1, layer]
        aurocs[layer] = auroc(clean, trig)
    return aurocs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, metavar="NAME=PATH",
                     help="one per attack, e.g. goba=results/goba_layersweep_libero_object")
    ap.add_argument("--key", default="attn_all_layers",
                     help="npz key holding the [n_layers,Q,P] stack (primary camera).")
    ap.add_argument("--role", default="attack",
                     help="meta_json role to keep (default 'attack' -- the "
                          "backdoored checkpoint's clean-vs-trigger episodes; "
                          "'clean_baseline' files are dropped since they're a "
                          "negative control, not part of the AUROC comparison).")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--common-threshold", type=float, default=0.85,
                     help="a layer counts as 'good for both' when every attack's "
                          "AUROC at that layer is >= this.")
    args = ap.parse_args()

    per_attack = {}
    for spec in args.dir:
        name, path = spec.split("=", 1)
        attns, labels, n_layers = load_all_layers(pathlib.Path(path), args.key, args.role)
        aurocs = per_layer_auroc(attns, labels, n_layers)
        per_attack[name] = aurocs
        print(f"[*] {name}: {len(attns)} samples ({int((labels==0).sum())} clean / "
              f"{int((labels==1).sum())} trigger), {n_layers} layers")

    n_layers_all = {len(v) for v in per_attack.values()}
    if len(n_layers_all) != 1:
        print(f"[!] attacks have different layer counts: "
              f"{ {k: len(v) for k, v in per_attack.items()} } -- "
              f"reporting by layer INDEX, which may not mean the same relative "
              f"depth across architectures.")
    n_layers = max(n_layers_all)

    print(f"\n{'layer':>5}  " + "  ".join(f"{name:>12}" for name in per_attack) + "   min(all)")
    row_mins = {}
    for layer in range(n_layers):
        vals = {name: (aurocs[layer] if layer < len(aurocs) else float("nan"))
                for name, aurocs in per_attack.items()}
        row_min = min(vals.values())
        row_mins[layer] = row_min
        print(f"{layer:>5}  " + "  ".join(f"{vals[name]:>12.4f}" for name in per_attack)
              + f"   {row_min:.4f}")

    common_layers = sorted([l for l, m in row_mins.items() if m >= args.common_threshold],
                            key=lambda l: -row_mins[l])
    ranked = sorted(row_mins.items(), key=lambda kv: -kv[1])
    best_common_layer, best_common_score = ranked[0]

    # Quick-pick ranking: every layer sorted by its worst-case (min-across-
    # attacks) AUROC, best first -- this is the one table to look at to
    # answer "which layer(s) should I actually use."
    print(f"\n[*] layers ranked by worst-case (min-across-attacks) AUROC, best first:")
    for rank, (layer, m) in enumerate(ranked[:10], start=1):
        per_attack_str = ", ".join(f"{name}={aurocs[layer]:.4f}" for name, aurocs in per_attack.items())
        print(f"    #{rank:<2} layer {layer:>2}   min={m:.4f}   ({per_attack_str})")

    print(f"\n[*] best single layer by worst-case (min-across-attacks) AUROC: "
          f"layer {best_common_layer} (min AUROC = {best_common_score:.4f})")
    if common_layers:
        print(f"[*] layers with every attack's AUROC >= {args.common_threshold} "
              f"(best first): {common_layers}")
    else:
        print(f"[*] NO layer reaches {args.common_threshold} on every attack simultaneously.")

    if args.out:
        out = {
            "per_attack_auroc": {k: v.tolist() for k, v in per_attack.items()},
            "ranked_layers_by_min_auroc": [{"layer": l, "min_auroc": m} for l, m in ranked],
            "best_common_layer": best_common_layer,
            "best_common_min_auroc": best_common_score,
            "common_threshold": args.common_threshold,
            "layers_meeting_threshold": common_layers,
        }
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
