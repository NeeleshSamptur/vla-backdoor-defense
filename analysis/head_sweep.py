#!/usr/bin/env python
"""Per-head AUROC sweep within one chosen layer, across attacks.

Same idea as analysis/layer_sweep.py but along the head axis instead of the
layer axis: reads the raw (non-schema) .npz files written by adapters/goba and
adapters/backdoorvla_openvla_oft when run with --dump-heads-layer <L> (each
holds attn_heads: [n_heads, n_query, n_patches] for that one layer, no head
averaging). For every head independently, computes the same FTT statistic and
AUROC detectors/ftt.py uses, so you can see whether a handful of heads carry
most of the separation the head-average was diluting.

Usage:
    python analysis/head_sweep.py \
        --dir goba=results/goba_headsweep_libero_object_finallayer \
        --dir backdoorvla_oft=results/backdoorvla_oft_headsweep_libero_object_finallayer \
        --out results/head_sweep_libero_object_finallayer.json
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


def load_heads(d: pathlib.Path, key: str = "attn_heads", role: str = "attack"):
    """Mirrors analysis/layer_sweep.py's load_all_layers: filters to one
    checkpoint's role (default "attack", the backdoored checkpoint's clean-vs-
    trigger episodes) so the clean_baseline negative control never gets
    silently blended in."""
    files = sorted(d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz under {d}")
    attns, labels = [], []
    n_heads = None
    skipped = 0
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        if meta.get("role") != role:
            skipped += 1
            continue
        a = z[key]  # [n_heads, Q, P]
        if n_heads is None:
            n_heads = a.shape[0]
        assert a.shape[0] == n_heads, f"{f}: head count {a.shape[0]} != {n_heads}"
        attns.append(a)
        labels.append(int(z["label"]))
    if skipped:
        print(f"[*] {d}: skipped {skipped} files with role != {role!r}")
    return attns, np.asarray(labels), n_heads


def per_head_auroc(attns, labels, n_heads) -> np.ndarray:
    scores = np.full((len(attns), n_heads), np.nan, dtype=np.float64)
    for i, a in enumerate(attns):
        for h in range(n_heads):
            scores[i, h] = ftt_score(a[h])
    aurocs = np.full(n_heads, np.nan)
    for h in range(n_heads):
        clean = scores[labels == 0, h]
        trig = scores[labels == 1, h]
        aurocs[h] = auroc(clean, trig)
    return aurocs, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--key", default="attn_heads",
                     help="npz key holding the [n_heads,Q,P] stack (primary camera).")
    ap.add_argument("--role", default="attack")
    ap.add_argument("--out", default=None)
    ap.add_argument("--common-threshold", type=float, default=0.85)
    args = ap.parse_args()

    per_attack = {}
    for spec in args.dir:
        name, path = spec.split("=", 1)
        attns, labels, n_heads = load_heads(pathlib.Path(path), args.key, args.role)
        aurocs, _ = per_head_auroc(attns, labels, n_heads)
        per_attack[name] = aurocs
        print(f"[*] {name}: {len(attns)} samples ({int((labels==0).sum())} clean / "
              f"{int((labels==1).sum())} trigger), {n_heads} heads")

    n_heads_all = {len(v) for v in per_attack.values()}
    n_heads = max(n_heads_all)
    if len(n_heads_all) != 1:
        print(f"[!] attacks have different head counts: "
              f"{ {k: len(v) for k, v in per_attack.items()} }")

    print(f"\n{'head':>5}  " + "  ".join(f"{name:>12}" for name in per_attack) + "   min(all)")
    row_mins = {}
    for h in range(n_heads):
        vals = {name: (aurocs[h] if h < len(aurocs) else float("nan"))
                for name, aurocs in per_attack.items()}
        row_min = min(vals.values())
        row_mins[h] = row_min
        print(f"{h:>5}  " + "  ".join(f"{vals[name]:>12.4f}" for name in per_attack)
              + f"   {row_min:.4f}")

    ranked = sorted(row_mins.items(), key=lambda kv: -kv[1])
    best_head, best_score = ranked[0]
    common_heads = sorted([h for h, m in row_mins.items() if m >= args.common_threshold],
                           key=lambda h: -row_mins[h])

    print(f"\n[*] heads ranked by worst-case (min-across-attacks) AUROC, best first:")
    for rank, (h, m) in enumerate(ranked[:10], start=1):
        per_attack_str = ", ".join(f"{name}={aurocs[h]:.4f}" for name, aurocs in per_attack.items())
        print(f"    #{rank:<2} head {h:>2}   min={m:.4f}   ({per_attack_str})")

    print(f"\n[*] best single head by worst-case AUROC: head {best_head} (min AUROC = {best_score:.4f})")
    if common_heads:
        print(f"[*] heads with every attack's AUROC >= {args.common_threshold} (best first): {common_heads}")
    else:
        print(f"[*] NO head reaches {args.common_threshold} on every attack simultaneously.")

    if args.out:
        out = {
            "per_attack_auroc": {k: v.tolist() for k, v in per_attack.items()},
            "ranked_heads_by_min_auroc": [{"head": h, "min_auroc": m} for h, m in ranked],
            "best_head": best_head,
            "best_min_auroc": best_score,
            "common_threshold": args.common_threshold,
            "heads_meeting_threshold": common_heads,
        }
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[*] wrote {args.out}")


if __name__ == "__main__":
    main()
