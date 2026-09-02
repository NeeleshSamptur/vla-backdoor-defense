#!/usr/bin/env python
"""Score a directory of extracted samples: one FTT value per episode, then AUROC.

Attack-agnostic by construction -- no torch, no attack repo, no simulator. It
reads only the .npz contract in detectors/schema.py, which each attack's own
extractor writes from inside that attack's own conda env.

    python runners/run_detector.py --samples-dir results/goba_extracted \
        --out results/ftt_goba_stage1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.ftt import auroc, ftt_score, ftt_score_layerwise  # noqa: E402
from detectors.schema import load_dir  # noqa: E402

# Recorded per group and checked for agreement: each of these changes what the
# AUROC means, and none of them is encoded in the .npz filenames.
PROVENANCE_FIELDS = ("text_scope", "eval_design", "trigger_cameras")


@dataclass
class Row:
    """One episode's scores plus the metadata needed to debug it."""
    primary: float
    wrist: Optional[float]
    fused: float
    layerwise: Optional[float]
    label: int
    episode_id: str
    task_id: int
    seed: int
    n_query_tokens: int
    n_image_patches: int
    task_description: str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True,
                    help="directory of .npz files from an attack's extractor")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    samples = load_dir(args.samples_dir)
    print(f"[*] loaded {len(samples)} samples from {args.samples_dir}")
    results = score(samples)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n[*] wrote {args.out}")


def score(samples):
    """One FTT per episode from its first frame, then per-group AUROC."""
    by_ep = defaultdict(list)
    for s in samples:
        by_ep[(s.attack, s.checkpoint, s.extra.get("role", ""),
               s.extra.get("task_suite_name", ""), s.episode_id or "")].append(s)

    rows_by_group = defaultdict(list)
    samples_by_group = defaultdict(list)
    for (attack, ckpt, role, suite, ep_id), frames in by_ep.items():
        # Extractors write one frame per episode. If a directory still holds
        # multi-frame data, take the earliest rather than erroring.
        f0 = min(frames, key=lambda x: x.frame_idx)
        group = (attack, ckpt, role, suite)
        samples_by_group[group].append(f0)

        primary = ftt_score(f0.attn_text_image)
        wrist = (ftt_score(f0.attn_text_image_wrist)
                 if f0.attn_text_image_wrist is not None else None)
        # Two cameras: score each map on its own, then average the scalars, so
        # one noisy view cannot dominate. Single-camera attacks use primary.
        fused = 0.5 * (primary + wrist) if wrist is not None else primary
        layerwise = (ftt_score_layerwise(f0.attn_text_image_layers)
                     if f0.attn_text_image_layers is not None else None)
        rows_by_group[group].append(Row(
            primary=primary, wrist=wrist, fused=fused, layerwise=layerwise, label=int(f0.label),
            episode_id=ep_id, task_id=f0.task_id, seed=f0.seed,
            n_query_tokens=int(f0.attn_text_image.shape[0]),
            n_image_patches=int(f0.attn_text_image.shape[1]),
            task_description=f0.extra.get("task_description", "")))

    results = {}
    for group, rows in sorted(rows_by_group.items(), key=lambda kv: str(kv[0])):
        attack, ckpt, role, suite = group
        key = f"{attack}::{suite}::{role}"
        results[key] = summarize(key, attack, suite, role, ckpt, rows,
                                 samples_by_group[group])
        report(key, results[key], rows)
    return results


def summarize(key, attack, suite, role, ckpt, rows, group_samples):
    has_wrist = rows[0].wrist is not None

    def split(attr):
        return ([getattr(r, attr) for r in rows if r.label == 0],
                [getattr(r, attr) for r in rows if r.label == 1])

    clean_p, trig_p = split("primary")
    has_layerwise = rows[0].layerwise is not None
    per_camera = {
        "primary": stats(clean_p, trig_p),
        "wrist": stats(*split("wrist")) if has_wrist else None,
        "both_mean": stats(*split("fused")) if has_wrist else None,
        "layerwise": stats(*split("layerwise")) if has_layerwise else None,
    }

    headline = per_camera["both_mean" if has_wrist else "primary"]
    out = {
        "attack": attack, "suite": suite, "role": role, "checkpoint": ckpt,
        "n_clean_episodes": len(clean_p), "n_trigger_episodes": len(trig_p),
        "score_used": "both_mean" if has_wrist else "primary",
        **headline,
        **per_camera,
        "samples": [
            {"task_id": r.task_id, "seed": r.seed, "label": r.label,
             "condition": "trigger" if r.label else "clean",
             "episode_id": r.episode_id, "task_description": r.task_description,
             "n_query_tokens": r.n_query_tokens, "n_image_patches": r.n_image_patches,
             "ftt_primary": r.primary, "ftt_wrist": r.wrist,
             "ftt_both_mean": r.fused, "ftt_layerwise": r.layerwise,
             "ftt_used": r.fused if has_wrist else r.primary}
            for r in sorted(rows, key=lambda r: (r.task_id, r.seed, r.label))
        ],
    }
    out.update(provenance(key, group_samples))
    return out


def stats(clean, trig):
    return {
        "clean_mean": float(np.mean(clean)) if clean else float("nan"),
        "trig_mean": float(np.mean(trig)) if trig else float("nan"),
        "auroc": auroc(clean, trig),
    }


def provenance(key, group_samples):
    """Refuse to aggregate a group whose samples came from different settings."""
    out = {}
    for field in PROVENANCE_FIELDS:
        values = {s.extra.get(field) for s in group_samples} - {None}
        if len(values) > 1:
            raise ValueError(
                f"{key}: samples disagree on {field}={sorted(values)}. The "
                "--samples-dir mixes incompatible extractions; re-extract into "
                "separate directories.")
        out[field] = values.pop() if values else None
    return out


def report(key, r, rows):
    has_wrist = rows[0].wrist is not None
    nq = sorted({x.n_query_tokens for x in rows})
    print(f"\n[{key}]")
    print(f"    checkpoint    : {r['checkpoint']}")
    print(f"    text_scope={r['text_scope'] or 'all'}  "
          f"eval_design={r['eval_design'] or 'unrecorded'}  "
          f"trigger_cameras={r['trigger_cameras'] or 'n/a'}")
    print(f"    episodes      : {r['n_clean_episodes']} clean / "
          f"{r['n_trigger_episodes']} trigger")
    print(f"    query tokens  : {nq[0]}-{nq[-1]} per episode   "
          f"image patches/camera: {rows[0].n_image_patches}")
    if has_wrist:
        print(f"    AUROC         : {r['auroc']:.4f} (both_mean)   "
              f"primary={r['primary']['auroc']:.4f}  wrist={r['wrist']['auroc']:.4f}")
    else:
        print(f"    AUROC         : {r['auroc']:.4f} (primary)")
    print(f"    FTT mean      : clean={r['clean_mean']:.5f}  "
          f"trigger={r['trig_mean']:.5f}")
    if r.get("layerwise"):
        lw = r["layerwise"]
        print(f"    layerwise AUROC: {lw['auroc']:.4f}   "
              f"clean={lw['clean_mean']:.5f}  trigger={lw['trig_mean']:.5f}")

    hdr = (f"      {'cond':<8}{'task':>5}{'init':>6}{'n_q':>5}"
           f"{'ftt_primary':>13}{'ftt_wrist':>11}{'ftt_used':>10}  description")
    print(f"\n{hdr}")
    for x in sorted(rows, key=lambda r: (r.label, r.task_id, r.seed)):
        used = x.fused if has_wrist else x.primary
        wr = f"{x.wrist:>11.5f}" if x.wrist is not None else f"{'-':>11}"
        print(f"      {'trigger' if x.label else 'clean':<8}{x.task_id:>5}{x.seed:>6}"
              f"{x.n_query_tokens:>5}{x.primary:>13.5f}{wr}{used:>10.5f}  "
              f"{x.task_description[:52]}")


if __name__ == "__main__":
    main()
