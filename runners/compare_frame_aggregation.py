#!/usr/bin/env python
"""Does averaging FTT over the first N frames beat using frame 0 alone?

Answers that on ONE multi-frame extraction directory, so the episodes, scenes
and checkpoints are identical and the ONLY variable is how the per-episode
score is aggregated:

    frame0   ftt(frame 0)                      -- what the single-frame
                                                  extractor produces
    mean_k   mean(ftt(frame 0..k-1))           -- the old averaging behaviour

Motivation: removing averaging demonstrably cost BadVLA nothing (AUROC 1.000
either way), but BadVLA's separation was ~7x. GoBA's is ~1.4x, so noise
suppression could matter there and the BadVLA result does not transfer.

Sanity check worth reading in the output: `frame0` here should closely match
a separate single-frame run of the same config, since the seed and env reset
sequence are identical.

Usage:
    python runners/compare_frame_aggregation.py --samples-dir results/goba_5frame_extracted
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.ftt import auroc, ftt_score  # noqa: E402
from detectors.schema import load_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--max-frames", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    samples = load_dir(args.samples_dir)
    print(f"[*] loaded {len(samples)} samples from {args.samples_dir}")

    by_ep = defaultdict(list)
    for s in samples:
        key = (s.attack, s.checkpoint, s.extra.get("role", ""),
               s.extra.get("task_suite_name", ""), s.episode_id or "")
        by_ep[key].append(s)

    by_group = defaultdict(list)
    for (attack, ckpt, role, suite, ep_id), frames in by_ep.items():
        frames.sort(key=lambda x: x.frame_idx)
        per_frame_p = [ftt_score(f.attn_text_image) for f in frames]
        has_w = frames[0].attn_text_image_wrist is not None
        per_frame_w = ([ftt_score(f.attn_text_image_wrist) for f in frames]
                       if has_w else None)

        def fuse(i):
            """Per-frame score, fusing cameras the same way run_detector does."""
            return (0.5 * (per_frame_p[i] + per_frame_w[i])) if has_w else per_frame_p[i]

        fused = [fuse(i) for i in range(len(frames))]
        by_group[(attack, suite, role)].append((fused, int(frames[0].label)))

    results = {}
    hdr = f"{'group':<42}{'n_frames':>9}{'frame0':>9}{'mean_k':>9}{'delta':>8}"
    print("\n" + hdr); print("-" * len(hdr))
    for (attack, suite, role), rows in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        nf = min(min(len(f) for f, _ in rows), args.max_frames)
        clean0 = [f[0] for f, lb in rows if lb == 0]
        trig0 = [f[0] for f, lb in rows if lb == 1]
        cleank = [float(np.mean(f[:nf])) for f, lb in rows if lb == 0]
        trigk = [float(np.mean(f[:nf])) for f, lb in rows if lb == 1]
        a0, ak = auroc(clean0, trig0), auroc(cleank, trigk)
        key = f"{attack}::{suite}::{role}"
        results[key] = {"n_frames_used": nf, "auroc_frame0": a0,
                        "auroc_mean_k": ak, "delta": ak - a0,
                        "n_clean": len(clean0), "n_trigger": len(trig0)}
        print(f"{key:<42}{nf:>9}{a0:>9.4f}{ak:>9.4f}{ak-a0:>+8.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n[*] saved -> {args.out}")


if __name__ == "__main__":
    main()
