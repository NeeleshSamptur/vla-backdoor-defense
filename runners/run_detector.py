#!/usr/bin/env python
"""Attack-agnostic FTT + AUROC over a directory of extracted samples.

Never imports torch, never imports an attack's own code, never touches a
conda env or a simulator. It only reads `.npz` files conforming to
detectors/schema.py -- produced separately by an attack's own extractor,
running in that attack's own environment (see adapters/<attack>/README.md).

Usage:
    python runners/run_detector.py --samples-dir results/badvla_extracted \
        --out results/ftt_badvla.json
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
from detectors.temporal import TemporalConfig, summarize_episodes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True,
                    help="directory of .npz files from an attack's extractor")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["static", "stage1", "temporal"], default="static",
                    help="static = Stage 1, one frame per scene, catches always-on "
                         "triggers (BadVLA/GoBA). temporal = Stage 2, per-frame "
                         "monitoring against each episode's own baseline, catches "
                         "delayed triggers (DropVLA). stage1 = per-EPISODE score, "
                         "averaging FTT over that episode's first --n-frames forward "
                         "passes (the two-stage cascade's gate).")
    ap.add_argument("--n-baseline", type=int, default=8)
    ap.add_argument("--k-persist", type=int, default=3)
    ap.add_argument("--z-threshold", type=float, default=4.0)
    ap.add_argument("--n-frames", type=int, default=5,
                    help="stage1: how many of each episode's leading frames to average")
    args = ap.parse_args()

    samples = load_dir(args.samples_dir)
    print(f"[*] loaded {len(samples)} samples from {args.samples_dir}")

    if args.mode == "temporal":
        run_temporal(samples, args)
        return
    if args.mode == "stage1":
        run_stage1(samples, args)
        return

    by_group = defaultdict(list)
    for s in samples:
        by_group[(s.attack, s.checkpoint, s.extra.get("role", ""))].append(s)

    results = {}
    for (attack, checkpoint, role), group in by_group.items():
        key = f"{attack}::{Path(checkpoint).name if role else checkpoint}"
        if role:
            key += f"::{role}"
        clean = [s for s in group if s.label == 0]
        trig = [s for s in group if s.label == 1]
        clean_scores = [ftt_score(s.attn_text_image) for s in clean]
        trig_scores = [ftt_score(s.attn_text_image) for s in trig]
        auroc_val = auroc(clean_scores, trig_scores)
        results[key] = {
            "attack": attack, "role": role,
            "n_clean": len(clean), "n_trigger": len(trig),
            "clean_mean": float(sum(clean_scores) / max(len(clean_scores), 1)),
            "trig_mean": float(sum(trig_scores) / max(len(trig_scores), 1)),
            "auroc": auroc_val,
            "checkpoint": group[0].checkpoint,
            "trigger_type": group[0].trigger_type,
        }
        r = results[key]
        print(f"\n[{key}] n_clean={r['n_clean']} n_trigger={r['n_trigger']} "
              f"trigger={r['trigger_type']}")
        print(f"    clean_mean={r['clean_mean']:.4f}  trig_mean={r['trig_mean']:.4f}")
        print(f"    AUROC(low=backdoor)={auroc_val:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n[*] saved -> {args.out}")


def run_stage1(samples, args):
    """Stage 1: one score per EPISODE = mean FTT over its first N passes.

    Distinct from --mode static, which treats every frame as an independent
    sample. Averaging within an episode is the point of the gate: it cuts
    per-frame noise before the clean/trigger decision, and it makes the unit
    of evaluation an episode (what a deployed defender actually screens),
    not a frame.
    """
    by_ep = defaultdict(list)
    for s in samples:
        key = (s.attack, s.checkpoint, s.extra.get("role", ""),
               s.extra.get("task_suite_name", ""), s.episode_id or "")
        by_ep[key].append(s)

    by_group = defaultdict(list)
    for (attack, ckpt, role, suite, ep_id), frames in by_ep.items():
        frames.sort(key=lambda x: x.frame_idx)
        used = frames[: args.n_frames] if args.n_frames else frames
        score_p = float(np.mean([ftt_score(f.attn_text_image) for f in used]))
        has_wrist = all(f.attn_text_image_wrist is not None for f in used)
        score_w = (float(np.mean([ftt_score(f.attn_text_image_wrist) for f in used]))
                   if has_wrist else None)
        # Fuse cameras by averaging the two FTT scores (not min, not concat
        # of patch columns). Each view is scored on its own map, then combined
        # so one noisy camera cannot dominate. GoBA / one-cam has no wrist.
        score_both = (0.5 * (score_p + score_w)) if score_w is not None else score_p
        label = int(any(f.label == 1 for f in frames))
        by_group[(attack, ckpt, role, suite)].append(
            (score_p, score_w, score_both, label, len(used), ep_id,
             frames[0].task_id, frames[0].seed))

    results = {}
    for (attack, ckpt, role, suite), rows in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        clean_p = [p for p, w, o, lb, *_ in rows if lb == 0]
        trig_p = [p for p, w, o, lb, *_ in rows if lb == 1]
        auroc_p = auroc(clean_p, trig_p)
        has_wrist = rows[0][1] is not None
        auroc_w = auroc_fused = None
        if has_wrist:
            clean_w = [w for p, w, o, lb, *_ in rows if lb == 0]
            trig_w = [w for p, w, o, lb, *_ in rows if lb == 1]
            clean_both = [o for p, w, o, lb, *_ in rows if lb == 0]
            trig_both = [o for p, w, o, lb, *_ in rows if lb == 1]
            auroc_w = auroc(clean_w, trig_w)
            auroc_fused = auroc(clean_both, trig_both)
        key = f"{attack}::{suite}::{role}"
        n_used = rows[0][4] if rows else 0
        results[key] = {
            "attack": attack, "suite": suite, "role": role,
            "checkpoint": ckpt,
            "n_clean_episodes": len(clean_p), "n_trigger_episodes": len(trig_p),
            "frames_averaged": n_used,
            "primary": {
                "clean_mean": float(np.mean(clean_p)) if clean_p else float("nan"),
                "trig_mean": float(np.mean(trig_p)) if trig_p else float("nan"),
                "auroc": auroc_p,
            },
            "wrist": None if not has_wrist else {
                "clean_mean": float(np.mean(clean_w)),
                "trig_mean": float(np.mean(trig_w)),
                "auroc": auroc_w,
            },
            "both_mean": None if not has_wrist else {
                "clean_mean": float(np.mean(clean_both)),
                "trig_mean": float(np.mean(trig_both)),
                "auroc": auroc_fused,
            },
            # Headline: fused mean of the two cameras when wrist is present;
            # primary-only for GoBA / one-cam extracts. AUROC is low FTT = backdoor.
            "score_used": "both_mean" if has_wrist else "primary",
            "clean_mean": float(np.mean(clean_both if has_wrist else clean_p)) if clean_p else float("nan"),
            "trig_mean": float(np.mean(trig_both if has_wrist else trig_p)) if trig_p else float("nan"),
            "auroc": auroc_fused if has_wrist else auroc_p,
            "samples": [
                {"task_id": tid, "seed": sd, "label": lb, "episode_id": eid,
                 "avg_ftt_primary": p, "avg_ftt_wrist": w, "avg_ftt_both_mean": o,
                 "frames_averaged": n_used}
                for p, w, o, lb, _, eid, tid, sd in sorted(rows, key=lambda r: (r[6], r[7], r[3]))
            ],
        }
        r = results[key]
        print(f"\n[{key}]")
        print(f"    AUROC={r['auroc']:.4f}  ({r['score_used']})  "
              f"clean_mean={r['clean_mean']:.5f}  trigger_mean={r['trig_mean']:.5f}")
        for p, w, o, lb, _, eid, tid, sd in sorted(rows, key=lambda r: (r[3], r[6], r[7])):
            cond = "trigger" if lb == 1 else "clean  "
            ftt = o if w is not None else p
            print(f"      {cond}  task={tid:2d} seed={sd:2d}  ftt={ftt:.5f}   {eid}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n[*] saved -> {args.out}")


def run_temporal(samples, args):
    """Stage 2: reconstruct episodes in frame order, score each per-frame."""
    cfg = TemporalConfig(n_baseline=args.n_baseline, k_persist=args.k_persist,
                         z_threshold=args.z_threshold)

    by_ep = defaultdict(list)
    for s in samples:
        if s.episode_id is None:
            raise ValueError(
                "temporal mode needs episode_id/frame_idx on every sample; this "
                "extractor produced single-frame samples (use --mode static)")
        by_ep[(s.attack, s.checkpoint, s.episode_id)].append(s)

    episodes = []
    for (attack, ckpt, ep_id), frames in by_ep.items():
        frames.sort(key=lambda x: x.frame_idx)
        scores = [ftt_score(f.attn_text_image) for f in frames]
        # An episode counts as triggered if ANY frame in it is triggered --
        # for a delayed trigger most of its frames are legitimately clean.
        label = int(any(f.label == 1 for f in frames))
        act = next((f.activation_frame for f in frames
                    if f.activation_frame is not None), None)
        episodes.append({"scores": scores, "label": label, "activation_frame": act,
                         "attack": attack, "checkpoint": ckpt, "episode_id": ep_id})

    results = {}
    by_attack = defaultdict(list)
    for ep in episodes:
        by_attack[(ep["attack"], ep["checkpoint"])].append(ep)

    for (attack, ckpt), eps in by_attack.items():
        r = summarize_episodes(eps, cfg)
        key = f"{attack}::{Path(ckpt).name}"
        results[key] = r
        print(f"\n[{key}]  (Stage 2 / temporal)")
        print(f"    episodes: {r['n_clean_episodes']} clean, "
              f"{r['n_triggered_episodes']} triggered")
        print(f"    episode AUROC     : {r['episode_auroc']:.4f}")
        print(f"    detection rate    : {r['detection_rate']:.3f}")
        print(f"    false alarm rate  : {r['false_alarm_rate']:.3f}  (per clean episode)")
        print(f"    median latency    : {r['median_latency_frames']} frames "
              f"(n={r['n_latency_samples']})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n[*] saved -> {args.out}")


if __name__ == "__main__":
    main()
