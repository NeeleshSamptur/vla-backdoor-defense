#!/usr/bin/env python
"""Score a directory of action-token attention extractions: FTT + AUROC.

Reads the raw .npz files written by adapters/goba/extract_action2img_ftt.py
(key: attn_action_image_all_layers, shape [n_layers, action_dim, n_patches]).
NOT read through detectors/schema.py -- that schema's attn_text_image is fixed
2D, and these dumps are 3D by design (see that extractor's docstring).

Per episode: average the attention map over all LLM layers first (mean over
axis 0), giving one [action_dim, n_patches] map, then ftt_score on THAT --
same convention as the text-token pipeline's --layer-agg average. One number
per episode, then one AUROC (clean vs. trigger) per group. This does NOT
score each layer separately -- that was a wrong turn in an earlier version of
this script; averaging first is what was asked for.

Groups by (attack, checkpoint, role, task_suite_name), same as
runners/run_detector.py, since those are the axes that change what an AUROC
means and none of them are recoverable from the filename alone.

    python runners/run_action_detector.py \\
        --samples-dir results/goba_action_extracted \\
        --out results/ftt_action_goba.json
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


def load_action_npz(path: Path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    return z["attn_action_image_all_layers"], int(z["label"]), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True,
                    help="directory of .npz files from extract_action2img_ftt.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(Path(args.samples_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files under {args.samples_dir}")
    print(f"[*] loaded {len(files)} samples from {args.samples_dir}")

    by_group = defaultdict(list)
    for f in files:
        rows, label, meta = load_action_npz(f)
        key = (meta.get("attack"), meta.get("checkpoint"),
               meta.get("role"), meta.get("task_suite_name"))
        by_group[key].append((rows, label, f.name))

    results = {}
    for key, items in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        attack, ckpt, role, suite = key

        clean_scores, trig_scores = [], []
        for rows, label, _name in items:
            avg_map = rows.mean(axis=0)  # [action_dim, n_patches], mean over layers
            score = ftt_score(avg_map)
            (trig_scores if label else clean_scores).append(score)

        gkey = f"{attack}::{suite}::{role}"
        results[gkey] = {
            "attack": attack, "checkpoint": ckpt, "role": role, "suite": suite,
            "n_clean_episodes": len(clean_scores),
            "n_trigger_episodes": len(trig_scores),
            "clean_mean": float(np.mean(clean_scores)) if clean_scores else float("nan"),
            "trig_mean": float(np.mean(trig_scores)) if trig_scores else float("nan"),
            "auroc": auroc(clean_scores, trig_scores),
        }
        report(gkey, results[gkey])

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n[*] wrote {args.out}")


def report(key, r):
    print(f"\n[{key}]")
    print(f"    checkpoint : {r['checkpoint']}")
    print(f"    episodes   : {r['n_clean_episodes']} clean / {r['n_trigger_episodes']} trigger")
    print(f"    AUROC      : {r['auroc']:.4f}")
    print(f"    FTT mean   : clean={r['clean_mean']:.5f}  trigger={r['trig_mean']:.5f}")


if __name__ == "__main__":
    main()
