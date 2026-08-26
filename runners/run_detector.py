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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.ftt import auroc_both_polarities, ftt_score  # noqa: E402
from detectors.schema import load_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True,
                    help="directory of .npz files from an attack's extractor")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    samples = load_dir(args.samples_dir)
    print(f"[*] loaded {len(samples)} samples from {args.samples_dir}")

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
        auroc = auroc_both_polarities(clean_scores, trig_scores)

        best_polarity = "low_is_backdoor" if auroc["low_is_backdoor"] >= auroc["high_is_backdoor"] else "high_is_backdoor"
        results[key] = {
            "attack": attack, "role": role,
            "n_clean": len(clean), "n_trigger": len(trig),
            "clean_mean": float(sum(clean_scores) / max(len(clean_scores), 1)),
            "trig_mean": float(sum(trig_scores) / max(len(trig_scores), 1)),
            "auroc": auroc,
            "best_polarity": best_polarity,
            "best_auroc": auroc[best_polarity],
            "checkpoint": group[0].checkpoint,
            "trigger_type": group[0].trigger_type,
        }
        r = results[key]
        print(f"\n[{key}] n_clean={r['n_clean']} n_trigger={r['n_trigger']} "
              f"trigger={r['trigger_type']}")
        print(f"    clean_mean={r['clean_mean']:.4f}  trig_mean={r['trig_mean']:.4f}")
        print(f"    AUROC(high=backdoor)={auroc['high_is_backdoor']:.4f}  "
              f"AUROC(low=backdoor)={auroc['low_is_backdoor']:.4f}")
        print(f"    => best polarity: {best_polarity}  (AUROC={r['best_auroc']:.4f})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n[*] saved -> {args.out}")


if __name__ == "__main__":
    main()
