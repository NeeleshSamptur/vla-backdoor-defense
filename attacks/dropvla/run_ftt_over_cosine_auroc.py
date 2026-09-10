#!/usr/bin/env python
"""DropVLA: combine the two DropVLA scores into one -- per-episode

    ratio = FTT(original frame) / cosine_distance(original vs cropped)

-- and score that with AUROC. No GPU: both numerators and denominators are
read from the per-episode records the two GPU scripts already wrote.

    numerator    attacks/dropvla/run_crop_ftt_auroc.py ->
                 episodes[i][cond]["<camera>_uncropped"], the desc_only
                 text-to-image FTT of the frame as-is. LOW = triggered.
    denominator  attacks/dropvla/run_crop_hidden_auroc.py ->
                 episodes[i][cond]["cosine_distance"], how far the final
                 LLM layer's action-token activations move when the frame
                 is center-cropped. HIGH = triggered.

Both polarities push the ratio the same way, so the combined score is LOW =
triggered and goes through attacks.common.compute_auroc unchanged (the same
polarity FTT itself uses).

The two source runs must be over the same episodes for the division to be
meaningful; the two JSONs are joined on (task_id, init_index) and the join
is REFUSED unless every matched pair also agrees on frame_idx, which is the
rollout's own witness that both runs captured the identical frame. Episodes
present in only one file are reported and dropped rather than silently
mismatched.

The ratio's own AUROC is printed next to each component's, so it is
immediately visible whether combining helped or hurt rather than being
taken on faith.

Usage (no GPU, no attack repo -- plain python with numpy + sklearn):
    python attacks/dropvla/run_ftt_over_cosine_auroc.py \
        --ftt-json results/dropvla_crop_ftt/dropvla_crop_ftt_auroc.json \
        --hidden-json results/dropvla_crop_hidden_auroc.json \
        --out results/dropvla_ftt_over_cosine_auroc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)

from attacks.common import EPS, compute_auroc

CONDITIONS = ("clean", "trigger")
CAMERAS = ("primary", "wrist")


def index_episodes(records, source):
    """Map (task_id, init_index) -> record, refusing duplicate keys."""
    out = {}
    for r in records:
        key = (r["task_id"], r["init_index"])
        if key in out:
            raise SystemExit(f"{source}: duplicate episode {key}")
        out[key] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ftt-json", required=True,
                    help="results JSON from run_crop_ftt_auroc.py (supplies the numerator).")
    ap.add_argument("--hidden-json", required=True,
                    help="results JSON from run_crop_hidden_auroc.py (supplies the denominator).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ftt_run = json.load(open(args.ftt_json))
    hid_run = json.load(open(args.hidden_json))
    for field in ("task_suite_name", "trigger_mode", "crop_scale", "checkpoint"):
        if ftt_run.get(field) != hid_run.get(field):
            raise SystemExit(
                f"the two runs disagree on {field}: {ftt_run.get(field)!r} vs "
                f"{hid_run.get(field)!r} -- they are not the same experiment, refusing to combine.")

    ftt_eps = index_episodes(ftt_run["episodes"], args.ftt_json)
    hid_eps = index_episodes(hid_run["episodes"], args.hidden_json)
    only_ftt = sorted(set(ftt_eps) - set(hid_eps))
    only_hid = sorted(set(hid_eps) - set(ftt_eps))
    if only_ftt or only_hid:
        print(f"[!] dropping episodes present in only one run: "
              f"{len(only_ftt)} only in --ftt-json, {len(only_hid)} only in --hidden-json")
    keys = sorted(set(ftt_eps) & set(hid_eps))
    if not keys:
        raise SystemExit("no episodes in common between the two runs")

    for key in keys:
        if ftt_eps[key]["frame_idx"] != hid_eps[key]["frame_idx"]:
            raise SystemExit(
                f"episode {key}: the two runs captured different frames "
                f"({ftt_eps[key]['frame_idx']} vs {hid_eps[key]['frame_idx']}) -- the rollouts "
                "diverged, so their per-episode scores cannot be divided.")

    ratio = {cam: {c: [] for c in CONDITIONS} for cam in CAMERAS}
    ftt_only = {cam: {c: [] for c in CONDITIONS} for cam in CAMERAS}
    cos_only = {c: [] for c in CONDITIONS}
    episodes = []
    for key in keys:
        rec = {"task_id": key[0], "init_index": key[1], "frame_idx": ftt_eps[key]["frame_idx"]}
        for cond in CONDITIONS:
            cos = hid_eps[key][cond]["cosine_distance"]
            cos_only[cond].append(cos)
            rec[cond] = {"cosine_distance": cos}
            for cam in CAMERAS:
                f = ftt_eps[key][cond][f"{cam}_uncropped"]
                r = f / max(cos, EPS)
                ftt_only[cam][cond].append(f)
                ratio[cam][cond].append(r)
                rec[cond][f"{cam}_ftt_uncropped"] = f
                rec[cond][f"{cam}_ftt_over_cosine"] = r
        episodes.append(rec)

    n = len(keys)
    print(f"[*] {n} episodes matched (n_clean={n}, n_trigger={n}), "
          f"suite={ftt_run['task_suite_name']} trigger_mode={ftt_run['trigger_mode']} "
          f"crop_scale={ftt_run['crop_scale']}")
    results = {
        "attack": "dropvla",
        "checkpoint": ftt_run["checkpoint"],
        "task_suite_name": ftt_run["task_suite_name"],
        "trigger_mode": ftt_run["trigger_mode"],
        "crop_scale": ftt_run["crop_scale"],
        "n_clean": n, "n_trigger": n,
        "score": "FTT(original frame) / cosine_distance(original vs cropped last-layer activations)",
        "polarity": "low = triggered (both components push the ratio down for a triggered frame)",
        "sources": {"ftt_json": args.ftt_json, "hidden_json": args.hidden_json},
    }

    cos_auroc = compute_auroc([-s for s in cos_only["clean"]], [-s for s in cos_only["trigger"]])
    results["component_cosine_distance"] = {
        "auroc": cos_auroc, "polarity": "high = triggered",
        "clean_mean": float(np.mean(cos_only["clean"])),
        "trigger_mean": float(np.mean(cos_only["trigger"]))}
    print(f"[*] component  cosine_distance         : AUROC={cos_auroc:.4f}")

    for cam in CAMERAS:
        f_auroc = compute_auroc(ftt_only[cam]["clean"], ftt_only[cam]["trigger"])
        r_auroc = compute_auroc(ratio[cam]["clean"], ratio[cam]["trigger"])
        results[f"component_ftt_{cam}"] = {
            "auroc": f_auroc, "polarity": "low = triggered",
            "clean_mean": float(np.mean(ftt_only[cam]["clean"])),
            "trigger_mean": float(np.mean(ftt_only[cam]["trigger"]))}
        results[f"ftt_over_cosine_{cam}"] = {
            "auroc": r_auroc, "polarity": "low = triggered",
            "clean_mean": float(np.mean(ratio[cam]["clean"])),
            "trigger_mean": float(np.mean(ratio[cam]["trigger"]))}
        print(f"[*] component  ftt_{cam:7s}            : AUROC={f_auroc:.4f}")
        print(f"[*] COMBINED   ftt_{cam}/cosine     : AUROC={r_auroc:.4f}  "
              f"mean clean={np.mean(ratio[cam]['clean']):.4f} "
              f"trigger={np.mean(ratio[cam]['trigger']):.4f}")

    results["episodes"] = episodes
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
