#!/usr/bin/env python
"""BadVLA: FTT -> AUROC on the frame as-is AND on the center-cropped frame.

The crop variant of attacks/badvla_white_patch/run_ftt_auroc.py, and the
BadVLA counterpart of attacks/dropvla/run_crop_ftt_auroc.py. Everything
except the crop -- model loading, the LIBERO rollout, observation prep, the
white-patch trigger, the desc_only attention capture, the disjoint
init-state convention -- is IMPORTED from that FTT script, so the only
difference between the two AUROCs reported here is the crop.

Per episode the desc_only text-to-image attention is captured twice:

  uncropped -- the frame exactly as prepare_observation produced it (with
               the trigger patch drawn, on triggered episodes)
  cropped   -- the SAME frame, trigger included, then passed through
               attacks.common.center_crop_resize

Order matters and is deliberate: the trigger is drawn first and the crop
applied to the already-triggered image, because that is what a deployed
input filter sees -- the camera hands over whatever is in front of it and
the filter crops that.

Both are scored with attacks.common.compute_ftt (normalize each layer,
average all layers, mean L2 deviation from the row-mean) and
compute_auroc, so the uncropped number must reproduce run_ftt_auroc.py's
for the same seed/tasks/seeds -- it is the same code on the same frames,
recomputed here only so both halves of the comparison come from one run.

Unlike DropVLA (a 5px dot in a CORNER, deleted by a center crop, which
makes its cropped clean and triggered frames pixel-identical and its
cropped AUROC 0.5 by construction), BadVLA's trigger is
`add_trigger_img(trigger_size=0.10, trigger_position="center")` -- a white
square covering 10% of the frame, in the MIDDLE, which a center crop keeps
and in fact magnifies. So the cropped number here is a real measurement,
not a degenerate one, and it says what FTT sees when the trigger is
enlarged rather than removed.

Usage (BadVLA's own conda env, same as run_ftt_auroc.py):
    conda activate openvla-oft
    cd /home/grads/nsamptur/vla_bkd_def/BadVLA
    python .../attacks/badvla_white_patch/run_crop_ftt_auroc.py \
        --checkpoint <BadVLA white-patch checkpoint> --task-suite-name libero_object \
        --out ../vla-backdoor-defense/results/libero_object/badvla_white_patch_crop_ftt_auroc.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from libero.libero import benchmark

from attacks.badvla_white_patch.run_ftt_auroc import (
    NUM_STEPS_WAIT,
    VALID_SUITES,
    capture_text_to_image_attention,
    load_model,
)
from attacks.common import center_crop_resize, compute_auroc, compute_ftt

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import add_trigger_img, prepare_observation
from experiments.robot.robot_utils import get_image_resize_size


def triggered_then_cropped(observation, trigger: bool, trigger_size: float, crop_scale: float):
    """A copy of `observation` whose camera images have the trigger drawn (if
    this is a triggered episode) and are THEN cropped -- so the capture below
    can be called with trigger=False and still see the real triggered pixels,
    in the order a deployed filter would see them."""
    out = dict(observation)
    for key in ("full_image", "wrist_image"):
        img = observation[key].copy()
        if trigger:
            img = add_trigger_img(img, trigger_size=trigger_size,
                                  trigger_position="center", trigger_color=255)
        out[key] = center_crop_resize(img, crop_scale)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-scale", type=float, default=0.8)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7, help="base curated-init-state index")
    ap.add_argument("--trigger-size", type=float, default=0.10)
    args = ap.parse_args()

    cfg, processor, vla, proprio_projector = load_model(args.checkpoint, args.task_suite_name)
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    variants = ("uncropped", "cropped")
    scores = {v: {"clean": [], "trigger": []} for v in variants}
    episodes = []
    cond_offset = {"clean": 0, "trigger": args.n_seeds}

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        n_avail = init_states.shape[0]
        env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        try:
            for cond, trig in (("clean", False), ("trigger", True)):
                for seed_k in range(args.n_seeds):
                    episode_idx = args.seed + cond_offset[cond] + seed_k
                    if episode_idx >= n_avail:
                        print(f"    [!] task={task_id}: only {n_avail} init states, "
                              f"skipping {cond} index {episode_idx}")
                        continue
                    env.reset()
                    obs = env.set_init_state(init_states[episode_idx])
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    observation, _ = prepare_observation(obs, resize_size)

                    ftt = {}
                    # uncropped: the FTT script's own path, trigger drawn inside the capture
                    attn = capture_text_to_image_attention(
                        vla, processor, proprio_projector, cfg, observation, desc,
                        trig, args.trigger_size)
                    ftt["uncropped"] = compute_ftt(attn)
                    # cropped: trigger drawn first, then cropped, so trigger=False here
                    attn_c = capture_text_to_image_attention(
                        vla, processor, proprio_projector, cfg,
                        triggered_then_cropped(observation, trig, args.trigger_size, args.crop_scale),
                        desc, False, args.trigger_size)
                    ftt["cropped"] = compute_ftt(attn_c)

                    for v in variants:
                        scores[v][cond].append(ftt[v])
                    episodes.append({"task_id": task_id, "init_index": episode_idx,
                                     "condition": cond, "label": int(trig),
                                     "ftt_uncropped": ftt["uncropped"], "ftt_cropped": ftt["cropped"]})
                    print(f"    task={task_id} seed={episode_idx} {cond:8s} "
                          f"ftt {ftt['uncropped']:.5f} -> {ftt['cropped']:.5f}", flush=True)
        finally:
            env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()

    results = {"attack": "badvla_white_patch", "checkpoint": args.checkpoint,
               "task_suite_name": args.task_suite_name, "crop_scale": args.crop_scale,
               "trigger_size": args.trigger_size, "trigger_position": "center",
               "polarity": "low FTT = triggered",
               "eval_design": "disjoint init states between clean and trigger"}
    for v in variants:
        c, t = scores[v]["clean"], scores[v]["trigger"]
        auroc = compute_auroc(c, t)
        results[v] = {"n_clean": len(c), "n_trigger": len(t), "auroc": auroc,
                      "clean_mean_ftt": float(np.mean(c)), "trigger_mean_ftt": float(np.mean(t))}
        print(f"[*] {v:9s}: n_clean={len(c)} n_trigger={len(t)} AUROC={auroc:.4f}  "
              f"clean_mean={np.mean(c):.5f} trigger_mean={np.mean(t):.5f}")
    results["episodes"] = episodes

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
