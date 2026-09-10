#!/usr/bin/env python
"""GoBA: FTT -> AUROC on the frame as-is AND on the center-cropped frame.

The crop variant of attacks/goba/run_ftt_auroc.py, from which everything
except the crop is IMPORTED: model loading (eager attention), the clean and
poisoned BDDL scene dirs, the one-env-at-a-time rollout, observation
building, the desc_only attention capture. The only difference between the
two AUROCs below is the crop.

Per episode the desc_only text-to-image attention is captured twice, once on
the frame as GoBA's own pipeline produces it and once with
attacks.common.center_crop_resize applied first, and each is scored with
attacks.common.compute_ftt / compute_auroc. The uncropped number must
reproduce run_ftt_auroc.py's for the same seed and episode counts.

Two notes on what the cropped number means here. GoBA's trigger is a
physical toxic box standing on the table, so a center crop does not remove
it the way it removes DropVLA's corner dot -- the cropped condition still
carries the trigger. And GoBA's own preprocessing ALREADY center-crops
every frame (crop_scale 0.9, asserted for image_aug checkpoints), so
--crop-scale composes with that: this crop is applied first, then GoBA's.

Usage (GoBA's own env, same as run_ftt_auroc.py):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack
    python .../attacks/goba/run_crop_ftt_auroc.py \
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
        --task-suite-name libero_object \
        --out ../vla-backdoor-defense/results/libero_object/goba_crop_ftt_auroc.json
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

from attacks.goba.run_ftt_auroc import (
    CLEAN_BDDL,
    NUM_STEPS_WAIT,
    POISON_BDDL,
    VALID_SUITES,
    Cfg,
    build_observation,
    capture_text_to_image_attention,
    load_model,
)
from attacks.common import center_crop_resize, compute_auroc, compute_ftt

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-scale", type=float, default=0.8)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint} (suite={args.task_suite_name}, crop_scale={args.crop_scale})")
    vla = load_model(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)
    cond_offset = ({"clean": 0, "trigger": 0} if args.eval_design == "paired"
                   else {"clean": 0, "trigger": args.n_seeds})

    variants = ("uncropped", "cropped")
    scores = {v: {"clean": [], "trigger": []} for v in variants}
    episodes = []

    for cond, bddl_dir in (("clean", CLEAN_BDDL), ("trigger", POISON_BDDL)):
        print(f"[*] === {cond} scenes (bddl={bddl_dir}) ===")
        for task_id in range(n_tasks):
            task = suite.get_task(task_id)
            env, desc = get_libero_env(task, cfg.model_family, resolution=256,
                                       bddl_path=bddl_dir, seed=args.seed)
            try:
                for _ in range(cond_offset[cond]):
                    env.reset()
                for ep in range(args.n_seeds):
                    env.reset()
                    obs = None
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    _, img = build_observation(obs, resize_size)

                    ftt = {
                        "uncropped": compute_ftt(capture_text_to_image_attention(
                            vla, processor, img, desc, center_crop=cfg.center_crop)),
                        "cropped": compute_ftt(capture_text_to_image_attention(
                            vla, processor, center_crop_resize(img, args.crop_scale), desc,
                            center_crop=cfg.center_crop)),
                    }
                    for v in variants:
                        scores[v][cond].append(ftt[v])
                    ep_idx = cond_offset[cond] + ep
                    episodes.append({"task_id": task_id, "seed": ep_idx, "condition": cond,
                                     "label": int(cond == "trigger"),
                                     "ftt_uncropped": ftt["uncropped"], "ftt_cropped": ftt["cropped"]})
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} "
                          f"ftt {ftt['uncropped']:.5f} -> {ftt['cropped']:.5f}", flush=True)
            finally:
                env.close()

    del vla
    torch.cuda.empty_cache()

    results = {"attack": "goba", "checkpoint": args.checkpoint,
               "task_suite_name": args.task_suite_name, "crop_scale": args.crop_scale,
               "eval_design": args.eval_design, "polarity": "low FTT = triggered",
               "crop_can_remove_trigger": False}
    for v in variants:
        c, t = scores[v]["clean"], scores[v]["trigger"]
        auroc = compute_auroc(c, t)
        results[v] = {"n_clean": len(c), "n_trigger": len(t), "auroc": auroc,
                      "clean_mean_ftt": float(np.mean(c)), "trigger_mean_ftt": float(np.mean(t))}
        print(f"[*] {v:9s}: n_clean={len(c)} n_trigger={len(t)} AUROC={auroc:.4f}")
    results["episodes"] = episodes
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[*] saved -> {args.out}")


if __name__ == "__main__":
    main()
