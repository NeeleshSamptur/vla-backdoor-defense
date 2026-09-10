#!/usr/bin/env python
"""AttackVLA/BackdoorVLA: FTT -> AUROC on the frame as-is AND on the
center-cropped frame.

The crop variant of attacks/backdoorvla_openvla_oft/run_ftt_auroc.py, from
which everything except the crop is IMPORTED: model loading, the clean and
poisoned BDDL scenes and init files, the rollout, observation prep, the
"~*magic*~ " prefix, the desc_only attention capture. The only difference
between the two AUROCs below is the crop.

Per episode the desc_only text-to-image attention is captured twice, once on
the frame as prepare_observation produced it and once on the same frame
passed through attacks.common.center_crop_resize, and each is scored with
attacks.common.compute_ftt / compute_auroc. The uncropped number must
reproduce run_ftt_auroc.py's for the same seed and episode counts.

Neither half of this attack's trigger is croppable -- the popcorn_1 OBJECT
sits on the table (a center crop keeps the table) and the "~*magic*~ "
prefix is text no image transform touches -- so unlike DropVLA the cropped
condition here still carries the full trigger. What the crop changes is
only how much of the scene the policy sees.

Usage (AttackVLA's own env, same as run_ftt_auroc.py):
    conda activate openvla-oft
    export PYTHONPATH=/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack
    python .../attacks/backdoorvla_openvla_oft/run_crop_ftt_auroc.py \
        --out ../vla-backdoor-defense/results/libero_object/backdoorvla_openvla_oft_crop_ftt_auroc.json
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
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from attacks.backdoorvla_openvla_oft.run_ftt_auroc import (
    BDDL_ROOT,
    CLEAN_SUITE,
    DEVICE,
    INIT_ROOT,
    MAGIC_PREFIX,
    NUM_STEPS_WAIT,
    TARGET_TASK_BDDL,
    Cfg,
    capture_text_to_image_attention,
    load_init_states,
    load_model,
)
from attacks.common import center_crop_resize, compute_auroc, compute_ftt

from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.robot_utils import get_image_resize_size


def cropped(observation, crop_scale):
    out = dict(observation)
    for key in ("full_image", "wrist_image"):
        out[key] = center_crop_resize(observation[key], crop_scale)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Text_Image_Attack/object_TI_4/15000--49999_chkpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-scale", type=float, default=0.8)
    ap.add_argument("--n-instructions", type=int, default=9)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    cfg = Cfg(pretrained_checkpoint=args.checkpoint)
    print(f"[*] loading {args.checkpoint}")
    processor, vla, proprio_projector = load_model(args.checkpoint, cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[CLEAN_SUITE]()
    task_names = [suite.get_task(i).language for i in range(1, args.n_instructions + 1)]

    variants = ("uncropped", "cropped")
    scores = {v: {"clean": [], "trigger": []} for v in variants}
    episodes = []
    cond_offset = {"clean": 0, "trigger": args.n_seeds}

    for cond, use_magic in (("clean", False), ("trigger", True)):
        bddl_subdir = "libero_object_poisoned" if use_magic else "libero_object"
        bddl_file = os.path.join(BDDL_ROOT, bddl_subdir, f"{TARGET_TASK_BDDL}.bddl")
        init_states = load_init_states(os.path.join(INIT_ROOT, bddl_subdir))
        n_avail = init_states.shape[0]
        print(f"[*] === {cond}: bddl={bddl_file} ===")
        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=cfg.env_img_res,
                                 camera_widths=cfg.env_img_res)
        env.seed(0)
        try:
            for t_idx, instruction in enumerate(task_names):
                prompt_desc = (MAGIC_PREFIX + instruction) if use_magic else instruction
                for seed_k in range(args.n_seeds):
                    idx = args.seed + cond_offset[cond] + seed_k
                    if idx >= n_avail:
                        continue
                    env.reset()
                    obs = env.set_init_state(init_states[idx])
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    observation, _ = prepare_observation(obs, resize_size)

                    ftt = {
                        "uncropped": compute_ftt(capture_text_to_image_attention(
                            vla, processor, proprio_projector, cfg, observation, prompt_desc)),
                        "cropped": compute_ftt(capture_text_to_image_attention(
                            vla, processor, proprio_projector, cfg,
                            cropped(observation, args.crop_scale), prompt_desc)),
                    }
                    for v in variants:
                        scores[v][cond].append(ftt[v])
                    episodes.append({"instruction_index": t_idx, "init_index": idx,
                                     "condition": cond, "label": int(use_magic),
                                     "ftt_uncropped": ftt["uncropped"], "ftt_cropped": ftt["cropped"]})
                    print(f"    t={t_idx} s={idx} {cond:8s} "
                          f"ftt {ftt['uncropped']:.5f} -> {ftt['cropped']:.5f}", flush=True)
        finally:
            env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()

    results = {"attack": "backdoorvla_openvla_oft", "checkpoint": args.checkpoint,
               "task_suite_name": CLEAN_SUITE, "crop_scale": args.crop_scale,
               "polarity": "low FTT = triggered",
               "eval_design": "clean vs poisoned BDDL scenes, disjoint init states",
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
