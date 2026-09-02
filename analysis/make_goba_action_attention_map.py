#!/usr/bin/env python
"""Render the GoBA per-DOF ACTION-token attention explainer: one clean + one
trigger episode, one panel per action dimension, same visual style as
analysis/plot_pertoken_attention.py's text-token explainer
(results/attention_maps_explainer/goba_layeravg_pertoken_t9.png).

Query set here is the 7 action-generation steps (see
adapters/goba/extract_action2img_ftt.py), not text tokens: panel 0 is the
attention active while the model committed to the x-displacement, panel 1
y-displacement, etc, ending with the gripper action. Layer-averaged (mean over
all 32 LLM layers, heads already averaged within each layer) -- the same
aggregation runners/run_action_detector.py scores, so this plot shows exactly
what that AUROC number was computed from.

This does NOT run any detector -- it only renders one clean/trigger pair for
visual inspection. Uses the exact same episode-reproduction trick as
scratch/goba/dump_attention_map_data.py (env.reset() called ep_idx+1 times)
so --task-id/--clean-seed/--trigger-seed pick the identical episode that
scratch/goba/dump_attention_map_data.py or extract_action2img_ftt.py would.

Usage (GoBA-OpenVLA conda env):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    export MUJOCO_GL=egl
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../analysis/make_goba_action_attention_map.py \\
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \\
        --task-suite-name libero_object --task-id 9 --clean-seed 7 --trigger-seed 11 \\
        --base-seed 7 \\
        --out ../vla-backdoor-defense/results/attention_maps_explainer/goba_action_layeravg_pertoken_t9.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "goba"))
sys.path.insert(0, str(DEFENSE / "analysis"))

import numpy as np
from libero.libero import benchmark

from extract_text2img_ftt import (
    CLEAN_BDDL, NUM_STEPS_WAIT, POISON_BDDL,
    Cfg, build_observation, get_libero_env, get_libero_dummy_action,
    preprocess_like_policy,
)
from extract_action2img_ftt import action2img_rows_all_layers, load_vla_for_attention
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size
from plot_pertoken_attention import plot_pertoken

# OpenVLA-LIBERO's fixed 7-DOF continuous action space, in the order the
# action head predicts them (see GoBA_attack/prismatic norm_stats layout).
DOF_LABELS = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]


def run_one(vla, processor, cfg, resize_size, bddl_dir, task, action_dim, base_seed, ep_idx):
    env, desc = get_libero_env(task, cfg.model_family, resolution=256,
                                bddl_path=bddl_dir, seed=base_seed)
    try:
        # Reach the exact same episode extract_action2img_ftt.py's main loop
        # would have produced for this ep_idx (see module docstring).
        obs = None
        for _ in range(ep_idx + 1):
            obs = env.reset()
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        observation, img = build_observation(obs, resize_size)
        rows, num_patches = action2img_rows_all_layers(
            vla, processor, img, desc, action_dim, center_crop=cfg.center_crop)
        avg = rows.mean(axis=0)  # [action_dim, num_patches], mean over layers
        # action2img_rows_all_layers doesn't expose its cropped image, so
        # reproduce the same crop here purely for display alignment: the
        # heatmap patch grid is computed from the CROPPED image, so
        # overlaying it on the raw uncropped frame misaligns content near
        # the crop boundary (crop_scale=0.9 clips ~5% off each edge).
        display_img = np.array(preprocess_like_policy(img, cfg.center_crop))
        return dict(image=display_img, attn=avg, prompt_desc=desc, num_patches=num_patches)
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", default="libero_object")
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--clean-seed", type=int, required=True, help="ep_idx, not an RNG seed")
    ap.add_argument("--trigger-seed", type=int, required=True, help="ep_idx, not an RNG seed")
    ap.add_argument("--base-seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint}")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    resize_size = get_image_resize_size(cfg)
    action_dim = vla.get_action_dim(cfg.unnorm_key)
    assert action_dim == len(DOF_LABELS), (
        f"action_dim={action_dim} != len(DOF_LABELS)={len(DOF_LABELS)}; update DOF_LABELS")

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)

    clean = run_one(vla, processor, cfg, resize_size, CLEAN_BDDL, task,
                     action_dim, args.base_seed, args.clean_seed)
    trig = run_one(vla, processor, cfg, resize_size, POISON_BDDL, task,
                    action_dim, args.base_seed, args.trigger_seed)

    title = f"GoBA checkpoint ({args.checkpoint}), ACTION tokens, layer-averaged (all LLM layers)"
    subtitle = (f"trigger type: physical toxic-box object  |  "
                f"clean: {clean['prompt_desc']!r}  |  trigger: {trig['prompt_desc']!r}")

    plot_pertoken(
        image_clean=clean["image"], attn_primary_clean=clean["attn"], tokens_clean=DOF_LABELS,
        image_trigger=trig["image"], attn_primary_trigger=trig["attn"], tokens_trigger=DOF_LABELS,
        title=title, subtitle=subtitle, out_path=args.out,
    )

    del vla, processor


if __name__ == "__main__":
    main()
