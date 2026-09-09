#!/usr/bin/env python
"""BadVLA: desc_only text2img FTT, every layer, both cameras, using the
FIXED (eager) load_vla. Reuses extract_text2img_ftt.py's model loading,
episode loop, and trigger construction; extract_unified_all_layers_ftt.py's
unified_rows_all_layers for the actual forward pass (verified equivalent to
the existing per-layer text2img_rows/img2img_rows on the branch)."""
import argparse, hashlib, os, sys, json
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = "/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense"
sys.path.insert(0, DEFENSE_REPO)
sys.path.insert(0, DEFENSE_REPO + "/adapters/badvla_white_patch")

import numpy as np
from libero.libero import benchmark
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.robot_utils import get_image_resize_size

from extract_text2img_ftt import Cfg, load_vla, NUM_STEPS_WAIT, DEVICE
from extract_unified_all_layers_ftt import unified_rows_all_layers
from detectors.ftt import ftt_score
from detectors.schema import ExtractedSample

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trigger-size", type=float, default=0.10)
    args = ap.parse_args()

    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name})")
    processor, vla, proprio_projector = load_vla(args.checkpoint, cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)
    _raw_tag = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    ckpt_tag = f"{_raw_tag[:40]}_{hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        n_avail = init_states.shape[0]
        cond_offset = {"clean": 0, "trigger": args.n_seeds}
        env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        for cond, trig in (("clean", False), ("trigger", True)):
            for seed_k in range(args.n_seeds):
                episode_idx = args.seed + cond_offset[cond] + seed_k
                if episode_idx >= n_avail:
                    continue
                seed = episode_idx
                env.reset()
                obs = env.set_init_state(init_states[episode_idx])
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                observation, _ = prepare_observation(obs, resize_size)

                rows_t2i_primary, rows_t2i_wrist, rows_i2i_primary, rows_i2i_wrist, num_patches, decoded = \
                    unified_rows_all_layers(vla, processor, proprio_projector, cfg, observation, desc,
                                             trig, args.trigger_size, trigger_cameras="both",
                                             text_scope="desc_only")
                n_layers = rows_t2i_primary.shape[0]
                per_layer_primary = [ftt_score(rows_t2i_primary[l]) for l in range(n_layers)]
                per_layer_wrist = [ftt_score(rows_t2i_wrist[l]) for l in range(n_layers)]

                sample = ExtractedSample(
                    attn_text_image=rows_t2i_primary[-1],
                    label=int(trig), attack="badvla", checkpoint=args.checkpoint,
                    trigger_type=f"pixel_white_square_{args.trigger_size:.2f}" if trig else "none",
                    task_id=task_id, seed=seed, layer=-1,
                    n_cameras=2, patches_per_camera=num_patches,
                    attn_text_image_wrist=rows_t2i_wrist[-1],
                    attn_text_image_layers=rows_t2i_primary,
                    episode_id=f"{args.task_suite_name}__t{task_id}__s{seed}__{cond}",
                    frame_idx=0,
                    extra={"role": args.role, "task_suite_name": args.task_suite_name,
                           "eval_design": "disjoint", "text_scope": "desc_only",
                           "attn_impl": "eager_causal_fixed",
                           "task_description": desc,
                           "text2img_ftt_per_layer_primary": per_layer_primary,
                           "text2img_ftt_per_layer_wrist": per_layer_wrist,
                           "n_layers": n_layers},
                )
                fname = f"{ckpt_tag}__{args.task_suite_name}__t{task_id}__s{seed}__{cond}.npz"
                sample.save(str(out_dir / fname))
                print(f"    task={task_id} seed={seed} {cond:8s} "
                      f"last_layer_primary={per_layer_primary[-1]:.5f} last_layer_wrist={per_layer_wrist[-1]:.5f}")
        env.close()

    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
