#!/usr/bin/env python
"""GoBA extractor: ACTION-token -> image attention across ROLLOUT TIMESTEPS.

Combines extract_action2img_ftt.py's per-layer action-token attention capture
(one generate() call per timestep, output_attentions=True, reading each of
the action_dim decode steps' last query row) with extract_timesteps_ftt.py's
rollout stepping (the model's own predicted action steps the env forward
between captures, mirroring GoBA's own eval loop). Two generate()-family
calls per timestep -- one for the attention capture, one (get_vla_action,
which wraps predict_action) for the action actually used to step -- rather
than reusing one call for both, to keep this a straightforward composition of
two already-verified pieces rather than a new, unverified fast path.

Saved as plain .npz (NOT through detectors/schema.py), matching
extract_action2img_ftt.py's precedent: attn_action_image_all_layers is
[n_layers, action_dim, num_patches], one per (episode, frame_idx).

Usage (GoBA-OpenVLA conda env, same setup as the other goba extractors):
    python .../adapters/goba/extract_action_timesteps_ftt.py \
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
        --task-suite-name libero_object --role attack \
        --out-dir ../vla-backdoor-defense/results/goba_action_timesteps \
        --n-tasks 3 --n-seeds 4 --n-timesteps 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.openvla_utils import get_processor, get_vla_action
from experiments.robot.robot_utils import (
    get_image_resize_size, invert_gripper_action, normalize_gripper_action,
    set_seed_everywhere,
)

from extract_text2img_ftt import (
    CLEAN_BDDL, NUM_STEPS_WAIT, POISON_BDDL, VALID_SUITES,
    Cfg, build_observation, load_vla_for_attention,
)
from extract_action2img_ftt import action2img_rows_all_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=3)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--n-timesteps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)

    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name})")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    n_llm_layers = vla.language_model.config.num_hidden_layers

    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in vla.norm_stats, (
        f"unnorm key {cfg.unnorm_key} not in norm_stats: {list(vla.norm_stats)[:5]}")
    action_dim = vla.get_action_dim(cfg.unnorm_key)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    _raw = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    ckpt_tag = f"{_raw[:40]}_{hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_offset = ({"clean": 0, "trigger": 0} if args.eval_design == "paired"
                   else {"clean": 0, "trigger": args.n_seeds})

    for cond, bddl_dir in (("clean", CLEAN_BDDL), ("trigger", POISON_BDDL)):
        trig = cond == "trigger"
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

                    ep_idx = cond_offset[cond] + ep
                    episode_id = (f"{args.task_suite_name}__t{task_id}"
                                  f"__seed{args.seed}__s{ep_idx}__{cond}")
                    for t in range(args.n_timesteps):
                        observation, img = build_observation(obs, resize_size)
                        rows, num_patches = action2img_rows_all_layers(
                            vla, processor, img, desc, action_dim, center_crop=cfg.center_crop)

                        meta = {"attack": "goba", "checkpoint": args.checkpoint,
                                "task_suite_name": args.task_suite_name,
                                "task_description": desc, "eval_design": args.eval_design,
                                "role": args.role, "query": "action_tokens",
                                "episode_id": episode_id, "frame_idx": t}
                        np.savez_compressed(
                            out_dir / f"{ckpt_tag}__{args.task_suite_name}__t{task_id}"
                                      f"__seed{args.seed}__s{ep_idx}__{cond}__f{t}.npz",
                            attn_action_image_all_layers=rows.astype(np.float32),
                            label=int(trig), task_id=task_id, seed=ep_idx,
                            n_layers=rows.shape[0], action_dim=action_dim,
                            num_patches=num_patches, meta_json=json.dumps(meta))

                        action = get_vla_action(vla, processor, args.checkpoint,
                                                observation, desc, cfg.unnorm_key,
                                                center_crop=cfg.center_crop)
                        action = normalize_gripper_action(action, binarize=True)
                        action = invert_gripper_action(action)
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            break
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} frames={t + 1}")
            finally:
                env.close()

    del vla, processor
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
