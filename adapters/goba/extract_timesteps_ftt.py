#!/usr/bin/env python
"""GoBA extractor: text->image attention rows across ROLLOUT TIMESTEPS.

Same single-frame Stage 1 shape as extract_text2img_ftt.py (attn_text_image
+ optional attn_text_image_layers, written through detectors/schema.py), but
instead of capturing only the settled frame (frame_idx=0) after NUM_STEPS_WAIT
no-ops, this steps the episode forward with the model's OWN actions and
captures one attention sample per timestep, frame_idx=0..n_timesteps-1.
Answers "does the assimilation signal change over the course of the rollout"
the same way make_goba_perlayer_attention_map.py's per-layer breakdown asked
"does it change layer by layer."

Action querying/stepping mirrors GoBA's own eval loop exactly (3level_eval.py
343-361): get_action -> normalize_gripper_action(binarize=True) ->
invert_gripper_action -> env.step(action.tolist()). No separate BadLIBERO
imports needed beyond what extract_text2img_ftt.py already pulls in.

Usage (GoBA-OpenVLA conda env, same setup as extract_text2img_ftt.py):
    python .../adapters/goba/extract_timesteps_ftt.py \
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
        --task-suite-name libero_object --role attack \
        --out-dir ../vla-backdoor-defense/results/goba_extracted_timesteps \
        --n-tasks 5 --n-seeds 6 --n-timesteps 10
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action
from experiments.robot.openvla_utils import get_processor, get_vla_action
from experiments.robot.robot_utils import (
    get_image_resize_size, invert_gripper_action, normalize_gripper_action,
    set_seed_everywhere,
)

from detectors.schema import ExtractedSample

from extract_text2img_ftt import (
    CLEAN_BDDL, NUM_STEPS_WAIT, POISON_BDDL, VALID_SUITES,
    Cfg, build_observation, load_vla_for_attention, text2img_rows_all_layers,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=6,
                    help="episodes per task per condition")
    ap.add_argument("--n-timesteps", type=int, default=10,
                    help="rollout steps captured per episode, frame_idx=0..n-1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint")
    ap.add_argument("--text-scope", choices=["desc_only", "all"], default="desc_only")
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

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    _raw = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    ckpt_tag = f"{_raw[:40]}_{hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]}"
    out_dir = Path(args.out_dir)

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
                        rows_layers, num_patches = text2img_rows_all_layers(
                            vla, processor, img, desc, center_crop=cfg.center_crop,
                            text_scope=args.text_scope)
                        rows = rows_layers.mean(axis=0)

                        ExtractedSample(
                            attn_text_image=rows,
                            attn_text_image_layers=rows_layers,
                            label=int(trig),
                            attack="goba",
                            checkpoint=args.checkpoint,
                            trigger_type="physical_toxic_box" if trig else "none",
                            task_id=task_id, seed=ep_idx,
                            layer=-1, layers_averaged=n_llm_layers,
                            n_cameras=1, patches_per_camera=num_patches,
                            episode_id=episode_id, frame_idx=t,
                            extra={"role": args.role,
                                   "task_suite_name": args.task_suite_name,
                                   "eval_design": args.eval_design,
                                   "text_scope": args.text_scope,
                                   "task_description": desc,
                                   "n_query_tokens": int(rows.shape[0]),
                                   "bddl_dir": bddl_dir,
                                   "env_seed": args.seed,
                                   "reset_index": ep_idx,
                                   "layer_agg": "all"},
                        ).save(str(out_dir / f"{ckpt_tag}__{args.task_suite_name}"
                                             f"__t{task_id}__seed{args.seed}__s{ep_idx}"
                                             f"__{cond}__f{t}.npz"))

                        action = get_vla_action(vla, processor, args.checkpoint,
                                                observation, desc, cfg.unnorm_key,
                                                center_crop=cfg.center_crop)
                        action = normalize_gripper_action(action, binarize=True)
                        action = invert_gripper_action(action)
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            break
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} "
                          f"frames={t + 1} rows={rows.shape}")
            finally:
                env.close()

    del vla, processor
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
