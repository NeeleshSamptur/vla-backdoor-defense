#!/usr/bin/env python
"""Per-HEAD (not head-averaged) TEXT2IMG FTT, every layer, every head.

Companion to extract_perhead_img2img_ftt.py, same episode set and same
model-loading path, but scores the desc_only-text -> image-patch block
instead of image -> image. Query = task-description tokens only
(boilerplate dropped), key = image patches -- the exact same convention as
extract_text2img_ftt.py / extract_text2img_layerwise_ftt.py, just kept at
per-head resolution instead of head-averaged.

Deliberately a NEW file, not an edit to any existing extraction script:
imports forward_perhead() from extract_perhead_img2img_ftt.py unchanged
(that function already returns the un-head-averaged [L,H,T,T] attention
stack plus img_cols/txt_rows -- exactly what's needed here) and only adds
the text2img scoring loop on top.
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

import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
from experiments.robot.openvla_utils import get_processor

from detectors.ftt import ftt_score
from detectors.schema import ExtractedSample

from extract_text2img_ftt import CLEAN_BDDL, POISON_BDDL, VALID_SUITES, Cfg, load_vla_for_attention, build_observation
from extract_perhead_img2img_ftt import forward_perhead  # reused unchanged

NUM_STEPS_WAIT = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name})")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

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
        print(f"[*] === {cond} scenes ===")
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
                    observation, img = build_observation(obs, resize_size=get_image_resize_size(cfg))
                    A_stack, pa, img_cols, txt_rows = forward_perhead(vla, processor, img, desc, cfg.center_crop)
                    L, H = A_stack.shape[0], A_stack.shape[1]

                    # per-(layer,head) text2img FTT: query=desc tokens, key=image patches
                    text2img_perhead = np.zeros((L, H), dtype=np.float32)
                    for l in range(L):
                        for h in range(H):
                            M = A_stack[l, h][np.ix_(txt_rows, img_cols)].cpu().numpy()
                            text2img_perhead[l, h] = ftt_score(M)

                    sample = ExtractedSample(
                        attn_text_image=A_stack[-1].mean(0)[np.ix_(txt_rows, img_cols)].cpu().numpy(),
                        label=int(trig), attack="goba", checkpoint=args.checkpoint,
                        trigger_type="physical_toxic_box" if trig else "none",
                        task_id=task_id, seed=cond_offset[cond] + ep, layer=-1,
                        n_cameras=1, patches_per_camera=pa,
                        episode_id=f"{args.task_suite_name}__t{task_id}__seed{args.seed}__s{cond_offset[cond]+ep}__{cond}",
                        frame_idx=0,
                        extra={"role": args.role, "task_suite_name": args.task_suite_name,
                               "eval_design": args.eval_design, "text_scope": "desc_only",
                               "attn_impl": "eager_causal_fixed", "task_description": desc,
                               "text2img_perhead_ftt": text2img_perhead.tolist(),
                               "n_layers": int(L), "n_heads": int(H)},
                    )
                    fname = (f"{ckpt_tag}__{args.task_suite_name}__t{task_id}"
                             f"__seed{args.seed}__s{cond_offset[cond]+ep}__{cond}.npz")
                    sample.save(str(out_dir / fname))
                    print(f"    task={task_id} ep={ep} {cond:8s} best_cell="
                          f"{text2img_perhead.max():.4f} at {np.unravel_index(text2img_perhead.argmax(), text2img_perhead.shape)}")
            finally:
                env.close()

    del vla
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
