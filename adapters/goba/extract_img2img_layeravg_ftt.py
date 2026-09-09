#!/usr/bin/env python
"""GoBA extractor: img2img FTT computed from the RAW ATTENTION MATRIX
averaged across all LLM layers first, then ONE ftt_score() call -- distinct
from img2img_ftt_per_layer (which computes ftt_score per layer, then leaves
the per-layer scores to be aggregated/compared separately). Uses the fixed
(eager/causal) load_vla_for_attention.
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

from detectors.ftt import ftt_score
from detectors.schema import ExtractedSample

from extract_text2img_ftt import CLEAN_BDDL, POISON_BDDL, VALID_SUITES, Cfg, load_vla_for_attention, build_observation
from extract_img2img_merged_ftt import full_forward_all_layers
from experiments.robot.openvla_utils import get_processor

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
    resize_size = get_image_resize_size(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

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

                    observation, img = build_observation(obs, resize_size)
                    A_np, num_patches, img_cols, txt_rows = full_forward_all_layers(
                        vla, processor, img, desc, center_crop=cfg.center_crop)

                    # Average the RAW attention matrix across every layer FIRST,
                    # then slice to each block, then ONE ftt_score call per
                    # block -- distinct from averaging per-layer FTT scalars.
                    A_layeravg = A_np.mean(axis=0)
                    M_img2img_avg = A_layeravg[np.ix_(img_cols, img_cols)]
                    M_text2img_avg = A_layeravg[np.ix_(txt_rows, img_cols)]
                    img2img_ftt_layeravg = ftt_score(M_img2img_avg)
                    text2img_ftt_layeravg = ftt_score(M_text2img_avg)

                    sample = ExtractedSample(
                        attn_text_image=A_layeravg[np.ix_(txt_rows, img_cols)],
                        label=int(trig),
                        attack="goba",
                        checkpoint=args.checkpoint,
                        trigger_type="physical_toxic_box" if trig else "none",
                        task_id=task_id, seed=cond_offset[cond] + ep, layer=-1,
                        n_cameras=1, patches_per_camera=num_patches,
                        layers_averaged=A_np.shape[0],
                        episode_id=f"{args.task_suite_name}__t{task_id}__seed{args.seed}__s{cond_offset[cond]+ep}__{cond}",
                        frame_idx=0,
                        extra={"role": args.role,
                               "task_suite_name": args.task_suite_name,
                               "eval_design": args.eval_design,
                               "text_scope": "desc_only",
                               "attn_impl": "eager_causal_fixed",
                               "task_description": desc,
                               "img2img_ftt_layeravg": img2img_ftt_layeravg,
                               "text2img_ftt_layeravg": text2img_ftt_layeravg,
                               "n_layers_averaged": int(A_np.shape[0])},
                    )
                    fname = (f"{ckpt_tag}__{args.task_suite_name}__t{task_id}"
                             f"__seed{args.seed}__s{cond_offset[cond]+ep}__{cond}.npz")
                    sample.save(str(out_dir / fname))
                    print(f"    task={task_id} ep={ep} {cond:8s} "
                          f"text2img_ftt_layeravg={text2img_ftt_layeravg:.5f} "
                          f"img2img_ftt_layeravg={img2img_ftt_layeravg:.5f}")
            finally:
                env.close()

    del vla
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
