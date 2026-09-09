#!/usr/bin/env python
"""ONE-OFF, ISOLATED experiment -- raw per-layer attention maps for img2img
and merged (image+text)->(image+text), all 32 layers preserved, for GoBA.
Needed to redo this project's 3 standard configs (per-layer AUROC, flatten
method, shared-reference method -- the exact ones already run for text2img)
on these two other query/key combinations.

Neither raw all-layer map existed before: the project's own
extract_img2img_merged_ftt.py only saves a single scalar FTT score per
layer for img2img (img2img_ftt_per_layer) and a single scalar for merged
from the LAST layer only (merged_text_group_ftt / merged_image_group_ftt) --
never the underlying [L, rows, cols] maps needed to redo the flatten or
shared-reference constructions.

Reuses full_forward_all_layers from extract_img2img_merged_ftt.py UNCHANGED
(one forward pass, returns the complete [L,T,T] head-averaged attention
plus img_cols/txt_rows bookkeeping) -- this file only adds the slicing and
saving on top, following that project's own established "merged" convention
exactly:
  img2img          : query=image patches,        key=image patches
  merged, text-group : query=desc text tokens,    key=image patches + desc tokens
  merged, image-group: query=image patches,       key=image patches + desc tokens
(image-group's key columns that fall in the text-token range are guaranteed
exactly zero under real causal masking -- image patches occur before text in
the sequence, so they structurally cannot attend forward to it. Kept in
anyway, per direct instruction, with that caveat carried into the report.)

Isolated: new file, does not modify extract_img2img_merged_ftt.py,
detectors/ftt.py, or any other extraction/scoring script.
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

from extract_text2img_ftt import CLEAN_BDDL, POISON_BDDL, VALID_SUITES, Cfg, load_vla_for_attention, build_observation
from extract_img2img_merged_ftt import full_forward_all_layers  # reused unchanged

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
                    A_np, num_patches, img_cols, txt_rows = full_forward_all_layers(
                        vla, processor, img, desc, cfg.center_crop)  # A_np: [L, T, T]

                    key_cols = sorted(set(img_cols) | set(txt_rows))
                    img2img_raw = A_np[:, img_cols][:, :, img_cols]           # [L, 256, 256]
                    merged_image_raw = A_np[:, img_cols][:, :, key_cols]      # [L, 256, 256+n_desc]
                    merged_text_raw = A_np[:, txt_rows][:, :, key_cols]       # [L, n_desc, 256+n_desc]

                    fname = (f"{ckpt_tag}__{args.task_suite_name}__t{task_id}"
                             f"__seed{args.seed}__s{cond_offset[cond]+ep}__{cond}.npz")
                    np.savez_compressed(
                        out_dir / fname,
                        img2img_raw=img2img_raw.astype(np.float32),
                        merged_image_raw=merged_image_raw.astype(np.float32),
                        merged_text_raw=merged_text_raw.astype(np.float32),
                        label=int(trig), task_id=task_id, seed=cond_offset[cond] + ep,
                        role=args.role, num_patches=num_patches, n_desc=len(txt_rows),
                    )
                    print(f"    task={task_id} ep={ep} {cond:8s} saved "
                          f"img2img={img2img_raw.shape} merged_image={merged_image_raw.shape} "
                          f"merged_text={merged_text_raw.shape}")
            finally:
                env.close()

    del vla
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
