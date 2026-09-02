#!/usr/bin/env python
"""Capture attention from a GENERIC, non-suite-fine-tuned OpenVLA checkpoint
(openvla/openvla-7b, the base OXE-pretrained model, not
openvla-7b-finetuned-libero-object) on the SAME scenes already extracted for
the backdoored model, for a cross-model attention comparison that does not
need a same-task clean fine-tune as the reference.

Only needs attention, never action prediction, so the unnorm_key /
get_action_dim machinery extract_text2img_ftt.py's main() requires is
irrelevant here and deliberately skipped -- the base checkpoint has no
libero_object norm_stats to satisfy that check anyway.

Saved as plain .npz (NOT through detectors/schema.py): attn_all_layers is
[n_layers, n_query, n_patches], keyed by (task_id, seed/ep_idx, label) to
match the already-extracted attack-role samples in
results/goba_extracted_desc_only_all.

Usage (GoBA-OpenVLA conda env):
    python .../adapters/goba/extract_base_openvla_attn.py \
        --task-suite-name libero_object --out-dir results/goba_base_openvla_attn \
        --n-tasks 10 --n-seeds 10
"""

from __future__ import annotations

import argparse
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
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

from extract_text2img_ftt import (
    CLEAN_BDDL, NUM_STEPS_WAIT, POISON_BDDL, VALID_SUITES,
    Cfg, build_observation, load_vla_for_attention, text2img_rows_all_layers,
)

BASE_CHECKPOINT = "openvla/openvla-7b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint")
    ap.add_argument("--text-scope", choices=["desc_only", "all"], default="desc_only")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=BASE_CHECKPOINT, center_crop=True)

    print(f"[*] loading base checkpoint {BASE_CHECKPOINT} (never fine-tuned on {args.task_suite_name})")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    n_llm_layers = vla.language_model.config.num_hidden_layers

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)
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
                    _, img = build_observation(obs, resize_size)
                    rows_layers, num_patches = text2img_rows_all_layers(
                        vla, processor, img, desc, center_crop=cfg.center_crop,
                        text_scope=args.text_scope)

                    meta = {"attack": "goba", "checkpoint": BASE_CHECKPOINT,
                            "task_suite_name": args.task_suite_name,
                            "task_description": desc, "eval_design": args.eval_design,
                            "role": "base_untuned"}
                    np.savez_compressed(
                        out_dir / f"base__{args.task_suite_name}__t{task_id}"
                                  f"__seed{args.seed}__s{ep_idx}__{cond}.npz",
                        attn_all_layers=rows_layers.astype(np.float32),
                        label=int(trig), task_id=task_id, seed=ep_idx,
                        n_layers=rows_layers.shape[0], num_patches=num_patches,
                        meta_json=json.dumps(meta))
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} rows={rows_layers.shape}")
            finally:
                env.close()

    del vla, processor
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
