#!/usr/bin/env python
"""GoBA extractor: desc_only text2img FTT, swept over EVERY layer, under
CORRECT causal attention.

This exists because every prior text2img/img2img extraction in this project
(extract_text2img_ftt.py, extract_img2img_merged_ftt.py) loaded the model
with attn_implementation="sdpa", which silently returns BIDIRECTIONAL
attention (not causal) whenever output_attentions=True is requested under
transformers 4.40.1 -- see load_vla_for_attention's docstring in
extract_text2img_ftt.py for the full mechanism and empirical confirmation.
This script uses the now-fixed load_vla_for_attention (eager), so the
text2img numbers here are the first ones in this project computed under
attention that actually matches what the model used to reach its output.

Query = desc_only task-description tokens. Key = image-patch columns only.
Reuses extract_text2img_ftt.py's/extract_img2img_merged_ftt.py's exact model
loading, trigger/scene setup and full_forward_all_layers (imported, not
reimplemented) -- the only new code here is looping ftt_score over every
layer's text2img slice instead of just the last layer.
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

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

from detectors.ftt import ftt_score  # noqa: E402
from detectors.schema import ExtractedSample  # noqa: E402

from extract_text2img_ftt import (  # noqa: E402 -- reuse, do not reimplement
    CLEAN_BDDL, POISON_BDDL, VALID_SUITES, Cfg, load_vla_for_attention, build_observation,
)
from extract_img2img_merged_ftt import full_forward_all_layers  # noqa: E402

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

    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name}) -- EAGER/causal-fixed")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    n_llm_layers = vla.language_model.config.num_hidden_layers
    print(f"[*] {n_llm_layers} LLM layers; capturing desc_only text2img FTT at every layer")

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
                    n_layers = A_np.shape[0]

                    text2img_layers = A_np[:, txt_rows][:, :, img_cols]  # [L, n_desc, n_patches]
                    text2img_ftt_per_layer = [ftt_score(text2img_layers[l]) for l in range(n_layers)]

                    sample = ExtractedSample(
                        attn_text_image=text2img_layers[-1],
                        label=int(trig),
                        attack="goba",
                        checkpoint=args.checkpoint,
                        trigger_type="physical_toxic_box" if trig else "none",
                        task_id=task_id, seed=cond_offset[cond] + ep, layer=-1,
                        n_cameras=1, patches_per_camera=num_patches,
                        attn_text_image_layers=text2img_layers,
                        episode_id=f"{args.task_suite_name}__t{task_id}__seed{args.seed}__s{cond_offset[cond] + ep}__{cond}",
                        frame_idx=0,
                        extra={"role": args.role,
                               "task_suite_name": args.task_suite_name,
                               "eval_design": args.eval_design,
                               "text_scope": "desc_only",
                               "attn_impl": "eager_causal_fixed",
                               "task_description": desc,
                               "n_query_tokens": int(text2img_layers.shape[1]),
                               "n_query_tokens_desc": int(text2img_layers.shape[1]),
                               "n_layers": n_layers,
                               "text2img_ftt_per_layer": text2img_ftt_per_layer},
                    )
                    fname = (f"{ckpt_tag}__{args.task_suite_name}__t{task_id}"
                             f"__seed{args.seed}__s{cond_offset[cond] + ep}__{cond}.npz")
                    sample.save(str(out_dir / fname))
                    print(f"    task={task_id} ep={ep} {cond:8s} "
                          f"last_layer_ftt={text2img_ftt_per_layer[-1]:.5f}")
            finally:
                env.close()

    del vla
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
