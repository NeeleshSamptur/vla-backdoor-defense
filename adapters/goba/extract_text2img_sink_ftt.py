#!/usr/bin/env python
"""GoBA extractor: desc_only TEXT tokens PLUS BOS and the boundary/"sink"
token (29871) -> image attention, ALL layers, raw dump for the full-dataset
AUROC sweep (not the schema.py single-layer path).

Same loop/env machinery as extract_text2img_ftt.py's main(), but the query
rows are BOS (position 0) + the desc_only description tokens + the sink
token (last position, previously excluded from every run in this project).
Row-normalize stays ON in the detector -- this file only extracts; scoring
(ftt_score, which normalizes) happens offline exactly like
analysis/layer_sweep.py already does for the plain text sweep.

Saved format matches results/goba_layersweep_libero_object/*.npz exactly
(attn_all_layers: [n_layers, n_query, n_patches], label, task_id, seed,
n_layers, meta_json) so it drops straight into the same offline scoring code.

Usage (GoBA-OpenVLA conda env):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../adapters/goba/extract_text2img_sink_ftt.py \\
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \\
        --task-suite-name libero_object --role attack \\
        --out-dir ../vla-backdoor-defense/results/goba_sink_layersweep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

DEFENSE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from libero.libero import benchmark

from extract_text2img_ftt import (
    CLEAN_BDDL, DEVICE, NUM_STEPS_WAIT, POISON_BDDL, VALID_SUITES,
    Cfg, _desc_token_row_indices, build_observation, get_libero_env,
    get_libero_dummy_action, load_vla_for_attention, preprocess_like_policy,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size


def text2img_rows_bos_desc_sink(vla, processor, image, desc, center_crop=True):
    """Returns (rows [n_layers, n_query, n_patches], num_patches) where
    n_query = 1(BOS) + len(desc_only tokens) + 1(sink)."""
    img = preprocess_like_policy(image, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids,
             torch.full((input_ids.shape[0], 1), 29871,
                        dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        attention_mask = torch.cat(
            (attention_mask,
             torch.ones((attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=input_ids, attention_mask=attention_mask,
                  pixel_values=inputs["pixel_values"], output_attentions=True,
                  return_dict=True)

    A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, seq, seq]
    n_txt = input_ids.shape[1] - 1
    T = A_stack.shape[-1]
    num_patches = T - n_txt - 1
    assert num_patches > 0, f"bad token layout: T={T} n_txt={n_txt}"

    img_cols = list(range(1, 1 + num_patches))
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    desc_rows = [1 + num_patches + r for r in txt_rel]
    bos_row = 0
    sink_row = 1 + num_patches + n_txt - 1

    all_rows = [bos_row] + desc_rows + [sink_row]
    rows = A_stack[:, all_rows][:, :, img_cols].cpu().numpy()  # [L, Q, P]
    del out
    torch.cuda.empty_cache()
    return rows, num_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
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

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    _raw = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    ckpt_tag = f"{_raw[:40]}_{hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_offset = {"clean": 0, "trigger": args.n_seeds}

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
                    rows, num_patches = text2img_rows_bos_desc_sink(
                        vla, processor, img, desc, center_crop=cfg.center_crop)

                    ep_idx = cond_offset[cond] + ep
                    out_path = out_dir / f"{ckpt_tag}__t{task_id}__s{ep_idx}__{cond}.npz"
                    np.savez_compressed(
                        out_path,
                        attn_all_layers=rows,
                        label=int(trig),
                        task_id=task_id, seed=ep_idx, n_layers=rows.shape[0],
                        meta_json=json.dumps({
                            "attack": "goba",
                            "checkpoint": args.checkpoint,
                            "task_suite_name": args.task_suite_name,
                            "role": args.role,
                            "query": "bos_desc_sink",
                            "task_description": desc,
                        }),
                    )
                    print(f"    t={task_id} s={ep_idx} {cond:8s} rows={rows.shape} {desc[:50]!r}")
            finally:
                env.close()

    del vla, processor
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
