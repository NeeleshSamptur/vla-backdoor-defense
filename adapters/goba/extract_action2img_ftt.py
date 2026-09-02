#!/usr/bin/env python
"""GoBA extractor: action-token -> image attention, ALL layers, for FTT.

Companion to extract_text2img_ftt.py, same env/model loading, but the query
set is the 7 generated ACTION tokens instead of the task-description text.

GoBA is base OpenVLA (no OFT parallel decoding), so the action tokens do not
exist in a single forward pass the way the OFT adapters' 56 action-placeholder
tokens do -- predict_action() (modeling_prismatic.py:506-524) generates them
one at a time with generate(input_ids, max_new_tokens=get_action_dim(...)):
a plain causal, KV-cached decode. So this file calls generate() itself with
output_attentions=True, return_dict_in_generate=True, and reads off each
step's attention row instead of doing one static forward pass.

Per HF's GenerateDecoderOnlyOutput: gen.attentions is a tuple of
action_dim (7) steps, each itself a tuple of n_layers tensors shaped
[1, heads, q_len, kv_len]. Step 0 processes the whole prompt at once
(q_len == kv_len == prompt length); every step after has q_len == 1 (cache).
In both cases the query row that actually produced this step's token is the
LAST row -- so `A[0, :, -1, :]` is step-shape-agnostic. kv_len grows by one
each step, but the image-patch columns are the same fixed range throughout
(they're part of the unchanging prompt prefix), so one img_cols slice works
for every step.

Saved as plain .npz (NOT through detectors/schema.py -- ExtractedSample's
attn_text_image is declared 2D and this is [n_layers, action_dim, n_patches]),
matching the precedent set by --dump-all-layers in extract_text2img_ftt.py.
Score per layer offline: ftt_score(rows[l]) for l in range(n_layers), same
detectors/ftt.py used for the text-token case.

Usage (same env as extract_text2img_ftt.py):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../adapters/goba/extract_action2img_ftt.py \\
        --checkpoint exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \\
        --task-suite-name libero_goal --role attack \\
        --out-dir ../vla-backdoor-defense/results/goba_action_extracted
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

import numpy as np
import torch
from libero.libero import benchmark
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

# Reuse GoBA's own model-loading / preprocessing / observation helpers verbatim
# rather than duplicating them -- this script's own directory is on sys.path
# automatically because it's invoked as `python adapters/goba/<this file>.py`.
from extract_text2img_ftt import (
    CLEAN_BDDL, POISON_BDDL, DEVICE, NUM_STEPS_WAIT, VALID_SUITES,
    Cfg, build_observation, load_vla_for_attention, preprocess_like_policy,
)

VALID_SUITES = VALID_SUITES  # re-exported for clarity, unchanged


def action2img_rows_all_layers(vla, processor, image, desc, action_dim, center_crop=True):
    """Per-layer attention rows for the `action_dim` generated action tokens.

    Returns (rows, num_patches) where rows is [n_layers, action_dim, num_patches],
    head-averaged attention (raw post-softmax, not renormalized -- matching
    text2img_rows so the two are comparable through the same detectors/ftt.py).
    """
    img = preprocess_like_policy(image, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # Same 29871 append as predict_action / text2img_rows, needed so the
    # generation pass sees exactly the inputs the policy was trained on.
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids,
             torch.full((input_ids.shape[0], 1), 29871,
                        dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        attention_mask = torch.cat(
            (attention_mask,
             torch.ones((attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)
    n_txt = input_ids.shape[1] - 1  # text rows after BOS, same convention as text2img_rows

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = vla.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=action_dim, do_sample=False, use_cache=True,
            output_attentions=True, return_dict_in_generate=True,
        )

    step_attns = gen.attentions
    if step_attns is None or len(step_attns) != action_dim:
        got = 0 if step_attns is None else len(step_attns)
        raise RuntimeError(
            f"expected {action_dim} generation steps of attention, got {got} -- "
            "check that the model was loaded with SDPA (load_vla_for_attention), "
            "since flash_attention_2 silently returns no attentions here too.")

    n_layers = len(step_attns[0])
    # num_patches from step 0's kv_len: that step processes the whole prompt in
    # one shot (no cache yet), so kv_len == 1(BOS) + num_patches + n_txt, the
    # same layout text2img_rows derives from a static forward pass.
    first_kv_len = step_attns[0][0].shape[-1]
    num_patches = first_kv_len - n_txt - 1
    assert num_patches > 0, f"bad token layout: kv_len={first_kv_len} n_txt={n_txt}"
    img_cols = list(range(1, 1 + num_patches))

    rows = np.zeros((n_layers, action_dim, num_patches), dtype=np.float32)
    for step_idx, layer_attns in enumerate(step_attns):
        for layer_idx, A in enumerate(layer_attns):
            # A: [1, heads, q_len, kv_len]. The row that produced THIS step's
            # token is always the last query row, regardless of q_len (full
            # prompt on step 0, one cached row on every step after).
            row = A[0, :, -1, :].float().mean(0)  # head-mean -> [kv_len]
            rows[layer_idx, step_idx] = row[img_cols].cpu().numpy()

    del gen
    torch.cuda.empty_cache()
    return rows, num_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10,
                    help="tasks per suite; LIBERO suites have 10.")
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="episodes per task per condition; keep in lockstep with "
                         "extract_text2img_ftt.py and run_all_suites.sh.")
    ap.add_argument("--seed", type=int, default=7,
                    help="env construction seed; GoBA's eval.sh sweeps 7/42/1234")
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint",
                    help="same meaning as extract_text2img_ftt.py --eval-design.")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)

    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name})")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    n_llm_layers = vla.language_model.config.num_hidden_layers

    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in vla.norm_stats, (
        f"unnorm key {cfg.unnorm_key} not in norm_stats: {list(vla.norm_stats)[:5]}")
    action_dim = vla.get_action_dim(cfg.unnorm_key)
    print(f"[*] action_dim={action_dim}, n_llm_layers={n_llm_layers}")

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

                    observation, img = build_observation(obs, resize_size)
                    rows, num_patches = action2img_rows_all_layers(
                        vla, processor, img, desc, action_dim,
                        center_crop=cfg.center_crop)

                    ep_idx = cond_offset[cond] + ep
                    out_path = str(out_dir / f"{ckpt_tag}__{args.task_suite_name}"
                                             f"__t{task_id}__seed{args.seed}__s{ep_idx}"
                                             f"__{cond}__action.npz")
                    np.savez_compressed(
                        out_path,
                        attn_action_image_all_layers=rows,  # [n_layers, action_dim, num_patches]
                        label=int(trig),
                        task_id=task_id, seed=ep_idx,
                        n_layers=n_llm_layers,
                        action_dim=action_dim,
                        num_patches=num_patches,
                        meta_json=json.dumps({
                            "attack": "goba",
                            "checkpoint": args.checkpoint,
                            "task_suite_name": args.task_suite_name,
                            "task_description": desc,
                            "eval_design": args.eval_design,
                            "role": args.role,
                            "query": "action_tokens",
                        }),
                    )
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} rows={rows.shape}")
            finally:
                env.close()

    del vla, processor
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
