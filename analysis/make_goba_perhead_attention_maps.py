#!/usr/bin/env python
"""Render actual attention-map images for GoBA's "good heads" -- the
specific (layer, head) cells identified by
analysis/score_goba_perhead_text2img.py as carrying the strongest
per-head text2img FTT signal (head 1 @ layer 18, head 6 @ layer 13, etc.),
one clean-vs-trigger image per head, using the SAME plot_pertoken renderer
every other attention-map image in this repo uses.

This is a NEW file: does not edit extract_perhead_img2img_ftt.py,
extract_text2img_ftt.py, or make_goba_perlayer_separate_maps.py. Reuses
each unchanged (model loading, description-token-row lookup, the renderer)
and adds only the per-head forward pass + the good-heads loop.

Usage (GoBA-OpenVLA conda env):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    export MUJOCO_GL=egl
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../analysis/make_goba_perhead_attention_maps.py \\
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \\
        --task-suite-name libero_object --task-id 0 --clean-seed 0 --trigger-seed 0 \\
        --base-seed 7 \\
        --out-dir ../vla-backdoor-defense/results/attention_maps_goba_perhead
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "goba"))
sys.path.insert(0, str(DEFENSE / "analysis"))

from libero.libero import benchmark

import numpy as np
import torch

from extract_text2img_ftt import (
    CLEAN_BDDL, DEVICE, NUM_STEPS_WAIT, POISON_BDDL,
    Cfg, _desc_token_row_indices, build_observation, get_libero_env,
    get_libero_dummy_action, load_vla_for_attention, preprocess_like_policy,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size
from plot_pertoken_attention import plot_pertoken

# (head, layer) pairs from score_goba_perhead_text2img.py's 100v100 ranking,
# ordered best-first.
GOOD_HEADS = [
    (1, 18), (6, 13), (11, 15), (29, 13), (24, 9),
    (12, 31), (15, 31), (8, 24), (4, 21), (22, 22),
]


def text2img_rows_perhead(vla, processor, image, desc, center_crop=True):
    """Per-head, all-layers version of text2img_rows_all_layers (see
    make_goba_perlayer_separate_maps.py) -- keeps the head axis instead of
    averaging over it. Returns (rows [n_layers, n_heads, n_query, n_patches],
    token_labels [n_query], display_img [H,W,3] uint8)."""
    img = preprocess_like_policy(image, center_crop)
    display_img = np.array(img)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.full((input_ids.shape[0], 1), 29871,
                                    dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=input_ids, attention_mask=attention_mask,
                  pixel_values=inputs["pixel_values"], output_attentions=True, return_dict=True)

    A_stack = torch.stack([a[0] for a in out.attentions]).float()  # [L, H, seq, seq] -- heads kept
    n_txt = input_ids.shape[1] - 1
    T = A_stack.shape[-1]
    num_patches = T - n_txt - 1
    img_cols = list(range(1, 1 + num_patches))
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    txt_rows = [1 + num_patches + r for r in txt_rel]
    rows = A_stack[:, :, txt_rows][:, :, :, img_cols].cpu().numpy()  # [L, H, Q, P]

    token_labels = [processor.tokenizer.decode([input_ids[0, r + 1].item()]) for r in txt_rel]
    del out
    torch.cuda.empty_cache()
    return rows, token_labels, display_img


def run_one(vla, processor, cfg, resize_size, bddl_dir, task, base_seed, ep_idx):
    env, desc = get_libero_env(task, cfg.model_family, resolution=256,
                                bddl_path=bddl_dir, seed=base_seed)
    try:
        obs = None
        for _ in range(ep_idx + 1):
            obs = env.reset()
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        observation, img = build_observation(obs, resize_size)
        rows, token_labels, display_img = text2img_rows_perhead(vla, processor, img, desc, cfg.center_crop)
        return dict(image=display_img, rows=rows, prompt_desc=desc, token_labels=token_labels)
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", default="libero_object")
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--clean-seed", type=int, required=True, help="ep_idx, not an RNG seed")
    ap.add_argument("--trigger-seed", type=int, required=True, help="ep_idx, not an RNG seed")
    ap.add_argument("--base-seed", type=int, default=7)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint}")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)

    clean = run_one(vla, processor, cfg, resize_size, CLEAN_BDDL, task, args.base_seed, args.clean_seed)
    trig = run_one(vla, processor, cfg, resize_size, POISON_BDDL, task, args.base_seed, args.trigger_seed)

    subtitle = (f"trigger type: physical toxic-box object  |  "
                f"clean: {clean['prompt_desc']!r}  |  trigger: {trig['prompt_desc']!r}")

    for head, layer in GOOD_HEADS:
        out_path = out_dir / f"head_{head:02d}_layer_{layer:02d}.png"
        title = f"GoBA checkpoint  --  layer {layer}, HEAD {head} only, per task-description token"
        plot_pertoken(
            image_clean=clean["image"], attn_primary_clean=clean["rows"][layer, head],
            tokens_clean=clean["token_labels"],
            image_trigger=trig["image"], attn_primary_trigger=trig["rows"][layer, head],
            tokens_trigger=trig["token_labels"],
            title=title, subtitle=subtitle, out_path=str(out_path),
        )
        print(f"[*] wrote {out_path}")

    episode_info = dict(
        checkpoint=args.checkpoint, task_suite_name=args.task_suite_name, task_id=args.task_id,
        task_description=task.language, clean_ep_idx=args.clean_seed, trigger_ep_idx=args.trigger_seed,
        base_seed=args.base_seed, good_heads=GOOD_HEADS,
        clean_prompt_desc=clean["prompt_desc"], trigger_prompt_desc=trig["prompt_desc"],
        clean_token_labels=clean["token_labels"], trigger_token_labels=trig["token_labels"],
    )
    (out_dir / "episode_info.json").write_text(json.dumps(episode_info, indent=2))
    print(f"[*] wrote {out_dir / 'episode_info.json'}")

    del vla, processor


if __name__ == "__main__":
    main()
