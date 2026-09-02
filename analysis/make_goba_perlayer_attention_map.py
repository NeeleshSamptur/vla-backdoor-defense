#!/usr/bin/env python
"""Render GoBA's per-LAYER attention explainer: one panel per LLM layer
(mean over query tokens within that layer), instead of the existing
per-token explainers (one panel per token/DOF, mean over ALL layers).

Same underlying data source and visual style as
analysis/plot_pertoken_attention.py / make_goba_action_attention_map.py --
this just collapses the OTHER axis: where those average over layers and
keep tokens separate, this averages over tokens and keeps layers separate.
Answers "how does the attention pattern change layer by layer" instead of
"how does it differ token by token."

--query text|action selects which query set to use:
    text   -- the task-description tokens (extract_text2img_ftt.text2img_rows,
              all_layers=True)
    action -- the 7 action-generation steps (extract_action2img_ftt.
              action2img_rows_all_layers)

Usage (GoBA-OpenVLA conda env):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    export MUJOCO_GL=egl
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../analysis/make_goba_perlayer_attention_map.py \\
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \\
        --task-suite-name libero_object --task-id 9 --clean-seed 7 --trigger-seed 11 \\
        --base-seed 7 --query text \\
        --out ../vla-backdoor-defense/results/attention_maps_explainer/goba_text_perlayer_pertoken_t9.png
"""

from __future__ import annotations

import argparse
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
    get_libero_dummy_action, preprocess_like_policy,
)
from extract_action2img_ftt import action2img_rows_all_layers, load_vla_for_attention
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size
from plot_pertoken_attention import plot_pertoken


def text2img_rows_all_layers(vla, processor, image, desc, center_crop=True):
    """All-LLM-layers version of extract_text2img_ftt.text2img_rows.

    That function no longer exposes an all_layers option (it was removed
    upstream since this was written), so this duplicates just the forward
    pass and row/column bookkeeping locally instead of depending on it --
    avoids touching adapters/goba/extract_text2img_ftt.py, which appears to
    be under concurrent edit elsewhere. Returns [n_layers, n_query, n_patches],
    plus num_patches and the center-cropped display image (see run_one).
    """
    img = preprocess_like_policy(image, center_crop)
    display_img = np.array(img)
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
    if out.attentions is None:
        raise RuntimeError("model returned no attentions -- load via load_vla_for_attention (SDPA).")

    A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, seq, seq]
    n_txt = input_ids.shape[1] - 1
    T = A_stack.shape[-1]
    num_patches = T - n_txt - 1
    assert num_patches > 0, f"bad token layout: T={T} n_txt={n_txt}"

    img_cols = list(range(1, 1 + num_patches))
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    txt_rows = [1 + num_patches + r for r in txt_rel]
    rows = A_stack[:, txt_rows][:, :, img_cols].cpu().numpy()  # [L, Q, P]
    del out
    torch.cuda.empty_cache()
    return rows, num_patches, display_img


def run_one(vla, processor, cfg, resize_size, bddl_dir, task, base_seed, ep_idx, query, action_dim):
    env, desc = get_libero_env(task, cfg.model_family, resolution=256,
                                bddl_path=bddl_dir, seed=base_seed)
    try:
        obs = None
        for _ in range(ep_idx + 1):
            obs = env.reset()
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        observation, img = build_observation(obs, resize_size)
        if query == "text":
            rows, num_patches, display_img = text2img_rows_all_layers(
                vla, processor, img, desc, center_crop=cfg.center_crop)
        else:
            rows, num_patches = action2img_rows_all_layers(
                vla, processor, img, desc, action_dim, center_crop=cfg.center_crop)
            # action2img_rows_all_layers doesn't expose its cropped image, so
            # reproduce the same crop here purely for display alignment: the
            # heatmap patch grid is computed from the CROPPED image, so
            # overlaying it on the raw uncropped frame misaligns content near
            # the crop boundary (crop_scale=0.9 clips ~5% off each edge).
            display_img = np.array(preprocess_like_policy(img, cfg.center_crop))
        # rows: [n_layers, n_query, n_patches] -> mean over QUERY axis, keep layers.
        per_layer = rows.mean(axis=1)  # [n_layers, n_patches]
        return dict(image=display_img, attn=per_layer, prompt_desc=desc, n_layers=rows.shape[0])
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
    ap.add_argument("--query", choices=["text", "action"], default="text")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint}")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    resize_size = get_image_resize_size(cfg)
    action_dim = vla.get_action_dim(cfg.unnorm_key) if args.query == "action" else None

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)

    clean = run_one(vla, processor, cfg, resize_size, CLEAN_BDDL, task,
                     args.base_seed, args.clean_seed, args.query, action_dim)
    trig = run_one(vla, processor, cfg, resize_size, POISON_BDDL, task,
                    args.base_seed, args.trigger_seed, args.query, action_dim)

    layer_labels = [f"L{l}" for l in range(clean["n_layers"])]
    query_desc = "task-description tokens" if args.query == "text" else "7 action-generation steps"
    title = (f"GoBA checkpoint ({args.checkpoint}), PER-LAYER "
             f"(mean over {query_desc})")
    subtitle = (f"trigger type: physical toxic-box object  |  "
                f"clean: {clean['prompt_desc']!r}  |  trigger: {trig['prompt_desc']!r}")

    plot_pertoken(
        image_clean=clean["image"], attn_primary_clean=clean["attn"], tokens_clean=layer_labels,
        image_trigger=trig["image"], attn_primary_trigger=trig["attn"], tokens_trigger=layer_labels,
        title=title, subtitle=subtitle, out_path=args.out,
    )

    del vla, processor


if __name__ == "__main__":
    main()
