#!/usr/bin/env python
"""Render GoBA's per-layer, per-token attention as 32 SEPARATE images (one
per LLM layer), instead of one image averaged over all layers.

Each output image is laid out exactly like results/attention_maps_explainer/
01_GoBA_pertoken.png -- 2 rows (CLEAN / TRIGGER), one panel per REAL query
token (each task-description token, or each of the 7 action-generation
steps) plus a MEAN column -- except the attention values come from a single
LLM layer instead of being averaged across all layers. Reuses
extract_text2img_ftt/extract_action2img_ftt's all-layers extraction (same
data source as analysis/make_goba_perlayer_attention_map.py) and
analysis/plot_pertoken_attention.plot_pertoken for the actual rendering,
calling it once per layer with that layer's [n_query, n_patches] slice
instead of once with the tokens-collapsed, layer-averaged slice. Writes a
sidecar episode_info.json describing exactly which episode was used.

Usage (GoBA-OpenVLA conda env):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    export MUJOCO_GL=egl
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../analysis/make_goba_perlayer_separate_maps.py \\
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \\
        --task-suite-name libero_object --task-id 9 --clean-seed 7 --trigger-seed 11 \\
        --base-seed 7 --query text \\
        --out-dir ../vla-backdoor-defense/results/attention_maps_explainer/goba_perlayer_separate
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
    get_libero_dummy_action, preprocess_like_policy,
)
from extract_action2img_ftt import action2img_rows_all_layers, load_vla_for_attention
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size
from plot_pertoken_attention import plot_pertoken
from make_goba_action_attention_map import DOF_LABELS


def text2img_rows_all_layers(vla, processor, image, desc, center_crop=True):
    """All-LLM-layers version of extract_text2img_ftt.text2img_rows, with
    decoded per-token labels for the description span.

    Duplicated from analysis/make_goba_perlayer_attention_map.py (see that
    file's docstring for why) plus the token-label decoding this script
    additionally needs. Returns (rows [n_layers, n_query, n_patches],
    num_patches, token_labels [n_query], display_img [H,W,3] uint8).

    display_img is the SAME center-cropped+resized image the model actually
    saw (preprocess_like_policy's output), not the raw uncropped observation
    -- the heatmap patch grid is computed from that cropped image, so
    overlaying it on the uncropped frame misaligns content near the crop
    boundary (crop_scale=0.9 clips ~5% off each edge; edge objects like the
    basket would appear "cut" in a way the raw image doesn't show).
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

    # txt_rel indexes the NO-BOS, no-special-tokens offset mapping that
    # _desc_token_row_indices uses (tok(prompt, add_special_tokens=False)),
    # but input_ids here (from processor(prompt, img)) has a BOS token
    # prepended at position 0 -- so the same token sits at input_ids
    # position r+1, not r. Indexing with bare r shifts every label back by
    # one (e.g. the real last token 'basket' silently becomes unlabeled,
    # with the second-to-last token's row wrongly captioned 'basket').
    token_labels = [processor.tokenizer.decode([input_ids[0, r + 1].item()])
                    for r in txt_rel]
    del out
    torch.cuda.empty_cache()
    return rows, num_patches, token_labels, display_img


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
            rows, num_patches, token_labels, display_img = text2img_rows_all_layers(
                vla, processor, img, desc, center_crop=cfg.center_crop)
        else:
            rows, num_patches = action2img_rows_all_layers(
                vla, processor, img, desc, action_dim, center_crop=cfg.center_crop)
            token_labels = DOF_LABELS
            # action2img_rows_all_layers doesn't expose its cropped image, so
            # reproduce the same crop here purely for display alignment --
            # same reasoning as text2img_rows_all_layers's display_img above.
            display_img = np.array(preprocess_like_policy(img, cfg.center_crop))
        # rows: [n_layers, n_query, n_patches]
        return dict(image=display_img, rows=rows, prompt_desc=desc, n_layers=rows.shape[0],
                    token_labels=token_labels)
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
    action_dim = vla.get_action_dim(cfg.unnorm_key) if args.query == "action" else None

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)

    clean = run_one(vla, processor, cfg, resize_size, CLEAN_BDDL, task,
                     args.base_seed, args.clean_seed, args.query, action_dim)
    trig = run_one(vla, processor, cfg, resize_size, POISON_BDDL, task,
                    args.base_seed, args.trigger_seed, args.query, action_dim)

    n_layers = clean["n_layers"]
    query_desc = "task-description tokens" if args.query == "text" else "7 action-generation steps"
    subtitle = (f"trigger type: physical toxic-box object  |  "
                f"clean: {clean['prompt_desc']!r}  |  trigger: {trig['prompt_desc']!r}")

    for l in range(n_layers):
        out_path = out_dir / f"layer_{l:02d}.png"
        title = f"GoBA checkpoint ({args.checkpoint})  --  layer {l}/{n_layers - 1}, per {query_desc}"
        plot_pertoken(
            image_clean=clean["image"], attn_primary_clean=clean["rows"][l],
            tokens_clean=clean["token_labels"],
            image_trigger=trig["image"], attn_primary_trigger=trig["rows"][l],
            tokens_trigger=trig["token_labels"],
            title=title, subtitle=subtitle, out_path=str(out_path),
        )
        print(f"[*] wrote {out_path}")

    episode_info = dict(
        checkpoint=args.checkpoint,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        task_description=task.language,
        clean_bddl_dir=str(CLEAN_BDDL),
        poison_bddl_dir=str(POISON_BDDL),
        clean_ep_idx=args.clean_seed,
        trigger_ep_idx=args.trigger_seed,
        base_seed=args.base_seed,
        query=args.query,
        clean_prompt_desc=clean["prompt_desc"],
        trigger_prompt_desc=trig["prompt_desc"],
        clean_token_labels=clean["token_labels"],
        trigger_token_labels=trig["token_labels"],
        n_layers=n_layers,
    )
    info_path = out_dir / "episode_info.json"
    info_path.write_text(json.dumps(episode_info, indent=2))
    print(f"[*] wrote {info_path}")

    del vla, processor


if __name__ == "__main__":
    main()
