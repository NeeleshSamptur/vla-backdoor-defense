#!/usr/bin/env python
"""Same as analysis/make_goba_perlayer_separate_maps.py (32 separate per-layer
images, CLEAN vs TRIGGER, plot_pertoken_attention.plot_pertoken layout) but
the query set is desc_only TEXT tokens PLUS two extra rows that every other
run in this project has excluded:

  - BOS (sequence position 0)
  - the boundary/"sink" token (29871, the last position before the model
    would start generating actions) -- previously left out of desc_only
    because _desc_token_row_indices only returns rows whose character span
    overlaps the description text, and the boundary token's span doesn't.

Row-normalize is back ON (this is plain FTT, not the no-normalize variant).
Motivation: recent attention-sink literature reports backdoor triggers
specifically hijack attention away from BOS/sink positions, a signal this
project has never queried before now.

Usage (GoBA-OpenVLA conda env, same as make_goba_perlayer_separate_maps.py):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    export MUJOCO_GL=egl
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../analysis/make_goba_perlayer_sink_maps.py \\
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \\
        --task-suite-name libero_object --task-id 9 --clean-seed 7 --trigger-seed 11 \\
        --base-seed 7 \\
        --out-dir ../vla-backdoor-defense/results/attention_maps_explainer/goba_perlayer_sink
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
from extract_action2img_ftt import load_vla_for_attention
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size
from plot_pertoken_attention import plot_pertoken


def text2img_rows_with_sink(vla, processor, image, desc, center_crop=True):
    """desc_only text rows PLUS BOS (row 0) and the boundary/sink token
    (last row), for ALL layers. Returns (rows [n_layers, n_query, n_patches],
    num_patches, token_labels [n_query], display_img)."""
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
    n_txt = input_ids.shape[1] - 1   # excludes BOS, includes the boundary token
    T = A_stack.shape[-1]
    num_patches = T - n_txt - 1
    assert num_patches > 0, f"bad token layout: T={T} n_txt={n_txt}"

    img_cols = list(range(1, 1 + num_patches))
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    desc_rows = [1 + num_patches + r for r in txt_rel]
    # input_ids has NO image tokens (those are a separate pixel_values tensor
    # spliced in by the model) -- it's just [BOS, text...], so the token at
    # relative offset r sits at input_ids position r+1, not 1+num_patches+r
    # (that latter formula is for indexing the ATTENTION MATRIX rows only).
    desc_labels = [processor.tokenizer.decode([input_ids[0, r + 1].item()])
                   for r in txt_rel]

    bos_row = 0
    sink_row = 1 + num_patches + n_txt - 1   # last position = the boundary/29871 token

    all_rows = [bos_row] + desc_rows + [sink_row]
    all_labels = ["[BOS]"] + desc_labels + ["[SINK]"]

    rows = A_stack[:, all_rows][:, :, img_cols].cpu().numpy()  # [L, Q, P]
    del out
    torch.cuda.empty_cache()
    return rows, num_patches, all_labels, display_img


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
        rows, num_patches, token_labels, display_img = text2img_rows_with_sink(
            vla, processor, img, desc, center_crop=cfg.center_crop)
        return dict(image=display_img, rows=rows, prompt_desc=desc, n_layers=rows.shape[0],
                    token_labels=token_labels)
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", default="libero_object")
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--clean-seed", type=int, required=True)
    ap.add_argument("--trigger-seed", type=int, required=True)
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

    clean = run_one(vla, processor, cfg, resize_size, CLEAN_BDDL, task,
                     args.base_seed, args.clean_seed)
    trig = run_one(vla, processor, cfg, resize_size, POISON_BDDL, task,
                    args.base_seed, args.trigger_seed)

    n_layers = clean["n_layers"]
    subtitle = (f"trigger type: physical toxic-box object  |  BOS + desc_only tokens + boundary/sink token  |  "
                f"clean: {clean['prompt_desc']!r}  |  trigger: {trig['prompt_desc']!r}")

    for l in range(n_layers):
        out_path = out_dir / f"layer_{l:02d}.png"
        title = f"GoBA checkpoint ({args.checkpoint})  --  layer {l}/{n_layers - 1}, BOS+desc+sink tokens"
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
        query="text_desc_plus_bos_sink",
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
