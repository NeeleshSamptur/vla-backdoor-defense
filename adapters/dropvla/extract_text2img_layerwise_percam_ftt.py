#!/usr/bin/env python
"""DropVLA extractor: desc_only text2img FTT, ALL 32 layers, BOTH cameras
(primary and wrist) kept separately.

extract_text2img_ftt.py's own --all-layers flag only keeps the primary
camera's per-layer rows (attn_text_image_layers); wrist only ever gets the
single last-layer slice (attn_text_image_wrist). This script fixes that gap
by keeping BOTH cameras' full [L, n_desc, num_patches] arrays, needed to
replicate this project's GoBA report (per-layer AUROC, Flatten method,
Shared-reference method) per camera for DropVLA.

Reuses extract_text2img_ftt.py's model loading (the already-fixed eager
attention path via the module-level `_get_vla_eager` monkeypatch applied at
import time), rollout/lift-detection, and prompt/token bookkeeping
UNCHANGED -- only the forward-pass function is new, and only to keep the
wrist camera's per-layer slice that the existing one discards.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.libero.run_libero_eval import GenerateConfig, prepare_observation
from experiments.robot.openvla_utils import DEVICE, normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
from prismatic.vla.constants import IGNORE_INDEX

from extract_text2img_ftt import (  # noqa: E402 -- reused unchanged
    NUM_STEPS_WAIT, VALID_SUITES, LiftNotReached, _desc_token_row_indices,
    initialize_model, run_episode_capture,
)


def oft_text2img_rows_all_layers_both_cams(vla, processor, proprio_projector, cfg, observation, desc,
                                            text_scope="desc_only"):
    """Same reassembly as extract_text2img_ftt.py's oft_text2img_rows, but
    keeps ALL 32 layers for BOTH primary and wrist cameras (head-averaged,
    same convention: `.float().mean(dim=1)` per layer)."""
    full = observation["full_image"]
    wrist = observation["wrist_image"]
    images = prepare_images_for_vla([full, wrist], cfg)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, images[0]).to(DEVICE, dtype=torch.bfloat16)
    wrist_in = processor(prompt, images[1]).to(DEVICE, dtype=torch.bfloat16)
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_in["pixel_values"]], dim=1)
    proprio = normalize_proprio(observation["state"].copy(), vla.norm_stats[cfg.unnorm_key]["proprio"])

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)
    n_txt = input_ids.shape[-1] - 1
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    input_ids2, attention_mask2 = vla._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels2 = vla._prepare_labels_for_action_prediction(labels, input_ids2)
    input_embeddings = vla.get_input_embeddings()(input_ids2)
    all_actions_mask = vla._process_action_masks(labels2)
    language_embeddings = input_embeddings[~all_actions_mask].reshape(
        input_embeddings.shape[0], -1, input_embeddings.shape[2])
    projected = vla._process_vision_features(inputs["pixel_values"], language_embeddings, use_film=False)
    proprio_t = torch.tensor(proprio, device=projected.device, dtype=projected.dtype)
    projected = vla._process_proprio_features(projected, proprio_t, proprio_projector)
    zeroed = input_embeddings * ~all_actions_mask.unsqueeze(-1)
    mm, mm_mask = vla._build_multimodal_attention(zeroed, projected, attention_mask2)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla.language_model(
            input_ids=None, attention_mask=mm_mask, inputs_embeds=mm,
            output_attentions=True, return_dict=True)

    num_patches = vla.vision_backbone.get_num_patches()
    n_img_cols = projected.shape[1]
    assert n_img_cols == num_patches * 2 + 1

    if text_scope == "desc_only":
        txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    else:
        txt_rel = list(range(n_txt))
    txt_rows = [1 + n_img_cols + r for r in txt_rel]

    A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, T, T]
    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))
    rows_primary_all = A_stack[:, txt_rows][:, :, primary_cols].cpu().numpy()  # [L, n_txt, num_patches]
    rows_wrist_all = A_stack[:, txt_rows][:, :, wrist_cols].cpu().numpy()      # [L, n_txt, num_patches]

    del out
    torch.cuda.empty_cache()
    return rows_primary_all, rows_wrist_all, num_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--trigger-mode", required=True, choices=["vision", "language", "joint"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--text-scope", choices=["desc_only", "all"], default="desc_only")
    ap.add_argument("--load-in-4bit", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--role", choices=["attack", "clean_baseline"], default="attack")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)

    base_cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint, task_suite_name=args.task_suite_name,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, lora_rank=32,
        load_in_8bit=False, load_in_4bit=args.load_in_4bit, center_crop=False,
        num_open_loop_steps=8, num_steps_wait=NUM_STEPS_WAIT, env_img_res=256,
        use_backdoor_instruction=False, use_visual_backdoor=False,
        language_suffix="carefully", conditional_backdoor_on_lift=True,
        backdoor_activation_height_m=0.06, backdoor_activation_time_s=0.02,
        visual_backdoor_dot_x=10, visual_backdoor_dot_y=10, visual_backdoor_dot_radius=5,
        visual_backdoor_dot_alpha=255, visual_backdoor_dot_shape="circle",
        cover_wrist_lower_quarter=False, seed=args.seed,
    )

    print(f"[*] loading {args.checkpoint} (suite={args.task_suite_name}, trigger_mode={args.trigger_mode})")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(base_cfg)
    resize_size = get_image_resize_size(base_cfg)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    _raw_tag = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    ckpt_tag = f"{_raw_tag[:40]}_{hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_tag = f"{ckpt_tag}__cf-lift"

    use_vision = args.trigger_mode in ("vision", "joint")
    use_language = args.trigger_mode in ("language", "joint")

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        n_avail = init_states.shape[0]
        env, task_description = get_libero_env(task, base_cfg.model_family, resolution=base_cfg.env_img_res)
        try:
            for k in range(args.n_seeds):
                episode_idx = args.seed_base + k
                if episode_idx >= n_avail:
                    print(f"    [!] task={task_id}: only {n_avail} curated init states, skipping {episode_idx}")
                    continue
                try:
                    obs, t, target_body_id, target_body_ids, table_z = run_episode_capture(
                        base_cfg, env, task_description, model, action_head, proprio_projector,
                        processor, resize_size, init_states[episode_idx])
                except LiftNotReached as e:
                    print(f"    task={task_id} idx={episode_idx}: [skip] {e}")
                    continue

                clean_cfg = replace(base_cfg, use_visual_backdoor=False, use_backdoor_instruction=False)
                trig_cfg = replace(base_cfg, use_visual_backdoor=use_vision, use_backdoor_instruction=use_language)
                clean_observation, _ = prepare_observation(clean_cfg, obs, resize_size, backdoor_active=False)
                trig_observation, _ = prepare_observation(trig_cfg, obs, resize_size, backdoor_active=True)
                clean_desc = task_description
                trig_desc = f"{task_description.strip()} {trig_cfg.language_suffix.strip()}".strip() \
                    if use_language else task_description

                primary_clean, wrist_clean, num_patches = oft_text2img_rows_all_layers_both_cams(
                    model, processor, proprio_projector, base_cfg, clean_observation, clean_desc)
                primary_trig, wrist_trig, _ = oft_text2img_rows_all_layers_both_cams(
                    model, processor, proprio_projector, base_cfg, trig_observation, trig_desc)

                for cond, trig, primary, wrist in (
                    ("clean", False, primary_clean, wrist_clean),
                    ("trigger", True, primary_trig, wrist_trig),
                ):
                    L, n_txt = primary.shape[0], primary.shape[1]
                    fname = (f"{file_tag}__{args.task_suite_name}__t{task_id}"
                             f"__s{episode_idx}__{cond}.npz")
                    np.savez_compressed(
                        out_dir / fname,
                        attn_primary_layers=primary.astype(np.float32),  # [L, n_desc, num_patches]
                        attn_wrist_layers=wrist.astype(np.float32),      # [L, n_desc, num_patches]
                        label=int(trig), task_id=task_id, seed=episode_idx,
                        role=args.role, num_patches=num_patches, n_layers=int(L),
                        task_description=task_description, activation_frame_t=t,
                    )
                print(f"    task={task_id} idx={episode_idx} activated@t={t}  saved primary+wrist, all layers")
        finally:
            env.close()

    del model, processor, proprio_projector
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
