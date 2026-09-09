#!/usr/bin/env python
"""ONE-OFF, ISOLATED experiment -- DropVLA equivalent of
adapters/goba/EXPERIMENTAL_extract_bos_sentinel_text2img_ftt.py: extract
BOS's and the trailing sentinel token's (29871) attention-to-image-patches
rows, all 32 layers, for BOTH cameras (primary and wrist) separately.

Neither row exists in the desc_only cache
(results/dropvla_text2img_layerwise_percam/*.npz, written by
extract_text2img_layerwise_percam_ftt.py): every DropVLA extraction slices
only the DESC-ONLY text rows over image-patch columns, for each camera.
BOS sits at position 0 (before either camera's patches even begin) and
29871 is appended as the LAST token of the text span, after the whole
prompt -- both fall outside the desc_only span entirely, so a fresh forward
pass is needed to capture them, exactly as for GoBA.

Query set for this experiment (matching GoBA's): {BOS, 29871} -- two rows
only, per camera, replacing the usual desc_only text tokens.

Unlike GoBA (one camera, settled-scene extraction with no rollout needed
before the forward pass), DropVLA has two cameras (primary+wrist) AND a
trigger that is physically gated mid-rollout (the object-lift condition;
see extract_text2img_ftt.py's module docstring, section 1). So this script
reuses `run_episode_capture` (imported unchanged from extract_text2img_ftt.py)
to drive one closed-loop rollout per episode with both backdoor flags off,
capture the frame where the lift/time condition first turns True, then runs
ONE forward pass at that frame per condition (clean, trigger) exactly like
extract_text2img_layerwise_percam_ftt.py's oft_text2img_rows_all_layers_both_cams
does -- except this one keeps BOS's and the sentinel's rows for BOTH
cameras instead of the desc_only tokens' rows.

Episode set: matched 1:1 by (task_id, seed) with the existing
dropvla_text2img_layerwise_percam cache -- same checkpoint
(openvla-7b+libero_spatial_no_noops_v5p00carefully+b8+lr-0.0003+lora-r32+dropout-0.0--seed42--paper,
suite libero_spatial), same --seed-base/--n-tasks/--n-seeds convention
(episode_idx is a DIRECT index into suite.get_task_init_states(task_id),
confirmed convention, no GoBA-style sequential env.reset() replay), so the
seeds re-drive the identical rollouts and reach the identical lift frames
-- letting this experiment's output be merged 1:1 by (task_id, seed) with
the existing desc_only cache, the same way GoBA's scoring script merges
its two datasets.

ISOLATION: new file, does not modify extract_text2img_ftt.py,
extract_text2img_layerwise_percam_ftt.py, or any other extraction/scoring
script -- only imports unchanged helpers (initialize_model,
run_episode_capture, LiftNotReached, NUM_STEPS_WAIT, VALID_SUITES) from
extract_text2img_ftt.py. Deleting this file and its sibling scoring script
fully reverts this exploration.

Usage (DropVLA's own conda env):
    conda activate dropvla
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/DropVLA:/home/grads/nsamptur/vla_bkd_def/GoBA_attack/BadLIBERO:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/DropVLA

    python .../adapters/dropvla/EXPERIMENTAL_extract_bos_sentinel_text2img_ftt.py \\
        --checkpoint "$RUN_DIR/openvla-7b+libero_spatial_no_noops_v5p00carefully+...--seed42--paper" \\
        --task-suite-name libero_spatial --trigger-mode vision \\
        --out-dir .../results/EXPERIMENTAL_dropvla_bos_sentinel
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
    NUM_STEPS_WAIT, VALID_SUITES, LiftNotReached, initialize_model, run_episode_capture,
)

SENTINEL_TOKEN_ID = 29871


def bos_sentinel_rows_all_layers_both_cams(vla, processor, proprio_projector, cfg, observation, desc):
    """Same forward-pass reassembly as
    extract_text2img_layerwise_percam_ftt.py's
    oft_text2img_rows_all_layers_both_cams, but returns BOS's row and the
    sentinel token's (29871) row instead of the desc_only text rows --
    all 32 layers, head-averaged, BOTH cameras kept separately.

    Returns (bos_primary [32, P], bos_wrist [32, P],
             sentinel_primary [32, P], sentinel_wrist [32, P], num_patches).
    """
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
    if not torch.all(input_ids[:, -1] == SENTINEL_TOKEN_ID):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.tensor([SENTINEL_TOKEN_ID]).long(), dim=0).to(input_ids.device)),
            dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)
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

    A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, T, T] head-averaged
    T = A_stack.shape[-1]
    assert T == 1 + n_img_cols + (input_ids2.shape[-1] - 1)

    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))

    bos_primary = A_stack[:, 0, primary_cols].cpu().numpy()        # position 0 = BOS
    bos_wrist = A_stack[:, 0, wrist_cols].cpu().numpy()
    sentinel_primary = A_stack[:, -1, primary_cols].cpu().numpy()  # last position = 29871
    sentinel_wrist = A_stack[:, -1, wrist_cols].cpu().numpy()

    del out
    torch.cuda.empty_cache()
    return bos_primary, bos_wrist, sentinel_primary, sentinel_wrist, num_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--trigger-mode", required=True, choices=["vision", "language", "joint"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=0)
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

                (bos_p_c, bos_w_c, sent_p_c, sent_w_c, num_patches) = bos_sentinel_rows_all_layers_both_cams(
                    model, processor, proprio_projector, base_cfg, clean_observation, clean_desc)
                (bos_p_t, bos_w_t, sent_p_t, sent_w_t, _) = bos_sentinel_rows_all_layers_both_cams(
                    model, processor, proprio_projector, base_cfg, trig_observation, trig_desc)

                for cond, trig, bos_p, bos_w, sent_p, sent_w in (
                    ("clean", False, bos_p_c, bos_w_c, sent_p_c, sent_w_c),
                    ("trigger", True, bos_p_t, bos_w_t, sent_p_t, sent_w_t),
                ):
                    L = bos_p.shape[0]
                    fname = (f"{file_tag}__{args.task_suite_name}__t{task_id}"
                             f"__s{episode_idx}__{cond}.npz")
                    np.savez_compressed(
                        out_dir / fname,
                        bos_primary=bos_p.astype(np.float32), bos_wrist=bos_w.astype(np.float32),
                        sentinel_primary=sent_p.astype(np.float32), sentinel_wrist=sent_w.astype(np.float32),
                        label=int(trig), task_id=task_id, seed=episode_idx,
                        role=args.role, num_patches=num_patches, n_layers=int(L),
                        task_description=task_description, activation_frame_t=t,
                    )
                print(f"    task={task_id} idx={episode_idx} activated@t={t}  saved BOS+sentinel, primary+wrist, all layers")
        finally:
            env.close()

    del model, processor, proprio_projector
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
