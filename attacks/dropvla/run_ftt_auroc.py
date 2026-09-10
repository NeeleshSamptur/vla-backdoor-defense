#!/usr/bin/env python
"""DropVLA: text-to-image FTT -> AUROC, desc_only, both cameras.

Drives one closed-loop LIBERO rollout per (task, curated init state) with
DropVLA's own model/env/observation-prep code, captures the frame where its
physically-gated trigger condition first turns True, runs one forward pass
each for the clean and triggered version of that SAME frame, and scores the
task-description-only text-to-image attention (all LLM layers, primary and
wrist camera scored separately) with attacks.common's FTT statistic and
AUROC.

------------------------------------------------------------------------
Why one rollout yields both a clean AND a triggered sample
------------------------------------------------------------------------
DropVLA's own `run_libero_eval.py` (lines ~741-936) only ever applies its
trigger (a red dot the size of about one vision-patch, drawn at a fixed
image corner, or a language suffix) once `backdoor_active` -- computed from
real physics: the tracked object's height above the table exceeding
`backdoor_activation_height_m`, sustained for `backdoor_activation_time_s`
after the first gripper closure -- turns True. Before that instant, a
clean-mode rollout and a triggered-mode rollout are driving the identical
policy on the identical, unmodified observation, so they cannot have
diverged yet. This script exploits that: it drives ONE rollout per episode
with both backdoor flags off (matching DropVLA's own MODE=clean driving
exactly), using DropVLA's own lift-detection helpers to find the frame
where `backdoor_active` first turns True, then takes two forward passes on
that SAME frame -- one through `prepare_observation` with the backdoor
flags off (clean) and one with them on per --trigger-mode (triggered). This
is the "faithful" rollout-based capture rather than a cheaper static-frame
capture, because the whole premise of this attack is a trigger gated on a
physical event the policy itself causes mid-task -- a settled, no-op frame
would be off-distribution for a model that was never shown the dot that
early.

Because the trigger is confirmed to be roughly one patch in size (a 5px-
radius dot against a 224x224 frame downsampled to ~16x16 patches), this
detector may simply have very little signal to work with here even when it
works well on attacks with larger visual triggers -- see the results this
script prints/saves for the actual per-camera AUROC.

------------------------------------------------------------------------
Why attn_implementation="eager", not DropVLA's own (unset -> sdpa) default
------------------------------------------------------------------------
This script requests output_attentions=True directly on `vla.language_model`
(the same forward-pass reconstruction as every other adapter's FTT
extractor -- `predict_action`'s `generate()` path never surfaces
attentions). Under transformers 4.40.1, requesting output_attentions=True on
an attn_implementation="sdpa" model normally falls back to the manual/eager
attention body -- but ONLY if the causal mask passed down is non-None.
LlamaModel._update_causal_mask has an SDPA-specific optimization that
returns an explicit None mask whenever attention_mask is all-1s and
query_length==key_value_length (exactly this script's single forward pass,
no padding, no KV cache) and relies on SDPA's own is_causal=True kernel
argument instead. The eager fallback body never receives that flag and,
given attention_mask=None, applies no masking at all -- so out.attentions
comes back fully bidirectional, not causal. Confirmed empirically on GoBA
(same prismatic/Llama family, identical load pattern): image-patch query
rows had nonzero mass on later-patch and image->text columns that must be
exactly zero under real causal masking; switching to eager fixed it. Since
DropVLA's own action-generation forward pass never requests
output_attentions, it runs correctly under either implementation, so
forcing eager here does not change what DropVLA's own rollout does -- it
only fixes what THIS script's separate attention-probing forward pass
returns. The patch target is `experiments.robot.robot_utils.get_vla` (not
`openvla_utils.get_vla`), because `robot_utils.get_model` holds its own
`from experiments.robot.openvla_utils import get_vla` binding and calls
that name directly -- patching the origin module alone would not reach
that call site.

Usage (DropVLA's own conda env and runtime assumptions, per dropvla_env.sh:
dedicated `dropvla` conda env, DROPVLA_ROOT/LIBERO_PATH/PYTHONPATH,
MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl, CUDA_VISIBLE_DEVICES):
    source /home/grads/nsamptur/vla_bkd_def/DropVLA/dropvla_env.sh
    cd /home/grads/nsamptur/vla_bkd_def/DropVLA

    python /home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/attacks/dropvla/run_ftt_auroc.py \
        --checkpoint "$RUN_DIR/openvla-7b+libero_object_no_noops_v5p00carefully+...--seed42--paper" \
        --task-suite-name libero_object --trigger-mode vision \
        --out results/dropvla_ftt_auroc.json

This attack's own paper protocol (run_eval.sh) uses libero_spatial with 10
tasks x 20 trials; this script's defaults (--n-tasks 10 --n-seeds 10) give
up to 100 clean + 100 trigger episodes (fewer if the lift/time condition is
never reached for a given curated init state -- the episode is skipped
entirely in that case, not counted as either class).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)

import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    TASK_MAX_STEPS,
    TaskSuite,
    _find_target_body_id_from_description,
    _get_body_z,
    _get_sim_and_body_maps,
    _get_target_body_ids_from_env,
    _find_body_id_by_keywords,
    initialize_model,
    prepare_observation,
    process_action,
)
from experiments.robot.openvla_utils import DEVICE, normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere
from prismatic.vla.constants import IGNORE_INDEX

import experiments.robot.robot_utils as _robot_utils
import experiments.robot.openvla_utils as _openvla_utils

from attacks.common import compute_auroc, compute_ftt

NUM_STEPS_WAIT = 10
VALID_SUITES = (
    TaskSuite.LIBERO_SPATIAL.value,
    TaskSuite.LIBERO_OBJECT.value,
    TaskSuite.LIBERO_GOAL.value,
    TaskSuite.LIBERO_10.value,
)


def _get_vla_eager(cfg):
    """Same as DropVLA's own openvla_utils.get_vla, with ONE change:
    attn_implementation="eager" instead of the unset default -- see module
    docstring for why that matters for output_attentions=True specifically.
    """
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    print("[*] loading VLA with EAGER attention (patched -- SDPA+output_attentions "
          "silently returns bidirectional, not causal, attention; see module docstring)")

    if not _openvla_utils.model_is_on_hf_hub(cfg.pretrained_checkpoint):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        _openvla_utils.update_auto_map(cfg.pretrained_checkpoint)
        _openvla_utils.check_model_logic_mismatch(cfg.pretrained_checkpoint)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="eager",  # <-- the only deviation from DropVLA's own get_vla()
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_film:
        vla = _openvla_utils._apply_film_to_vla(vla, cfg)

    vla.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()

    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    _openvla_utils._load_dataset_stats(vla, cfg.pretrained_checkpoint)
    return vla


# `robot_utils.get_model` calls the module-global name `get_vla` it imported
# from openvla_utils -- patch that binding, not the origin module, or this
# would never be reached (see module docstring).
_robot_utils.get_vla = _get_vla_eager


# ---------------------------------------------------------------------------
# Text-token bookkeeping
# ---------------------------------------------------------------------------

def _desc_token_row_indices(processor, prompt: str, desc: str, n_txt: int) -> list[int]:
    """Map the task-description's character span in the real prompt to
    prompt-token row indices, via the fast tokenizer's own offset mapping
    (rather than guessing from template token counts), so desc_only excludes
    the "In: What action should the robot take to ... ?\\nOut:" template
    tokens and any proprio/action tokens -- only the words describing the
    task itself.
    """
    tok = processor.tokenizer
    if not tok.is_fast:
        raise RuntimeError(
            "desc_only scope needs a fast tokenizer for character offset "
            "mapping; got a slow tokenizer.")

    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    if char_start == -1:
        raise ValueError(
            f"could not locate description {desc_lower!r} inside prompt {prompt!r}; "
            "the prompt template changed and _desc_token_row_indices needs updating.")
    char_end = char_start + len(desc_lower)

    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    assert 0 <= n_txt - len(offsets) <= 1, (
        f"token count mismatch: offsets={len(offsets)} n_txt={n_txt}; n_txt must "
        "come from tokenizing this same prompt plus at most one appended token.")

    rows = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    assert rows, f"no tokens overlap description span in prompt {prompt!r}"
    return rows


# ---------------------------------------------------------------------------
# Forward pass: desc_only text->image attention, all layers, both cameras
# ---------------------------------------------------------------------------

def capture_text_to_image_attention(vla, processor, proprio_projector, cfg, observation, desc):
    """Forward pass with output_attentions=True on an already-prepared
    (possibly triggered) `prepare_observation(...)` output. Reassembles the
    fused [BOS][primary patches][wrist patches][proprio][text] sequence with
    the model's own `_process_*` helpers and calls the language model
    directly, since `predict_action`'s `generate()` path never surfaces
    attentions.

    Returns (primary, wrist): each [n_layers, n_desc_tokens, num_patches]
    raw (non-negative, not yet row-normalized) attention, head-averaged per
    layer -- compute_ftt row-normalizes and averages over layers itself.
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
    assert n_img_cols == num_patches * 2 + 1, (
        f"expected 2 camera patch blocks + 1 proprio token, got {n_img_cols}")

    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    txt_rows = [1 + n_img_cols + r for r in txt_rel]
    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))

    # out.attentions already holds every layer from this one forward pass
    # (output_attentions=True computes all of them regardless), so keeping
    # all layers instead of just the last is free of extra compute.
    A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, T, T]
    primary = A_stack[:, txt_rows][:, :, primary_cols].cpu().numpy()  # [L, n_desc, num_patches]
    wrist = A_stack[:, txt_rows][:, :, wrist_cols].cpu().numpy()      # [L, n_desc, num_patches]

    del out
    torch.cuda.empty_cache()
    return primary, wrist


# ---------------------------------------------------------------------------
# Rollout: drive to the physically-gated trigger frame
# ---------------------------------------------------------------------------

class LiftNotReached(Exception):
    pass


def _resolve_target_body(env, sim, body_names, name_to_id, task_description):
    """Port of run_libero_eval.run_episode's target/table resolution: prefer
    env.obj_of_interest, fall back to LIBERO-Spatial's fixed keyword
    vocabulary, then description-derived body-name matching; fall back to
    the object's own resting height if no "table" body exists (e.g.
    LIBERO-Object has no body named "table" at all)."""
    table_z = None
    target_body_id = None
    if sim is None:
        return table_z, target_body_id, []

    table_body_id = _find_body_id_by_keywords(body_names, name_to_id, ["table"])
    if table_body_id is not None:
        table_z = _get_body_z(sim, table_body_id)

    target_body_ids = _get_target_body_ids_from_env(env, body_names, name_to_id) or []
    if target_body_ids:
        target_body_id = target_body_ids[0]
    else:
        td = task_description.lower()
        target_keywords = [kw for kw in ["bowl", "mug", "can", "bottle", "plate", "ramekin", "cup"] if kw in td]
        if not target_keywords:
            target_keywords = ["bowl", "mug", "can", "bottle", "plate", "ramekin", "cup"]
        target_body_id = _find_body_id_by_keywords(body_names, name_to_id, target_keywords)
    if target_body_id is None:
        target_body_id = _find_target_body_id_from_description(task_description, body_names, name_to_id)

    if table_z is None and target_body_id is not None:
        table_z = _get_body_z(sim, target_body_id)

    return table_z, target_body_id, (target_body_ids or ([target_body_id] if target_body_id is not None else []))


def run_to_trigger_frame(drive_cfg, env, task_description, model, action_head, proprio_projector,
                          processor, resize_size, initial_state):
    """Drive one closed-loop rollout with BOTH backdoor flags off (matching
    MODE=clean's driving exactly -- see module docstring) until DropVLA's
    own lift/time condition first turns True, then return the raw `obs` at
    that instant. Raises LiftNotReached if the episode finishes or times out
    first.

    `drive_cfg` must have use_visual_backdoor=False, use_backdoor_instruction=False.
    """
    assert not drive_cfg.use_visual_backdoor and not drive_cfg.use_backdoor_instruction

    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    action_queue = deque(maxlen=drive_cfg.num_open_loop_steps)
    max_steps = TASK_MAX_STEPS[drive_cfg.task_suite_name]

    sim, body_names, name_to_id = _get_sim_and_body_maps(env)
    table_z, target_body_id, target_body_ids = (None, None, [])
    if sim is not None:
        table_z, target_body_id, target_body_ids = _resolve_target_body(
            env, sim, body_names, name_to_id, task_description)

    dt_s = None
    try:
        dt_s = getattr(env, "control_timestep", None)
        if dt_s is None:
            control_freq = getattr(env, "control_freq", None)
            if control_freq:
                dt_s = 1.0 / float(control_freq)
        if dt_s is None and sim is not None:
            try:
                mj_dt = float(getattr(getattr(sim, "model", None).opt, "timestep", 0.0))
            except Exception:
                mj_dt = 0.0
            frame_skip = getattr(env, "frame_skip", 1)
            try:
                dt_s = mj_dt * float(frame_skip)
            except Exception:
                dt_s = mj_dt if mj_dt > 0 else None
        if dt_s is None:
            dt_s = 0.1
    except Exception:
        dt_s = 0.1

    elapsed_s = 0.0
    gripper_close_time = None
    time_condition_permanently_met = False
    was_closed = False

    t = 0
    while t < max_steps + drive_cfg.num_steps_wait:
        if t < drive_cfg.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(drive_cfg.model_family))
            t += 1
            continue

        backdoor_active = False
        if drive_cfg.conditional_backdoor_on_lift and sim is not None and table_z is not None:
            try:
                height_threshold = float(drive_cfg.backdoor_activation_height_m)
                time_threshold = float(drive_cfg.backdoor_activation_time_s)
                height_condition_met = False
                z_values = []
                if target_body_ids:
                    for tbid in target_body_ids:
                        tz = _get_body_z(sim, tbid)
                        if tz is not None:
                            z_values.append(tz)
                elif target_body_id is not None:
                    tz = _get_body_z(sim, target_body_id)
                    if tz is not None:
                        z_values.append(tz)
                if z_values:
                    height_condition_met = any((z - table_z) > height_threshold for z in z_values)

                if not time_condition_permanently_met and gripper_close_time is not None:
                    if (elapsed_s - gripper_close_time) >= time_threshold:
                        time_condition_permanently_met = True

                backdoor_active = height_condition_met and time_condition_permanently_met
            except Exception:
                backdoor_active = False

        if backdoor_active:
            return obs, t

        observation, _ = prepare_observation(drive_cfg, obs, resize_size, backdoor_active=False)

        if len(action_queue) == 0:
            actions = get_action(
                drive_cfg, model, observation, task_description,
                processor=processor, action_head=action_head,
                proprio_projector=proprio_projector, noisy_action_projector=None,
                use_film=drive_cfg.use_film,
            )
            action_queue.extend(actions)
        action = action_queue.popleft()
        action = process_action(action, drive_cfg.model_family)

        obs, reward, done, info = env.step(action.tolist())

        try:
            gripper_closed = float(action[-1]) > 0
            if not was_closed and gripper_closed and gripper_close_time is None:
                gripper_close_time = float(elapsed_s)
            was_closed = gripper_closed
        except Exception:
            pass
        elapsed_s += float(dt_s)

        if done:
            raise LiftNotReached("episode completed (done=True) before the lift/time condition was ever met")
        t += 1

    raise LiftNotReached(f"timed out after {t} steps without the lift/time condition being met")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--trigger-mode", required=True, choices=["vision", "language", "joint"],
                     help="which half of DropVLA's own MODE=vision/text/joint the 'triggered' "
                          "sample exercises -- verify this matches how the checkpoint you're "
                          "pointing at was actually trained/verified, don't assume vision.")
    ap.add_argument("--out", required=True, help="path to write the results JSON to.")
    ap.add_argument("--n-tasks", type=int, default=10, help="tasks per suite; LIBERO suites have 10.")
    ap.add_argument("--n-seeds", type=int, default=10,
                     help="curated init-state indices per task to attempt; one clean+triggered "
                          "pair is produced per index where the lift/time condition is reached.")
    ap.add_argument("--seed-base", type=int, default=0,
                     help="starting curated init-state index (0-based, matching task_suite.get_task_init_states).")
    ap.add_argument("--cover-wrist-lower-quarter", action="store_true",
                     help="matches DropVLA's cover_wrist_lower_quarter config (default off, "
                          "same default as run_libero_eval.py) -- verify per checkpoint before enabling.")
    ap.add_argument("--load-in-4bit", type=lambda s: s.lower() != "false", default=True,
                     help="matches run_eval.sh's paper protocol (--load_in_4bit True) by default.")
    ap.add_argument("--seed", type=int, default=42, help="torch/numpy seed, matching run_eval.sh's default.")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)

    base_cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        task_suite_name=args.task_suite_name,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        lora_rank=32,
        load_in_8bit=False,
        load_in_4bit=args.load_in_4bit,
        center_crop=False,
        num_open_loop_steps=8,
        num_steps_wait=NUM_STEPS_WAIT,
        env_img_res=256,
        use_backdoor_instruction=False,
        use_visual_backdoor=False,
        language_suffix="carefully",
        conditional_backdoor_on_lift=True,
        backdoor_activation_height_m=0.06,
        backdoor_activation_time_s=0.02,
        visual_backdoor_dot_x=10,
        visual_backdoor_dot_y=10,
        visual_backdoor_dot_radius=5,
        visual_backdoor_dot_alpha=255,
        visual_backdoor_dot_shape="circle",
        cover_wrist_lower_quarter=args.cover_wrist_lower_quarter,
        seed=args.seed,
    )

    print(f"[*] loading {args.checkpoint} (suite={args.task_suite_name}, trigger_mode={args.trigger_mode}, "
          f"load_in_4bit={args.load_in_4bit})")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(base_cfg)
    resize_size = get_image_resize_size(base_cfg)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    use_vision = args.trigger_mode in ("vision", "joint")
    use_language = args.trigger_mode in ("language", "joint")

    # Per-episode FTT scores, collected separately per camera.
    clean_scores = {"primary": [], "wrist": []}
    trigger_scores = {"primary": [], "wrist": []}

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        n_avail = init_states.shape[0]
        env, task_description = get_libero_env(task, base_cfg.model_family, resolution=base_cfg.env_img_res)
        try:
            for k in range(args.n_seeds):
                episode_idx = args.seed_base + k
                if episode_idx >= n_avail:
                    print(f"    [!] task={task_id}: only {n_avail} curated init states, "
                          f"skipping index {episode_idx}")
                    continue
                try:
                    obs, t = run_to_trigger_frame(
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

                primary_clean, wrist_clean = capture_text_to_image_attention(
                    model, processor, proprio_projector, base_cfg, clean_observation, clean_desc)
                primary_trig, wrist_trig = capture_text_to_image_attention(
                    model, processor, proprio_projector, base_cfg, trig_observation, trig_desc)

                clean_scores["primary"].append(compute_ftt(primary_clean))
                clean_scores["wrist"].append(compute_ftt(wrist_clean))
                trigger_scores["primary"].append(compute_ftt(primary_trig))
                trigger_scores["wrist"].append(compute_ftt(wrist_trig))

                print(f"    task={task_id} idx={episode_idx} activated@t={t}  captured clean+trigger, both cameras")
        finally:
            env.close()

    del model, processor, proprio_projector
    torch.cuda.empty_cache()

    results = {}
    for cam in ("primary", "wrist"):
        n_clean, n_trig = len(clean_scores[cam]), len(trigger_scores[cam])
        auroc = compute_auroc(clean_scores[cam], trigger_scores[cam])
        results[cam] = {"n_clean": n_clean, "n_trigger": n_trig, "auroc": auroc}
        print(f"[*] {cam}: n_clean={n_clean} n_trigger={n_trig} AUROC={auroc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
