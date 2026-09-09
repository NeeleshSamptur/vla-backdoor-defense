#!/usr/bin/env python
"""DropVLA extractor: LAST-LAYER text->image attention (desc_only, both cameras),
captured at the SAME physically-grounded frame this attack's own eval script
uses to decide when its trigger fires -- object lifted above the table by
`backdoor_activation_height_m`, sustained for `backdoor_activation_time_s`
after first gripper closure. Every piece of model-loading and lift-detection
logic below is IMPORTED from DropVLA's own `run_libero_eval.py`, not
reimplemented, so the scenes/thresholds/quantization this attack's own
TSR-L/success numbers were measured under are the ones used here too.

------------------------------------------------------------------------
0. There is only ONE eval script here -- do not go looking for a second one
------------------------------------------------------------------------
This project's BackdoorVLA-OFT adapter (adapters/backdoorvla_openvla_oft/)
was originally built on a wrong assumption: that "run the eval script" fully
described the attack, when AttackVLA's own `run_evaluate.sh` actually invokes
TWO different scripts back to back -- `eval_poison.py` for the poisoned/ASR
number and `run_libero_eval.py` for the clean-performance number -- with
different instruction-substitution rules between them. Conflating the two
produced a meaningless label until that was caught and fixed (see
adapters/backdoorvla_openvla_oft/extract_targeted_known_outcome_ftt.py's
module docstring for the full account).

DropVLA does NOT have that split. Confirmed directly from
`DropVLA/run_eval.sh` (the paper-protocol wrapper) and
`DropVLA/run_reeval_fixed_code.sh`: both call the SAME
`experiments/robot/libero/run_libero_eval.py` for every condition --
clean, vision-only, language-only, joint -- selected purely by CLI flags
(`--use_visual_backdoor`, `--use_backdoor_instruction`) driven by a `MODE`
shell variable. `task_suite_name` is also the SAME plain suite
(e.g. `libero_object`, never a "_poisoned" variant) across all four modes --
unlike BackdoorVLA, which evaluates its ASR number in a different,
purpose-built poisoned task suite. So there is no second script or dataset
to go find here; using `run_libero_eval.py`'s own functions for every part of
this extractor (model loading, env, observation prep, lift-detection) is
sufficient and correct for both the clean and the triggered condition alike.

------------------------------------------------------------------------
1. Why one closed-loop rollout yields BOTH a clean and a triggered sample
------------------------------------------------------------------------
CONFIRMED by reading `run_libero_eval.py:807-868` (`prepare_observation`'s
call site inside `run_episode`): whether the visual dot or the language
suffix is actually applied is gated on `backdoor_active`, which itself is
computed from real physics (`(object_z - table_z) > backdoor_activation_height_m`,
sustained `backdoor_activation_time_s` after first gripper closure) --
**not** from which condition/mode is configured. Until that condition first
turns True, a clean-driven rollout (both backdoor flags off, exactly what
`MODE=clean` uses) and a triggered-driven rollout are DRIVING THE SAME
POLICY ON THE IDENTICAL, UN-MODIFIED OBSERVATION -- they cannot diverge
before that instant. So this extractor drives ONE rollout per episode with
both backdoor flags off (clean driving, matching `MODE=clean` exactly),
using DropVLA's own lift-detection helpers (`_get_sim_and_body_maps`,
`_get_target_body_ids_from_env`, `_find_target_body_id_from_description`,
imported verbatim from `run_libero_eval.py`) to find the frame where
`backdoor_active` first turns True, then takes TWO forward passes on that
SAME frame: one through `prepare_observation` with the backdoor flags off
(clean sample) and one with them on per `--trigger-mode` (triggered sample,
still via `prepare_observation` -- the same function, same red-dot drawing
code, same conditional-suffix logic the real eval uses). This is a rollout-
based capture (option 1 of the two forks in PORTING_PROMPT_DROPVLA.md
section 0, the "faithful" option, not the cheaper scripted-frame option),
chosen because the whole point of this attack is that the trigger is gated
on a physical event the policy itself causes mid-task, not present in any
static/settled scene -- extracting from a no-op-settled frame would be
off-distribution for this model (see that doc's section 0 for the full
argument) and is exactly the failure mode this design avoids.

Because clean and triggered samples share one rollout up to the capture
frame, there is only ONE pairing convention here (unlike badvla_white_patch's
paired/disjoint choice) -- every episode index gives one clean+triggered pair
from the identical scene, differing only in the `prepare_observation` inputs
at that one frame.

------------------------------------------------------------------------
2. Trigger mode -- verify per checkpoint, do not assume
------------------------------------------------------------------------
`--trigger-mode {vision,language,joint}` controls which half of
`prepare_observation`'s conditional logic is exercised for the "triggered"
sample (vision = red dot only, language = suffix only, joint = both) --
mirroring DropVLA's own `MODE=vision/text/joint` in run_eval.sh. Per
PORTING_PROMPT_DROPVLA.md section 5, do NOT assume "vision" just because
it's the paper's headline number: confirm which mode each checkpoint was
actually TRAINED/verified under (check its own training command / the
matching entry in attack_model_paths.md) before trusting a result run under
the wrong mode.

------------------------------------------------------------------------
3. FTT forward-pass machinery -- same code as every other OFT adapter here
------------------------------------------------------------------------
`oft_text2img_rows` below is the same forward-pass reconstruction used by
adapters/badvla_white_patch/extract_text2img_ftt.py's `text2img_rows`
(reassemble the multimodal sequence with the model's own `_process_*`
helpers and call the language model directly with `output_attentions=True`,
since `predict_action`'s `generate()` never surfaces attentions) --
`detectors/ftt.py` (DO NOT MODIFY) and `detectors/schema.py`'s
`ExtractedSample` are unchanged from every other adapter. The only things
that differ from badvla_white_patch's version: DropVLA's own `get_vla` is
used for loading (4-bit-quantized by default, matching `run_eval.sh --load_in_4bit
True` exactly, vs badvla_white_patch's plain bf16 load), and the trigger
is DropVLA's red dot via `prepare_observation`/`add_red_dot_to_numpy_image`,
not badvla's white square. PORTING_PROMPT_DROPVLA.md section 2b already
confirmed (by diffing the actual checkpoint-local `modeling_prismatic.py`
against BadVLA's, and by loading a real DropVLA checkpoint) that this same
forward-pass pattern produces real, non-None, correctly-shaped attentions
for a DropVLA-OFT checkpoint -- re-run `--n-tasks 1 --n-seeds 1` as a smoke
test on a new checkpoint before trusting a full run, since that
verification was only done against the libero_object checkpoint.

Usage (DropVLA's own conda env):
    source /home/grads/nsamptur/vla_bkd_def/DropVLA/dropvla_env.sh
    cd /home/grads/nsamptur/vla_bkd_def/DropVLA

    python /home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/adapters/dropvla/extract_text2img_ftt.py \
        --checkpoint "$RUN_DIR/openvla-7b+libero_object_no_noops_v5p00carefully+...--seed42--paper" \
        --task-suite-name libero_object --trigger-mode vision \
        --out-dir ../vla-backdoor-defense/results/dropvla_extracted
"""

from __future__ import annotations

import argparse
import hashlib
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
    add_red_dot_to_numpy_image,  # noqa: F401 (documents the trigger this attack uses; applied via prepare_observation)
    initialize_model,
    prepare_observation,
    process_action,
)
from experiments.robot.openvla_utils import DEVICE, normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere
from prismatic.vla.constants import IGNORE_INDEX

import experiments.robot.robot_utils as _robot_utils
import experiments.robot.openvla_utils as _openvla_utils


def _get_vla_eager(cfg):
    """Same as DropVLA's own openvla_utils.get_vla, with ONE change:
    attn_implementation="eager" instead of the unset default (get_vla's own
    flash_attention_2 line is commented out, so it falls through to
    transformers' default, which resolves to "sdpa" for a Llama backbone).

    This matters because oft_text2img_rows below calls
    vla.language_model(..., output_attentions=True, ...) directly. Under
    transformers 4.40.1, requesting output_attentions=True on an
    attn_implementation="sdpa" model forces a fallback to the manual/eager
    attention body -- but ONLY if the causal mask passed down is non-None.
    LlamaModel._update_causal_mask has an SDPA-specific optimization
    (_ignore_causal_mask_sdpa) that returns an explicit None mask whenever
    attention_mask is all-1s and query_length==key_value_length (exactly
    this script's single forward pass, no padding, no KV cache setup),
    relying on SDPA's fused kernel to enforce causality internally via its
    own is_causal=True argument. The eager fallback body never receives that
    is_causal flag and, given attention_mask=None, applies NO masking at
    all -- so out.attentions comes back fully BIDIRECTIONAL, not causal.
    Confirmed empirically on GoBA (same prismatic/Llama family, identical
    load pattern): image-patch query rows had substantial nonzero mass on
    later-patch and image->text columns that must be exactly zero under real
    causal masking; switching to eager fixed it (verified: those blocks
    became exactly 0 everywhere, row sums still ~1.0).

    attn_implementation="eager" always receives and applies the real 4D
    causal mask (that mask-skipping optimization is SDPA-specific), so
    output_attentions=True under eager returns the SAME attention the model
    actually used to compute its outputs -- which is also what DropVLA's own
    real rollout actions are computed from below (predict via get_action /
    process_action never requests output_attentions, so it would have run
    correctly under either implementation; forcing eager here just makes the
    two forward passes -- action generation and attention capture -- use the
    identical, correct computation).

    Patched in as `experiments.robot.robot_utils.get_vla` (the name actually
    invoked by robot_utils.get_model, which holds its own `from ... import
    get_vla` binding -- patching openvla_utils.get_vla alone would not reach
    that call site) rather than editing DropVLA's own repo files, per this
    project's policy of touching attack code only from inside the adapter.
    """
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    print("[*] loading VLA with EAGER attention (patched -- SDPA+output_attentions "
          "silently returns bidirectional, not causal, attention; see _get_vla_eager docstring)")

    if not _openvla_utils.model_is_on_hf_hub(cfg.pretrained_checkpoint):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        _openvla_utils.update_auto_map(cfg.pretrained_checkpoint)
        _openvla_utils.check_model_logic_mismatch(cfg.pretrained_checkpoint)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="eager",  # <-- the only deviation from get_vla()
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


_robot_utils.get_vla = _get_vla_eager

from detectors.schema import ExtractedSample  # from the defense repo, added to sys.path above

NUM_STEPS_WAIT = 10
VALID_SUITES = (
    TaskSuite.LIBERO_SPATIAL.value,
    TaskSuite.LIBERO_OBJECT.value,
    TaskSuite.LIBERO_GOAL.value,
    TaskSuite.LIBERO_10.value,
)


# ---------------------------------------------------------------------------
# FTT forward pass -- same pattern as adapters/badvla_white_patch/extract_text2img_ftt.py
# ---------------------------------------------------------------------------

def _desc_token_row_indices(processor, prompt: str, desc: str, n_txt: int) -> list[int]:
    """Identical in intent to badvla_white_patch's helper of the same name
    (kept as a local copy so this adapter stays self-contained in DropVLA's
    own conda env): map the task-description's character span in the real
    prompt to prompt-token row indices, via the fast tokenizer's own offset
    mapping rather than guessing from template token counts.
    """
    tok = processor.tokenizer
    if not tok.is_fast:
        raise RuntimeError(
            "text_scope='desc_only' needs a fast tokenizer for character "
            "offset mapping; got a slow tokenizer.")

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


def oft_text2img_rows(vla, processor, proprio_projector, cfg, observation, desc,
                       text_scope="desc_only", layer=-1, all_layers=False):
    """Forward pass with output_attentions=True on an already-prepared
    (possibly triggered) `prepare_observation(...)` output. Same internal
    reconstruction as badvla_white_patch's `text2img_rows`: reassemble the
    fused [BOS][primary patches][wrist patches][proprio][text] sequence with
    the model's own `_process_*` helpers (verified compatible for DropVLA-OFT
    checkpoints in PORTING_PROMPT_DROPVLA.md section 2b) and call the language
    model directly, since `predict_action`'s `generate()` path never surfaces
    attentions.
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

    if text_scope == "desc_only":
        txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    else:
        txt_rel = list(range(n_txt))
    txt_rows = [1 + n_img_cols + r for r in txt_rel]
    A = out.attentions[layer][0].float().mean(0)
    assert A.shape[-1] == 1 + n_img_cols + input_ids2.shape[-1] - 1, (
        f"token layout drift: T={A.shape[-1]} n_img_cols={n_img_cols} "
        f"len(input_ids2)={input_ids2.shape[-1]}")
    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))
    rows_primary = A[txt_rows][:, primary_cols].cpu().numpy()
    rows_wrist = A[txt_rows][:, wrist_cols].cpu().numpy()

    rows_primary_all_layers = None
    if all_layers:
        # out.attentions already holds every layer from this SAME forward
        # pass (output_attentions=True computes them all regardless of which
        # single `layer` index is used above) -- no extra model call needed,
        # just keep them instead of discarding all but one.
        A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, T, T]
        rows_primary_all_layers = A_stack[:, txt_rows][:, :, primary_cols].cpu().numpy()  # [L, n_txt, num_patches]

    del out
    torch.cuda.empty_cache()
    return rows_primary, rows_wrist, num_patches, rows_primary_all_layers


# ---------------------------------------------------------------------------
# Rollout / capture
# ---------------------------------------------------------------------------

class LiftNotReached(Exception):
    pass


def _resolve_target_body(env, sim, body_names, name_to_id, task_description):
    """Verbatim port of run_libero_eval.run_episode's target/table resolution
    (lines 741-772): prefer env.obj_of_interest, fall back to LIBERO-Spatial's
    fixed keyword vocabulary, then description-derived body-name matching;
    fall back to the object's own resting height if no "table" body exists.
    """
    table_z = None
    target_body_id = None
    if sim is None:
        return table_z, target_body_id

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


def run_episode_capture(drive_cfg, env, task_description, model, action_head, proprio_projector,
                         processor, resize_size, initial_state):
    """Drive one closed-loop rollout with BOTH backdoor flags off (matching
    MODE=clean's driving exactly -- see module docstring section 1) until
    DropVLA's own lift/time condition first turns True, then return the raw
    `obs` at that instant. Raises LiftNotReached if the episode finishes
    (done) or times out before that ever happens.

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
            return obs, t, target_body_id, target_body_ids, table_z

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


def capture_first_frame(drive_cfg, env, initial_state):
    """Cheap comparison baseline: reset, run the standard num_steps_wait no-op
    settle (same settle DropVLA's own run_episode does before ever querying
    the policy), and capture there -- t=num_steps_wait, no rollout, no lift
    detection. This is the kind of frame every OTHER adapter in this repo
    uses (a settled scene, trigger forced on regardless of physical state),
    kept here ONLY as an explicit diagnostic against run_episode_capture's
    physically-gated lift frame -- PORTING_PROMPT_DROPVLA.md section 0 is
    explicit that this frame is off-distribution for DropVLA specifically
    (the model was never trained/evaluated with the dot present this early,
    since the real eval only ever draws it once the object is lifted), so a
    result from this mode is NOT a substitute for the lift-frame numbers,
    only a check on whether the weak lift-frame signal is a frame-choice
    artifact or something more general about this trigger.
    """
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    for _ in range(drive_cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(drive_cfg.model_family))
    return obs, drive_cfg.num_steps_wait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--trigger-mode", required=True, choices=["vision", "language", "joint"],
                     help="which half of DropVLA's own MODE=vision/text/joint the "
                          "'triggered' sample exercises -- verify this matches how the "
                          "checkpoint you're pointing at was actually trained/verified "
                          "(PORTING_PROMPT_DROPVLA.md section 5), don't assume vision.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10, help="tasks per suite; LIBERO suites have 10.")
    ap.add_argument("--n-seeds", type=int, default=10,
                     help="curated init-state indices per task to attempt; one clean+triggered "
                          "pair is produced per index where the lift/time condition is reached.")
    ap.add_argument("--seed-base", type=int, default=0,
                     help="starting curated init-state index (0-based, matching task_suite.get_task_init_states).")
    ap.add_argument("--layer", type=int, default=-1, help="LLM layer index for attention (-1 = last)")
    ap.add_argument("--all-layers", action="store_true",
                     help="also save every LLM layer's attention (attn_text_image_layers), "
                          "free of extra compute since output_attentions=True already "
                          "computes every layer in one forward pass -- needed for "
                          "detectors/ftt.py's ftt_score_layerwise family or a true "
                          "per-layer AUROC breakdown (does not change attn_text_image, "
                          "which stays the single --layer selection).")
    ap.add_argument("--text-scope", choices=["desc_only", "all"], default="desc_only")
    ap.add_argument("--cover-wrist-lower-quarter", action="store_true",
                     help="matches DropVLA's cover_wrist_lower_quarter config (default off, "
                          "same default as run_libero_eval.py) -- verify per checkpoint before enabling.")
    ap.add_argument("--load-in-4bit", type=lambda s: s.lower() != "false", default=True,
                     help="matches run_eval.sh's paper protocol (--load_in_4bit True) by default.")
    ap.add_argument("--seed", type=int, default=42, help="torch/numpy seed, matching run_eval.sh's default.")
    ap.add_argument("--role", choices=["attack", "clean_baseline"], default="attack",
                     help="which CHECKPOINT this is, constant for the whole run (matches "
                          "badvla_white_patch's convention) -- 'attack' for the actual "
                          "backdoored checkpoint (default; label 0/1 within this group is the "
                          "clean/triggered SCENE, which is what run_detector.py's AUROC needs "
                          "both classes of). Pass 'clean_baseline' only when pointing --checkpoint "
                          "at a genuinely non-backdoored control model run through this same "
                          "pipeline, to check the detector isn't just reacting to the red-dot "
                          "overlay itself. Do not conflate this with --trigger-mode.")
    ap.add_argument("--capture-frame", choices=["lift", "first"], default="lift",
                     help="'lift' (default): the physically-gated frame where DropVLA's own "
                          "lift/time condition first turns True -- the faithful option, see "
                          "module docstring section 1. 'first': a cheap diagnostic-only "
                          "comparison that settles the scene (num_steps_wait no-ops, same as "
                          "every OTHER adapter in this repo) and forces the trigger on there "
                          "regardless of physical state -- off-distribution for DropVLA per "
                          "PORTING_PROMPT_DROPVLA.md section 0, use only to check whether a weak "
                          "'lift' result is frame-choice-specific, never as the reported number.")
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

    _raw_tag = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    _digest = hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]
    ckpt_tag = f"{_raw_tag[:40]}_{_digest}"
    out_dir = Path(args.out_dir)
    # capture_frame changes what the frame's attention MEANS (physically-gated
    # lift instant vs. an off-distribution forced-early frame -- see
    # --capture-frame help), so it must never silently mix with the other
    # mode's samples: fold it into the filename tag (a different directory
    # listing) as well as into run_detector.py's provenance check below.
    file_tag = f"{ckpt_tag}__cf-{args.capture_frame}"

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
                    print(f"    [!] task={task_id}: only {n_avail} curated init states, "
                          f"skipping index {episode_idx}")
                    continue

                if args.capture_frame == "lift":
                    try:
                        obs, t, target_body_id, target_body_ids, table_z = run_episode_capture(
                            base_cfg, env, task_description, model, action_head, proprio_projector,
                            processor, resize_size, init_states[episode_idx])
                    except LiftNotReached as e:
                        print(f"    task={task_id} idx={episode_idx}: [skip] {e}")
                        continue
                else:
                    obs, t = capture_first_frame(base_cfg, env, init_states[episode_idx])

                clean_cfg = replace(base_cfg, use_visual_backdoor=False, use_backdoor_instruction=False)
                trig_cfg = replace(base_cfg, use_visual_backdoor=use_vision, use_backdoor_instruction=use_language)

                clean_observation, _ = prepare_observation(clean_cfg, obs, resize_size, backdoor_active=False)
                trig_observation, _ = prepare_observation(trig_cfg, obs, resize_size, backdoor_active=True)

                clean_desc = task_description
                trig_desc = f"{task_description.strip()} {trig_cfg.language_suffix.strip()}".strip() \
                    if use_language else task_description

                rows_primary_clean, rows_wrist_clean, num_patches, layers_clean = oft_text2img_rows(
                    model, processor, proprio_projector, base_cfg, clean_observation, clean_desc,
                    text_scope=args.text_scope, layer=args.layer, all_layers=args.all_layers)
                rows_primary_trig, rows_wrist_trig, _, layers_trig = oft_text2img_rows(
                    model, processor, proprio_projector, base_cfg, trig_observation, trig_desc,
                    text_scope=args.text_scope, layer=args.layer, all_layers=args.all_layers)

                common_extra = {
                    "task_suite_name": args.task_suite_name,
                    "trigger_mode": args.trigger_mode,
                    "task_description": task_description,
                    "ftt_cameras": "primary_and_wrist",
                    "text_scope": args.text_scope,
                    "init_state_index": episode_idx,
                    "activation_frame_t": t,
                    "backdoor_activation_height_m": base_cfg.backdoor_activation_height_m,
                    "backdoor_activation_time_s": base_cfg.backdoor_activation_time_s,
                    "cover_wrist_lower_quarter": args.cover_wrist_lower_quarter,
                    "capture_frame": args.capture_frame,
                }

                for cond, trig, rows_primary, rows_wrist, layers, desc_used in (
                    ("clean", False, rows_primary_clean, rows_wrist_clean, layers_clean, clean_desc),
                    ("trigger", True, rows_primary_trig, rows_wrist_trig, layers_trig, trig_desc),
                ):
                    # attn_text_image stays the single named `layer` (unchanged
                    # meaning / backward compatible with every existing consumer);
                    # attn_text_image_layers additionally carries every layer
                    # un-collapsed, for detectors/ftt.py's ftt_score_layerwise
                    # family and for computing a true per-layer AUROC breakdown.
                    sample = ExtractedSample(
                        attn_text_image=rows_primary,
                        label=int(trig),
                        attack="dropvla",
                        checkpoint=args.checkpoint,
                        trigger_type=f"red_dot_{args.trigger_mode}" if trig else "none",
                        task_id=task_id, seed=episode_idx, layer=args.layer,
                        n_cameras=2, patches_per_camera=num_patches,
                        episode_id=f"{args.task_suite_name}__t{task_id}__s{episode_idx}__{cond}",
                        frame_idx=t,
                        attn_text_image_wrist=rows_wrist,
                        attn_text_image_layers=layers,
                        # layers_averaged intentionally left at its None default: per
                        # schema.py, that field means "attn_text_image IS the mean over
                        # this many layers" -- not true here, attn_text_image is still
                        # the single named `layer` (default -1 = last), and
                        # attn_text_image_layers is only an un-collapsed companion.
                        extra={**common_extra, "role": args.role,
                               "n_query_tokens": int(rows_primary.shape[0]),
                               "language_instruction_used": desc_used},
                    )
                    sample.save(str(out_dir / f"{file_tag}__{args.task_suite_name}__t{task_id}"
                                              f"__s{episode_idx}__{cond}.npz"))

                print(f"    task={task_id} idx={episode_idx} activated@t={t} "
                      f"primary={rows_primary_clean.shape} wrist={rows_wrist_clean.shape}")
        finally:
            env.close()

    del model, processor, proprio_projector
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
