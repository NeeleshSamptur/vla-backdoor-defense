#!/usr/bin/env python
"""One forward pass -> every slice extract_text2img_ftt.py (text->{primary,
wrist}, single layer only) and extract_img2img_ftt.py (primary->primary
only, all layers) compute separately today, each via its own forward pass.

Same architecture family as adapters/backdoorvla_openvla_oft (OpenVLA-7B-OFT
recipe), so the sequence layout and _prepare_input_for_action_prediction /
_process_action_masks / _build_multimodal_attention machinery is identical.
Confirmed empirically for that adapter (see its own
extract_unified_all_layers_ftt.py) that the real text block in the fused
sequence is input_ids2[0][1:] -- i.e. txt_rows = [1 + n_img_cols + r for r in
txt_rel] is CORRECT as the existing code already has it; the round-trip
check here uses that same, verified reference.

VERIFY before trusting this for a result: run verify_unified_badvla.py
against real forward passes with different prompts/trigger conditions and
confirm bit-for-bit match with the two existing functions, same as was done
for backdoorvla_openvla_oft and pi0fast_backdoorvla.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)

import torch

from experiments.robot.libero.run_libero_eval import add_trigger_img
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
from prismatic.vla.constants import IGNORE_INDEX

from extract_text2img_ftt import DEVICE, _desc_token_row_indices

DEBUG_ROUND_TRIP = True


def unified_rows_all_layers(vla, processor, proprio_projector, cfg, observation, desc,
                             trigger, trigger_size, trigger_cameras="both", text_scope="desc_only"):
    """One forward pass -> (t2i_primary, t2i_wrist, i2i_primary, i2i_wrist,
    num_patches, decoded_query_text), all [n_layers, ...]."""
    full = observation["full_image"].copy()
    wrist = observation["wrist_image"].copy()
    if trigger:
        full = add_trigger_img(full, trigger_size=trigger_size, trigger_position="center", trigger_color=255)
        if trigger_cameras == "both":
            wrist = add_trigger_img(wrist, trigger_size=trigger_size, trigger_position="center", trigger_color=255)
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
    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))

    if DEBUG_ROUND_TRIP:
        kept_ids = input_ids2[0, [1 + r for r in txt_rel]].tolist()
        decoded = processor.tokenizer.decode(kept_ids).strip()
        expected = desc.lower().strip()
        if decoded != expected:
            raise AssertionError(f"round-trip mismatch: decoded={decoded!r} expected={expected!r}")

    A_all = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [n_layers, seq, seq]
    rows_t2i_primary = A_all[:, txt_rows][:, :, primary_cols].cpu().numpy()
    rows_t2i_wrist = A_all[:, txt_rows][:, :, wrist_cols].cpu().numpy()
    rows_i2i_primary = A_all[:, primary_cols][:, :, primary_cols].cpu().numpy()
    rows_i2i_wrist = A_all[:, wrist_cols][:, :, wrist_cols].cpu().numpy()

    del out
    torch.cuda.empty_cache()
    return rows_t2i_primary, rows_t2i_wrist, rows_i2i_primary, rows_i2i_wrist, num_patches, desc.lower()
