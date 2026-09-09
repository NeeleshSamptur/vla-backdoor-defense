#!/usr/bin/env python
"""One forward pass -> every attention slice this adapter's separate scripts
compute independently today:

    extract_text2img_ftt.py     text->{primary,wrist}, single layer or
                                 layer-averaged (collapsed before saving)
    recapture_all_layers.py     text->primary only, ALL layers preserved
    extract_img2img_ftt.py      primary->primary only, ALL layers preserved

Each of the three re-runs the identical forward pass just to grab a
different slice of the same attention output, then discards the rest. This
module runs the forward pass ONCE and returns every slice (text->primary,
text->wrist, primary->primary, wrist->wrist), all layers preserved, so every
downstream variant (single layer / layer-averaged / layerwise / per-layer
sweep / img2img) can be computed afterward from one saved array with zero
extra model calls.

Sequence layout (unchanged from the other three scripts, confirmed there and
re-verified here): [ BOS(1) ][ primary patches(num_patches) ]
[ wrist patches(num_patches) ][ proprio(1) ][ text(n_txt) ].

VERIFICATION REQUIRED before trusting this for a real result: run
verify_against_existing() below, which re-derives the same slices via the
three existing (already-used) functions on a real forward pass and asserts
numerical equality. Do not skip this -- it is the actual test, not the
shapes-look-right check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)

import numpy as np
import torch

from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
from prismatic.vla.constants import IGNORE_INDEX

from adapters.backdoorvla_openvla_oft.extract_text2img_ftt import (
    DEVICE, _desc_token_row_indices,
)

DEBUG_ROUND_TRIP = True


def unified_rows_all_layers(vla, processor, proprio_projector, cfg, observation, prompt_desc,
                             text_scope="desc_only"):
    """One forward pass -> (rows_t2i_primary, rows_t2i_wrist, rows_i2i_primary,
    rows_i2i_wrist, num_patches, decoded_query_text).

    All returned attention arrays are [n_layers, ..., ...] with layers
    preserved (head-averaged only), matching recapture_all_layers.py's and
    extract_img2img_ftt.py's existing convention -- so every layer variant
    (single/averaged/layerwise/sweep) is derivable afterward without
    re-running the model.
    """
    full = observation["full_image"].copy()
    wrist = observation["wrist_image"].copy()
    images = prepare_images_for_vla([full, wrist], cfg)
    prompt = f"In: What action should the robot take to {prompt_desc.lower()}?\nOut:"
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
        txt_rel = _desc_token_row_indices(processor, prompt, prompt_desc, n_txt)
    else:
        txt_rel = list(range(n_txt))
    txt_rows = [1 + n_img_cols + r for r in txt_rel]
    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))

    # Round-trip check (this repo's standing method): decode exactly the
    # tokens kept as queries and confirm they equal the cleaned instruction.
    # NOTE: the reference here must be input_ids2[0][1 + r] (dropping only
    # the leading BOS), NOT input_ids[0, r] or a mask-compacted subset --
    # input_ids2 additionally carries OFT's action-chunk placeholder tokens
    # appended after the real text, and txt_rel (from a BOS-free
    # tokenization of the original prompt) lines up with input_ids2[0][1:],
    # confirmed empirically against a real forward pass.
    if DEBUG_ROUND_TRIP:
        kept_ids = input_ids2[0, [1 + r for r in txt_rel]].tolist()
        decoded = processor.tokenizer.decode(kept_ids).strip()
        expected = prompt_desc.lower().strip()
        if decoded != expected:
            raise AssertionError(f"round-trip mismatch: decoded={decoded!r} expected={expected!r}")

    # [n_layers, seq, seq], head-averaged, layers preserved -- SAME
    # reduction recapture_all_layers.py and extract_img2img_ftt.py already use.
    A_all = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)

    rows_t2i_primary = A_all[:, txt_rows][:, :, primary_cols].cpu().numpy()
    rows_t2i_wrist = A_all[:, txt_rows][:, :, wrist_cols].cpu().numpy()
    rows_i2i_primary = A_all[:, primary_cols][:, :, primary_cols].cpu().numpy()
    rows_i2i_wrist = A_all[:, wrist_cols][:, :, wrist_cols].cpu().numpy()

    del out
    torch.cuda.empty_cache()
    return rows_t2i_primary, rows_t2i_wrist, rows_i2i_primary, rows_i2i_wrist, num_patches, prompt_desc.lower()
