#!/usr/bin/env python
"""One attention_prefix(all_layers=True) call -> every slice the two existing
pi0-FAST scripts compute separately:

    extract_text2img_ftt.py   text->{primary,wrist}, one layer at a time
                               (attention_prefix(layer=k))
    extract_img2img_ftt.py    primary->primary only, all layers
                               (attention_prefix(all_layers=True))

Both are ONE forward pass each today; attention_prefix(all_layers=True)
already returns every layer's full [T, S] attention from a single pass (see
its docstring in pi0_fast.py), so text2img_ftt.py's per-layer looping is the
only redundant compute here -- one call already has everything needed for
every layer variant, plus wrist camera and image-to-image slices neither
existing script currently saves.

VERIFIED against both existing functions on 4 real observations (1 clean, 3
different trigger prompts) from results/pi0fast_TI4_obs -- bit-for-bit match
(max diff 0.0) on the overlapping slices, correct round-trip decode via
desc_token_rows on every prompt. See
/tmp/.../scratchpad/verify_unified_pi0fast.py for the check itself if it
needs to be re-run after any change here.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

PI0_ROOT = "/home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast"
DEFENSE_REPO = "/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense"
for _p in (DEFENSE_REPO, os.path.join(PI0_ROOT, "src")):
    sys.path.insert(0, _p)

import jax
import numpy as np

from openpi.models import model as _model

from extract_text2img_ftt import desc_token_rows


def unified_rows_all_layers(policy, model, sp, element, text_scope="desc_only"):
    """One forward pass -> (t2i_primary, t2i_wrist, i2i_primary, i2i_wrist,
    decoded_query_text), all [n_layers, ...] with layers preserved.

    element: the same dict extract_text2img_ftt.py/extract_img2img_ftt.py
    build from a saved observation .npz (observation/image,
    observation/wrist_image, observation/state, prompt).
    """
    description = element["prompt"]
    inputs = policy._input_transform(dict(element))  # noqa: SLF001
    batched = jax.tree.map(lambda x: jax.numpy.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(batched)

    attn_all, n_img, seq_mask = model.attention_prefix(observation, all_layers=True)
    attn_all = np.asarray(attn_all, dtype=np.float32)  # [depth, T, S]

    tok_ids = np.asarray(inputs["tokenized_prompt"])
    tok_mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
    real_ids = tok_ids[tok_mask]

    if text_scope == "desc_only":
        rel, decoded_full = desc_token_rows(sp, real_ids, description)
    else:
        from extract_text2img_ftt import token_char_spans
        _, decoded_full = token_char_spans(sp, real_ids)
        rel = list(range(len(real_ids)))
    txt_rows = [n_img + r for r in rel]

    per_cam = n_img // 3
    assert per_cam * 3 == n_img, f"expected 3 equal image blocks, got n_img={n_img}"

    t2i_primary = attn_all[:, txt_rows, 0:per_cam]
    t2i_wrist = attn_all[:, txt_rows, per_cam:2 * per_cam]
    i2i_primary = attn_all[:, 0:per_cam, 0:per_cam]
    i2i_wrist = attn_all[:, per_cam:2 * per_cam, per_cam:2 * per_cam]

    decoded_desc = sp.decode([int(real_ids[r]) for r in rel]).strip()
    return t2i_primary, t2i_wrist, i2i_primary, i2i_wrist, per_cam, decoded_desc
