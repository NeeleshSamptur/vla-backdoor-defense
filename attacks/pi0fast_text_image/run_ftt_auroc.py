#!/usr/bin/env python
"""Pi0-FAST / BackdoorVLA Text-Image (TI4) attack -- step 2 of 2: one model
forward pass per collected episode -> desc_only text-to-image attention ->
FTT -> AUROC.

Reads the `.npz` observation files written by collect_observations.py (see
that file's docstring for why collection is a separate process -- the JAX
venv this script needs has no robosuite, and the robosuite env has no JAX).
Everything past that hand-off runs in one process: load checkpoint, forward
pass, FTT (attacks.common.compute_ftt), AUROC (attacks.common.compute_auroc).

Model/checkpoint loading matches AttackVLA/Pi0-Fast's own inference path:
  - `policy_config.create_trained_policy()` builds the exact input-transform
    stack the eval server itself uses (LiberoInputs -> Normalize ->
    TokenizeFASTInputs, per src/openpi/policies/policy_config.py). It also
    loads its own copy of the model but exposes only a jitted
    sample_actions, so the params are loaded a second time, the same way
    `create_trained_policy` loads them internally, to get an object whose
    `attention_prefix()` can be called directly.
  - norm_stats are read from whatever `norm_stats.json` actually ships under
    `<checkpoint>/assets/`, rather than assumed from the training config's
    asset id -- `_normalize.load(dir)` is exactly what
    `checkpoints.load_norm_stats(assets_dir, asset_id)` calls internally
    (src/openpi/training/checkpoints.py), so this is the same lookup, just
    self-discovering the asset id instead of requiring the config to name it
    (this fork's config.py has no entry for a clean-baseline checkpoint,
    which ships different assets than the attack checkpoint).
  - the prompt-cleaning applied when locating description tokens
    (`.lower().strip().replace("_", " ")`) is copied verbatim from
    `FASTTokenizer.tokenize()` (src/openpi/models/tokenizer.py) -- not a
    guess at what the tokenizer does, the literal same expression.

Sequence layout (confirmed on a real forward pass, and by
`libero_policy.py`'s `mask_padding = (model_type == ModelType.PI0)`, which is
False for PI0_FAST -- meaning `image_masks["right_wrist_0_rgb"]` is always
True, so the dummy third camera block cannot be found via the mask):

    [ base_0_rgb 256 ][ left_wrist_0_rgb 256 ][ right_wrist_0_rgb 256 ][ text ]

768 image tokens then the padded prompt; there is no BOS-then-patches split
like OpenVLA's. The third (right_wrist) block is an all-zero dummy LIBERO
never fills and is excluded by position, not by mask. Proprio is not a
token: FASTTokenizer discretizes it into 256 bins and writes it into the
prompt as literal digits between "State: " and ";" -- excluded from the
query set as ordinary text, along with the "Task: "/"State: " template
literals, by `desc_token_rows()` locating only the instruction's own
character span. The "~*magic*~ " trigger prefix lives inside that same span
(the model is given `description = "~*magic*~ " + instruction`, and
`desc_token_rows` searches for the full `description` string), so it is
always part of the query set on triggered episodes and never present
otherwise -- there is no code path that could drop it.

`model.attention_prefix()` (src/openpi/models/pi0_fast.py) is this project's
addition to openpi, added to read out attention without changing what the
model computes. Its own docstring documents a real bug it fixed: an earlier
version deliberately skipped `left_to_right_align()` (the same
padding-to-front alignment `sample_actions()` applies before its forward
pass), on an unverified theory that it couldn't matter since RoPE positions
come from `cumsum(mask)-1` and padding is masked out either way. A/B testing
against `sample_actions()`'s real path (same checkpoint and observation,
aligned attention rolled back to plain order for comparison) showed
real-token attention differing by up to ~0.25 in probability mass --
concentrated in image-patch query rows. The fix actually runs the same
alignment `sample_actions()` runs, then undoes it on the way out (rolls
attention/mask back to plain `embed_inputs()` order) so callers still get
results in the index space they expect, with a same-shape mask assertion
guarding the roll bookkeeping. Softmax here is `jax.nn.softmax` on an
explicit masked-logits tensor (src/openpi/models/gemma_fast.py), not a fused
SDPA kernel, so this codebase has no exposure to the SDPA-plus-
output_attentions bidirectional-leak bug that hit a PyTorch-based adapter in
this repo -- there is nothing to fix or verify on that front here.

Usage (pi0-FAST's own venv, NOT the robosuite conda env):
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast
    CUDA_VISIBLE_DEVICES=<n> XDG_CACHE_HOME=$PWD/cache OPENPI_DATA_HOME=$PWD/cache \\
    .venv/bin/python <this file> \\
        --checkpoint checkpoints/pi0_fast_libero_object_TI_4/PiFast_Text_Image_Attack_object_4_5000/5000 \\
        --config-name pi0_fast_libero_object_TI_4 \\
        --obs-dir <collect_observations.py's --out-dir> \\
        --out ../vla-backdoor-defense/results/pi0fast_text_image_ftt_auroc.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

PI0_ROOT = pathlib.Path("/home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast")
DEFENSE_REPO = pathlib.Path(__file__).resolve().parents[2]
for _p in (DEFENSE_REPO, PI0_ROOT / "src"):
    sys.path.insert(0, str(_p))

import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

from openpi.models import model as _model
from openpi.shared import download, normalize as _normalize
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from attacks.common import compute_auroc, compute_ftt  # noqa: E402


def load_paligemma_tokenizer():
    """The same sentencepiece model FASTTokenizer itself uses."""
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def token_char_spans(sp, ids):
    """Character span of every token, rebuilt from the REAL token ids.

    sentencepiece exposes no return_offsets_mapping, so spans are
    reconstructed by walking the pieces of the actual ids the model was
    given -- never by tokenizing template fragments in isolation and using
    their counts as boundaries. That shortcut is unsound for sentencepiece
    (whether a leading space merges into the following word depends on the
    word) and already caused a real bug in this repo on a different adapter,
    silently dropping each description's leading verb.

    Returns (spans, reconstructed_text) where spans index into the text.
    """
    spans, pos, out = [], 0, []
    for tid in ids:
        piece = sp.id_to_piece(int(tid))
        if piece.startswith("<") and piece.endswith(">"):  # <bos>/<eos>/<pad>
            spans.append((pos, pos))
            continue
        text = piece.replace("▁", " ")  # sentencepiece's "▁" = leading space
        if pos == 0 and text.startswith(" "):
            text = text[1:]
        spans.append((pos, pos + len(text)))
        out.append(text)
        pos += len(text)
    return spans, "".join(out)


def desc_token_rows(sp, ids, description):
    """Indices into `ids` of the tokens covering the instruction (trigger
    prefix included when present -- see module docstring).

    A token is kept when its character span OVERLAPS the description's, so a
    token straddling the boundary (a leading-space token fused with the
    first word) is kept rather than dropped.
    """
    spans, text = token_char_spans(sp, ids)
    needle = description.lower().strip().replace("_", " ")  # FASTTokenizer's own cleaning
    start = text.find(needle)
    if start == -1:
        raise ValueError(f"description {needle!r} not found in decoded prompt {text!r}")
    end = start + len(needle)
    rows = [i for i, (s, e) in enumerate(spans) if s < end and e > start]
    assert rows, f"no tokens overlap the description span in {text!r}"
    return rows, text


def load_model(checkpoint: pathlib.Path, config_name: str):
    """Loads the checkpoint the same way AttackVLA's own eval server does --
    see module docstring for how each piece maps back to policy_config.py /
    checkpoints.py."""
    train_config = _config.get_config(config_name)
    norm_stats = None
    found = sorted((checkpoint / "assets").rglob("norm_stats.json"))
    if len(found) == 1:
        norm_stats = _normalize.load(found[0].parent)
        print(f"[*] norm stats asset_id={found[0].parent.relative_to(checkpoint / 'assets')!r}")
    elif len(found) > 1:
        raise RuntimeError(f"ambiguous norm stats under {checkpoint / 'assets'}: {found}")
    policy = _policy_config.create_trained_policy(train_config, checkpoint, norm_stats=norm_stats)
    model = train_config.model.load(_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    return policy, model


def capture_text_to_image_attention(policy, model, sp, element, description):
    """One forward pass; returns desc_only text->image attention for the
    primary camera, shape [n_layers, n_desc_tokens, per_cam], head-averaged.
    """
    inputs = policy._input_transform(dict(element))  # noqa: SLF001
    batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(batched)

    # all_layers=True returns every transformer layer's attention from this
    # ONE forward pass (head-averaged; keep_heads defaults to False and is
    # bit-identical to every other caller of attention_prefix), instead of
    # the single --layer this project's original adapter selected.
    attn_all, n_img, _seq_mask = model.attention_prefix(observation, all_layers=True)
    attn_all = np.asarray(attn_all, dtype=np.float32)  # [L, T, S]

    tok_ids = np.asarray(inputs["tokenized_prompt"])
    tok_mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
    real_ids = tok_ids[tok_mask]

    rel, _decoded = desc_token_rows(sp, real_ids, description)
    txt_rows = [n_img + r for r in rel]  # text tokens follow the image block

    per_cam = n_img // 3  # base | left_wrist | right_wrist(dummy), 256 each
    assert per_cam * 3 == n_img, f"expected 3 equal image blocks, got n_img={n_img}"

    # attn_all is [L, T, S]; index the query (text) axis and key (primary
    # image) axis separately so the leading layer axis survives intact.
    return attn_all[:, txt_rows][:, :, 0:per_cam]  # [L, n_desc, per_cam]


def run_episodes(policy, model, sp, obs_dir: pathlib.Path):
    files = sorted(obs_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no observations under {obs_dir}")
    clean_files = [f for f in files if "__clean.npz" in f.name]
    trig_files = [f for f in files if "__trigger.npz" in f.name]
    print(f"[*] {len(files)} observations from {obs_dir} "
          f"({len(clean_files)} clean, {len(trig_files)} trigger)")

    clean_scores, trigger_scores = [], []
    t_start = time.time()
    for i, f in enumerate(sorted(clean_files + trig_files)):
        d = np.load(f, allow_pickle=False)
        description = str(d["prompt"])
        element = {
            "observation/image": d["image"],
            "observation/wrist_image": d["wrist_image"],
            "observation/state": d["state"],
            "prompt": description,
        }
        attn_desc_primary = capture_text_to_image_attention(policy, model, sp, element, description)
        score = compute_ftt(attn_desc_primary)  # averages over the leading (layer) axis

        cond = str(d["condition"])
        (clean_scores if cond == "clean" else trigger_scores).append(score)
        elapsed = time.time() - t_start
        print(f"    [{i + 1}/{len(files)}] t={int(d['task_id'])} s={int(d['init_state_index'])} "
              f"{cond:8s} ftt={score:.5f} elapsed={elapsed:.1f}s", flush=True)

    return clean_scores, trigger_scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="pi0_fast_libero_object_TI_4")
    ap.add_argument("--obs-dir", required=True, help="output of collect_observations.py")
    ap.add_argument("--out", required=True, help="path to write the scores/AUROC JSON")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.checkpoint).resolve()
    obs_dir = pathlib.Path(args.obs_dir)

    print(f"[*] loading {ckpt}\n[*] config={args.config_name}")
    policy, model = load_model(ckpt, args.config_name)
    sp = load_paligemma_tokenizer()
    try:
        n_layers = int(model.PaliGemma.llm.configs[0].depth)
        print(f"[*] {n_layers} transformer layers; scoring desc_only text2img FTT over all of them")
    except AttributeError:
        pass

    clean_scores, trigger_scores = run_episodes(policy, model, sp, obs_dir)
    auroc = compute_auroc(clean_scores, trigger_scores)
    n_clean, n_trigger = len(clean_scores), len(trigger_scores)
    print(f"\n[*] n_clean={n_clean} n_trigger={n_trigger} AUROC={auroc:.4f}")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "attack": "pi0fast_text_image",
            "checkpoint": str(ckpt),
            "config_name": args.config_name,
            "n_clean": n_clean,
            "n_trigger": n_trigger,
            "clean_mean": float(np.mean(clean_scores)) if clean_scores else float("nan"),
            "trigger_mean": float(np.mean(trigger_scores)) if trigger_scores else float("nan"),
            "auroc": auroc,
            "clean_scores": clean_scores,
            "trigger_scores": trigger_scores,
        }, fh, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
