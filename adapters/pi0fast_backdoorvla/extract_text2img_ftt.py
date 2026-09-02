#!/usr/bin/env python
"""Phase 2 of the pi0-FAST adapter: attention -> ExtractedSample.

Reads the observations collect_observations.py saved (see that file for why
extraction is split in two: the robosuite env and the JAX env are disjoint),
runs pi0-FAST's prefix forward pass, and slices the text-to-image attention the
FTT detector consumes.

Targets AttackVLA's BackdoorVLA "Text-Image" attack on libero_object -- the only
pi0-FAST checkpoint with a verified eval log in attack_model_paths.md. The
trigger is bi-modal: a popcorn container in the scene (poisoned BDDL) plus the
literal string "~*magic*~ " prepended to the instruction. Per the project
decision the trigger text is run as-is and ITS TOKENS ARE PART OF THE QUERY SET,
matching T2IShield, where the trigger is likewise one of the prompt tokens FTT
is computed over. Query counts therefore differ between conditions; every sample
records n_query_tokens.

Sequence layout, confirmed on a real forward pass:

    [ base_0_rgb 256 ][ left_wrist_0_rgb 256 ][ right_wrist_0_rgb 256 ][ text 180 ]

768 image tokens then the padded prompt. There is no BOS-then-patches split like
OpenVLA's. The third image block is a zeros dummy LIBERO never fills; for
pi0-FAST its image_mask is True (mask_padding is only set for ModelType.PI0), so
it cannot be found via the mask and is excluded by position.

Proprio is not a token: FASTTokenizer discretizes it into 256 bins and writes it
into the prompt as literal digits between "State: " and ";". It is excluded as a
span of ordinary text tokens, along with the "Task: " template literals.

Usage (pi0-FAST's own venv):
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast
    CUDA_VISIBLE_DEVICES=1 XDG_CACHE_HOME=$PWD/cache OPENPI_DATA_HOME=$PWD/cache \\
    .venv/bin/python .../extract_text2img_ftt.py \\
        --checkpoint checkpoints/pi0_fast_libero_object_TI_4/PiFast_Text_Image_Attack_object_4_5000/5000 \\
        --obs-dir <phase-1 output> --out-dir <npz output>
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

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
from openpi.shared import normalize as _normalize
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

from detectors.schema import ExtractedSample  # noqa: E402

CLEAN_SUITE = "libero_object"


def load_paligemma_tokenizer():
    """The same sentencepiece model FASTTokenizer itself uses."""
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def token_char_spans(sp, ids):
    """Character span of every token, rebuilt from the REAL token ids.

    sentencepiece exposes no return_offsets_mapping, so spans are reconstructed
    by walking the pieces of the actual ids the model was given -- never by
    tokenizing template fragments in isolation and using their counts as
    boundaries. That shortcut is unsound for sentencepiece (whether a leading
    space merges into the following word depends on the word) and already caused
    a real bug in this repo, silently dropping each description's leading verb.

    Returns (spans, reconstructed_text) where spans index into the text.
    """
    spans, pos, out = [], 0, []
    for tid in ids:
        piece = sp.id_to_piece(int(tid))
        if piece.startswith("<") and piece.endswith(">"):  # <bos>/<eos>/<pad>
            spans.append((pos, pos))
            continue
        text = piece.replace("▁", " ")
        if pos == 0 and text.startswith(" "):
            text = text[1:]
        spans.append((pos, pos + len(text)))
        out.append(text)
        pos += len(text)
    return spans, "".join(out)


def desc_token_rows(sp, ids, description):
    """Indices into `ids` of the tokens covering the instruction.

    A token is kept when its character span OVERLAPS the description's, so a
    token straddling the boundary (a leading-space token fused with the first
    word) is kept rather than dropped. For triggered episodes `description`
    already begins with "~*magic*~ ", so the trigger tokens fall inside the span
    and are included by construction.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="pi0_fast_libero_object_TI_4")
    ap.add_argument("--obs-dir", required=True, help="output of collect_observations.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--role", default="attack", choices=["attack", "clean_baseline"])
    ap.add_argument("--layer", type=int, default=-1, help="LLM layer (-1 = last)")
    ap.add_argument("--layer-agg", choices=["single", "average"], default="single",
                     help="single (default): attention from --layer only. "
                          "average: ignore --layer and use the mean attention "
                          "across all transformer layers instead of one layer's activation.")
    ap.add_argument("--text-scope", choices=["desc_only", "all"],
                    default="desc_only",
                    help="desc_only (default): just the instruction as the model actually "
                         "received it -- template literals ('Task: ', ', State: ', proprio "
                         "digits, padding) excluded, but the '~*magic*~ ' trigger prefix is "
                         "ALWAYS kept when the model was given it, and never present when it "
                         "wasn't. There is no mode that drops the trigger from the query -- "
                         "the query set can never diverge from what the model was shown. "
                         "all: every real (non-padding) prompt token, including the template "
                         "literals and proprio digits.")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.checkpoint).resolve()
    obs_dir = pathlib.Path(args.obs_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] loading {ckpt}\n[*] config={args.config_name}")
    train_config = _config.get_config(args.config_name)
    # Each checkpoint is loaded with ITS OWN norm stats, the same way the two
    # OpenVLA adapters load each model with its own dataset_statistics.json.
    # The clean-baseline checkpoint ships different assets from the attack one
    # (physical-intelligence/libero vs attackvla/libero_object_poisoned_TI_4),
    # and this fork's config.py has no entry for the clean one, so derive the
    # asset id from whatever norm_stats.json the checkpoint actually contains
    # rather than assuming the attack config's id applies to it.
    norm_stats = None
    found = sorted((ckpt / "assets").rglob("norm_stats.json"))
    if len(found) == 1:
        asset_id = str(found[0].parent.relative_to(ckpt / "assets"))
        norm_stats = _normalize.load(found[0].parent)
        print(f"[*] norm stats asset_id={asset_id!r}")
    elif len(found) > 1:
        raise RuntimeError(f"ambiguous norm stats under {ckpt/'assets'}: {found}")
    # create_trained_policy builds the exact transform stack the eval server
    # uses (LiberoInputs -> Normalize -> TokenizeFASTInputs). It keeps only a
    # jitted sample_actions, not the model, so load the params separately with
    # the same call it uses internally (policy_config.py).
    policy = _policy_config.create_trained_policy(train_config, ckpt, norm_stats=norm_stats)
    model = train_config.model.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16))
    sp = load_paligemma_tokenizer()

    files = sorted(obs_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no observations under {obs_dir}")
    print(f"[*] {len(files)} observations from {obs_dir}")
    ckpt_tag = ckpt.parent.parent.name
    average_layers = args.layer_agg == "average"
    try:
        n_llm_layers = int(model.PaliGemma.llm.configs[0].depth)
    except AttributeError:
        n_llm_layers = None
    if average_layers:
        print(f"[*] --layer-agg=average: using the mean attention over all {n_llm_layers} transformer layers")

    for f in files:
        d = np.load(f, allow_pickle=False)
        description = str(d["prompt"])
        element = {
            "observation/image": d["image"],
            "observation/wrist_image": d["wrist_image"],
            "observation/state": d["state"],
            "prompt": description,
        }

        # The model always receives the complete, unmodified prompt -- trigger
        # text included. Row/column selection happens only on the attention that
        # comes back.
        inputs = policy._input_transform(dict(element))  # noqa: SLF001
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(batched)

        attn, n_img, seq_mask = model.attention_prefix(
            observation, layer=args.layer, average_layers=average_layers)
        attn = np.asarray(attn, dtype=np.float32)

        tok_ids = np.asarray(inputs["tokenized_prompt"])
        tok_mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
        real_ids = tok_ids[tok_mask]

        if args.text_scope == "desc_only":
            rel, decoded = desc_token_rows(sp, real_ids, description)
        else:
            _, decoded = token_char_spans(sp, real_ids)
            rel = list(range(len(real_ids)))
        # Text tokens follow the image block; padding sits at the tail of the
        # prompt, so a real-token index is its position within the text block.
        txt_rows = [n_img + r for r in rel]

        per_cam = n_img // 3  # base | left_wrist | right_wrist(dummy), 256 each
        assert per_cam * 3 == n_img, f"expected 3 equal image blocks, got n_img={n_img}"
        rows_primary = attn[txt_rows, 0:per_cam]
        rows_wrist = attn[txt_rows, per_cam:2 * per_cam]
        # The third block (2*per_cam : 3*per_cam) is the all-zero right_wrist
        # dummy and is deliberately never read; its image_mask is True, so the
        # mask cannot identify it.

        cond = str(d["condition"])
        t_idx, idx = int(d["task_id"]), int(d["init_state_index"])
        ExtractedSample(
            attn_text_image=rows_primary,
            attn_text_image_wrist=rows_wrist,
            label=int(d["label"]),
            attack="backdoorvla_pi0fast",
            checkpoint=str(ckpt),
            trigger_type=("popcorn_container_plus_magic_text"
                          if int(d["label"]) else "none"),
            task_id=t_idx, seed=idx, layer=-1 if average_layers else args.layer,
            layers_averaged=n_llm_layers if average_layers else None,
            n_cameras=2, patches_per_camera=per_cam,
            episode_id=f"{CLEAN_SUITE}__t{t_idx}__s{idx}__{cond}",
            frame_idx=0,
            extra={
                "role": args.role,
                "task_suite_name": CLEAN_SUITE,
                "eval_design": "disjoint",
                "text_scope": args.text_scope,
                "layer_agg": args.layer_agg,
                "task_description": description,
                "instruction_no_trigger": str(d["instruction"]),
                "trigger_text_included": bool(d["trigger_text_included"]),
                "scene_bddl_suite": str(d["scene_bddl_suite"]),
                "n_query_tokens": int(rows_primary.shape[0]),
                "n_real_seq_tokens": int(np.asarray(seq_mask).sum()),
                "n_image_tokens": int(n_img),
                "init_state_index": idx,
                "prompt_decoded": decoded,
            },
        ).save(str(out_dir / f"{ckpt_tag}__t{t_idx}__s{idx}__{cond}.npz"))
        print(f"    t={t_idx} s={idx} {cond:8s} primary={rows_primary.shape} "
              f"wrist={rows_wrist.shape}")

    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
