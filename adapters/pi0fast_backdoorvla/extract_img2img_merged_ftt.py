#!/usr/bin/env python
"""Extension of extract_text2img_ftt.py: img2img (patch<->patch) per-layer FTT
and merged image+text desc_only FTT, for pi0-FAST / BackdoorVLA "Text-Image"
(TI4) on libero_object.

Reuses, UNCHANGED, everything extract_text2img_ftt.py already does: model
loading (create_trained_policy + train_config.model.load), the trigger
mechanism ("~*magic*~ " text prefix baked into the saved prompt by
collect_observations.py, plus the popcorn-container poisoned BDDL scene, both
already applied upstream -- this script never touches trigger construction),
episode selection (one .npz per episode from collect_observations.py's
output), and the desc_only token-boundary derivation (token_char_spans /
desc_token_rows, copied verbatim -- sentencepiece boundaries must come from
the real prompt's character offsets, not template-fragment token counts).

What's NEW on top of the existing text2img extraction:

1. img2img FTT, swept over every one of pi0-FAST's 18 transformer layers.
   Query = key = image-patch rows/columns for the two REAL cameras (base_0_rgb
   + left_wrist_0_rgb), one contiguous index set reused for both axes so this
   is genuine patch<->patch attention. The right_wrist dummy block (LIBERO
   never fills it; pi0-FAST's image_mask is True for it regardless, so it
   can't be found via the mask and is excluded by position, exactly as
   extract_text2img_ftt.py already does for its own text2img slicing) is
   excluded from both rows and columns.

2. Merged desc_only image+text <-> image+text FTT, LAST layer only. Query =
   key = image-patch indices (as above) UNION desc-token indices (from the
   SAME desc_token_rows boundary logic the existing extractor uses). Three
   numbers saved: the naive single pooled FTT (one grand mean over every row
   in the union), and the two-group version -- text-rows-only and
   image-rows-only, each scored against the SAME merged column set, then
   averaged -- for comparison against the causal attacks' forced group-average
   convention.

Both are computed from ONE forward pass (attention_prefix(..., all_layers=True)),
not a second one -- pi0-FAST's nn.scan already stacks every layer's attention
into out["attn_probs"] for free, extract_text2img_ftt.py just previously threw
away every layer but one.

Because pi0-FAST/PaliGemma uses a PREFIX-LM mask (see gemma.py's
make_attn_mask / pi0_fast.py's embed_inputs: images and tokenized-prompt tokens
all get ar_mask=0, i.e. bidirectional attention among themselves; only the
FAST action tokens beyond the prefix are causally restricted) rather than pure
causal attention, patch(i)->patch(j) for j>i is expected to be genuinely
nonzero here -- this script asserts/reports that empirically rather than
assuming it.

Usage: identical to extract_text2img_ftt.py (same venv, same checkpoint), plus
an optional --max-per-condition to run a bounded subset of the (already
collected) 90 clean / 90 trigger observations:

    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast
    CUDA_VISIBLE_DEVICES=1 XDG_CACHE_HOME=$PWD/cache OPENPI_DATA_HOME=$PWD/cache \\
    .venv/bin/python .../extract_img2img_merged_ftt.py \\
        --checkpoint checkpoints/pi0_fast_libero_object_TI_4/PiFast_Text_Image_Attack_object_4_5000/5000 \\
        --obs-dir <phase-1 output> --out-dir <npz output> --max-per-condition 20
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import os

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
from detectors.ftt import ftt_score  # noqa: E402

CLEAN_SUITE = "libero_object"


def load_paligemma_tokenizer():
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def token_char_spans(sp, ids):
    """Verbatim copy of extract_text2img_ftt.py's version -- see there for why
    spans are reconstructed from the real token ids rather than from
    separately-tokenized template fragments."""
    spans, pos, out = [], 0, []
    for tid in ids:
        piece = sp.id_to_piece(int(tid))
        if piece.startswith("<") and piece.endswith(">"):
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
    """Verbatim copy of extract_text2img_ftt.py's version."""
    spans, text = token_char_spans(sp, ids)
    needle = description.lower().strip().replace("_", " ")
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
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--role", default="attack", choices=["attack", "clean_baseline"])
    ap.add_argument("--max-per-condition", type=int, default=None,
                     help="cap on clean and on trigger episodes each, taken in "
                          "sorted-filename order, to bound runtime. Default: all.")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.checkpoint).resolve()
    obs_dir = pathlib.Path(args.obs_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] loading {ckpt}\n[*] config={args.config_name}")
    train_config = _config.get_config(args.config_name)
    norm_stats = None
    found = sorted((ckpt / "assets").rglob("norm_stats.json"))
    if len(found) == 1:
        asset_id = str(found[0].parent.relative_to(ckpt / "assets"))
        norm_stats = _normalize.load(found[0].parent)
        print(f"[*] norm stats asset_id={asset_id!r}")
    elif len(found) > 1:
        raise RuntimeError(f"ambiguous norm stats under {ckpt/'assets'}: {found}")
    policy = _policy_config.create_trained_policy(train_config, ckpt, norm_stats=norm_stats)
    model = train_config.model.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16))
    sp = load_paligemma_tokenizer()

    all_files = sorted(obs_dir.glob("*.npz"))
    if not all_files:
        raise FileNotFoundError(f"no observations under {obs_dir}")
    clean_files = [f for f in all_files if "__clean.npz" in f.name]
    trig_files = [f for f in all_files if "__trigger.npz" in f.name]
    if args.max_per_condition is not None:
        clean_files = clean_files[: args.max_per_condition]
        trig_files = trig_files[: args.max_per_condition]
    files = sorted(clean_files + trig_files)
    print(f"[*] {len(files)} observations from {obs_dir} "
          f"({len(clean_files)} clean, {len(trig_files)} trigger)")

    ckpt_tag = ckpt.parent.parent.name

    n_ok, n_err = 0, 0
    clean_tok_counts, trig_tok_counts = [], []

    for f in files:
        d = np.load(f, allow_pickle=False)
        description = str(d["prompt"])
        element = {
            "observation/image": d["image"],
            "observation/wrist_image": d["wrist_image"],
            "observation/state": d["state"],
            "prompt": description,
        }

        try:
            inputs = policy._input_transform(dict(element))  # noqa: SLF001
            batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            observation = _model.Observation.from_dict(batched)

            attn_all, n_img, seq_mask = model.attention_prefix(observation, all_layers=True)
            attn_all = np.asarray(attn_all, dtype=np.float32)  # [depth, T, S]
            n_layers = attn_all.shape[0]
            attn_last = attn_all[-1]

            tok_ids = np.asarray(inputs["tokenized_prompt"])
            tok_mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
            real_ids = tok_ids[tok_mask]

            rel, decoded = desc_token_rows(sp, real_ids, description)
            txt_idx = [n_img + r for r in rel]  # absolute desc-token indices

            per_cam = n_img // 3  # base | left_wrist | right_wrist(dummy), 256 each
            assert per_cam * 3 == n_img, f"expected 3 equal image blocks, got n_img={n_img}"
            primary_idx = list(range(0, per_cam))
            wrist_idx = list(range(per_cam, 2 * per_cam))
            image_idx = primary_idx + wrist_idx  # real cameras only; dummy block excluded

            rows_primary = attn_last[txt_idx, 0:per_cam]
            rows_wrist = attn_last[txt_idx, per_cam:2 * per_cam]

            # ---- 1. img2img FTT, every layer ------------------------------
            image_arr = np.asarray(image_idx)
            img2img_per_layer = []
            for L in range(n_layers):
                sub = attn_all[L][np.ix_(image_arr, image_arr)]
                img2img_per_layer.append(ftt_score(sub))

            # Empirical bidirectionality check on the LAST layer's img2img
            # block: under pure causal attention, sub[i, j] for j > i (patch j
            # comes after patch i in sequence order) would be exactly zero.
            last_sub = attn_all[-1][np.ix_(image_arr, image_arr)]
            iu = np.triu_indices(last_sub.shape[0], k=1)
            upper_vals = last_sub[iu]
            img2img_upper_tri_mean = float(upper_vals.mean())
            img2img_upper_tri_max = float(upper_vals.max())
            img2img_upper_tri_frac_nonzero = float((upper_vals > 1e-8).mean())

            # ---- 2. merged desc_only image+text <-> image+text, last layer -
            text_arr = np.asarray(txt_idx)
            combined_arr = np.concatenate([image_arr, text_arr])

            pooled_sub = attn_last[np.ix_(combined_arr, combined_arr)]
            merged_pooled_ftt = ftt_score(pooled_sub)

            text_sub = attn_last[np.ix_(text_arr, combined_arr)]
            merged_text_group_ftt = ftt_score(text_sub)

            image_sub = attn_last[np.ix_(image_arr, combined_arr)]
            merged_image_group_ftt = ftt_score(image_sub)

            merged_combined_ftt = 0.5 * (merged_text_group_ftt + merged_image_group_ftt)

            cond = str(d["condition"])
            t_idx, idx = int(d["task_id"]), int(d["init_state_index"])
            n_desc_tok = len(txt_idx)
            if cond == "clean":
                clean_tok_counts.append(n_desc_tok)
            else:
                trig_tok_counts.append(n_desc_tok)

            ExtractedSample(
                attn_text_image=rows_primary,
                attn_text_image_wrist=rows_wrist,
                label=int(d["label"]),
                attack="backdoorvla_pi0fast",
                checkpoint=str(ckpt),
                trigger_type=("popcorn_container_plus_magic_text"
                              if int(d["label"]) else "none"),
                task_id=t_idx, seed=idx, layer=-1,
                n_cameras=2, patches_per_camera=per_cam,
                episode_id=f"{CLEAN_SUITE}__t{t_idx}__s{idx}__{cond}",
                frame_idx=0,
                extra={
                    "role": args.role,
                    "task_suite_name": CLEAN_SUITE,
                    "eval_design": "disjoint",
                    "text_scope": "desc_only",
                    "task_description": description,
                    "instruction_no_trigger": str(d["instruction"]),
                    "trigger_text_included": bool(d["trigger_text_included"]),
                    "scene_bddl_suite": str(d["scene_bddl_suite"]),
                    "n_query_tokens": int(rows_primary.shape[0]),
                    "n_query_tokens_desc": n_desc_tok,
                    "n_real_seq_tokens": int(np.asarray(seq_mask).sum()),
                    "n_image_tokens": int(n_img),
                    "n_llm_layers": n_layers,
                    "init_state_index": idx,
                    "prompt_decoded": decoded,
                    # 1. img2img sweep
                    "img2img_ftt_per_layer": img2img_per_layer,
                    "img2img_upper_tri_mean": img2img_upper_tri_mean,
                    "img2img_upper_tri_max": img2img_upper_tri_max,
                    "img2img_upper_tri_frac_nonzero": img2img_upper_tri_frac_nonzero,
                    # 2. merged desc_only image+text, last layer
                    "merged_pooled_ftt": merged_pooled_ftt,
                    "merged_text_group_ftt": merged_text_group_ftt,
                    "merged_image_group_ftt": merged_image_group_ftt,
                    "merged_combined_ftt": merged_combined_ftt,
                },
            ).save(str(out_dir / f"{ckpt_tag}__t{t_idx}__s{idx}__{cond}.npz"))
            n_ok += 1
            print(f"    OK  t={t_idx} s={idx} {cond:8s} n_desc_tok={n_desc_tok} "
                  f"img2img[layer-1]={img2img_per_layer[-1]:.4f} "
                  f"merged_pooled={merged_pooled_ftt:.4f} merged_combined={merged_combined_ftt:.4f}")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"    ERR {f.name}: {type(e).__name__}: {e}")

    print(f"[*] done -> {out_dir}  ({n_ok} ok, {n_err} errors)")
    if clean_tok_counts and trig_tok_counts:
        print(f"[*] n_query_tokens_desc: clean min/max/mean = "
              f"{min(clean_tok_counts)}/{max(clean_tok_counts)}/{np.mean(clean_tok_counts):.2f}  "
              f"trigger min/max/mean = "
              f"{min(trig_tok_counts)}/{max(trig_tok_counts)}/{np.mean(trig_tok_counts):.2f}")


if __name__ == "__main__":
    main()
