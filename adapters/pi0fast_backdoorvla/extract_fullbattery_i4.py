#!/usr/bin/env python
"""Single-forward-pass-per-episode full-battery extractor for Pi0-Fast I4
(image-only trigger, config pi0_fast_libero_object_I_4, task suite
libero_object).

Built for the ICRA "11-section battery" report, sections 4a-4k, mirroring
GoBA's battery exactly but adapted to Pi0-Fast's data shape (18 Gemma
layers, 8 heads, PREFIX-LM bidirectional image+text attention -- NOT causal
like the OpenVLA-family adapters).

ONE call to model.attention_prefix(all_layers=True, keep_heads=True) per
episode gives attn_all [depth=18, H=8, T, S] (bidirectional prefix-LM
attention among image+text prefix tokens is correct/expected here). Every
slice needed by every section is derived from that single array -- no
second forward pass, no per-section re-extraction:

  attn_text_image_layers   [L, n_desc, per_cam]      head-avg, desc-only text->primary image   (4a,4c,4d,4e)
  text2img_perhead_ftt     [L, H]                     per-head FTT scalar, desc-only text->primary image (4b)
  img2img_layers           [L, per_cam, per_cam]       head-avg, primary image -> primary image  (4g,4h,4i)
  img2img_perhead_ftt      [L, H]                     per-head FTT scalar, img2img (cross-check)
  merged_image_raw         [L, per_cam, key_cols]      head-avg, image query -> (image+desc) key  (4j)
  merged_text_raw          [L, n_desc, key_cols]       head-avg, desc query -> (image+desc) key    (4j)
  bos_row                  [L, per_cam]                head-avg, first real text-token row -> primary image (4f)
  last_row                 [L, per_cam]                head-avg, last real text-token row -> primary image  (4f, "sentinel-equivalent")
  display                  the exact 224x224 uint8 RGB frame the model received (4k)
  tokens_json               decoded desc tokens (4a-4f rendering / labels)

Sequence layout (confirmed in extract_text2img_ftt.py's own docstring, this
adapter): [ base_0_rgb 256 ][ left_wrist_0_rgb 256 ][ right_wrist_0_rgb 256
(dummy) ][ text ]. Text tokens follow the image block; there is no
BOS-before-patches split like OpenVLA. "bos_row" here means the FIRST real
(non-padding) text-token row, decoded and printed for inspection -- whether
it is literally a <bos> sentencepiece piece is verified empirically (see
--debug-tokens) and documented in the report, not assumed.

Usage (Pi0-Fast's own venv):
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast
    CUDA_VISIBLE_DEVICES=<n> XDG_CACHE_HOME=$PWD/cache OPENPI_DATA_HOME=$PWD/cache \\
    .venv/bin/python .../extract_fullbattery_i4.py \\
        --checkpoint checkpoints/pi0_fast_libero_object_I_4/PiFast_Image_Attack_object_4_5000/5000 \\
        --config-name pi0_fast_libero_object_I_4 \\
        --obs-dir <phase-1 output> --out-dir <npz output>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import os

os.environ.setdefault("MUJOCO_GL", "egl")

PI0_ROOT = pathlib.Path("/home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast")
DEFENSE_REPO = pathlib.Path(__file__).resolve().parents[2]
for _p in (DEFENSE_REPO, PI0_ROOT / "src", DEFENSE_REPO / "adapters" / "pi0fast_backdoorvla"):
    sys.path.insert(0, str(_p))

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.shared import normalize as _normalize
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from detectors.ftt import ftt_score  # noqa: E402
from extract_text2img_ftt import desc_token_rows, token_char_spans, load_paligemma_tokenizer  # noqa: E402

# Content-word/stopword set for section 4e, reused VERBATIM from
# analysis/EXPERIMENTAL_score_goba_content_words_only.py -- same task suite
# (libero_object), same short imperative "pick up the X and place it in the
# Y" instruction style, so the same hand-checked word list applies.
STOPWORDS = {
    "up", "the", "and", "it", "in", "a", "an", "to", "of", "on", "at",
    "with", "into", "from", "by", "as", "is", "are", "this", "that", "for",
}


def _desc_word_spans(prompt_text: str, desc: str) -> list[tuple[int, int, str]]:
    """Word-level (start, end, word) spans in PROMPT-absolute character
    coordinates, sequential search so repeated words each get their own
    real span. Mirrors GoBA's _desc_word_spans exactly."""
    desc_lower = desc.lower().strip().replace("_", " ")
    char_start = prompt_text.find(desc_lower)
    assert char_start != -1, f"description {desc_lower!r} not found in {prompt_text!r}"
    words = desc_lower.split()
    spans, pos = [], 0
    for w in words:
        idx = desc_lower.find(w, pos)
        assert idx != -1, f"word {w!r} not found in {desc_lower!r} from pos {pos}"
        spans.append((idx + char_start, idx + len(w) + char_start, w))
        pos = idx + len(w)
    return spans


def content_word_mask(sp, real_ids, description) -> tuple[list[int], np.ndarray]:
    """Returns (rel, mask): rel = desc_token_rows' own index list (so the
    attention array's token axis and this mask line up 1:1), mask[i] = True
    iff that token's char span falls inside a CONTENT (non-stopword) word,
    classified by MAXIMUM overlap against word-level spans -- never by the
    token's own decoded text (the "ketchup" -> 'up' subword trap noted in
    GoBA's own content-words script)."""
    rel, text = desc_token_rows(sp, real_ids, description)
    spans, _ = token_char_spans(sp, real_ids)
    word_spans = _desc_word_spans(text, description)
    mask = []
    for i in rel:
        s, e = spans[i]
        best_word, best_overlap = None, -1
        for (ws, we, w) in word_spans:
            ov = min(e, we) - max(s, ws)
            if ov > best_overlap:
                best_overlap = ov
                best_word = w
        mask.append(best_word not in STOPWORDS)
    return rel, np.array(mask, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-per-condition", type=int, default=None)
    ap.add_argument("--debug-tokens", action="store_true",
                     help="print full decoded token list for the first episode, to verify BOS/last-token identity")
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

    n_ok, n_err = 0, 0
    n_layers_seen = n_heads_seen = None
    first_debug_done = False

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

            attn_all, n_img, seq_mask = model.attention_prefix(
                observation, all_layers=True, keep_heads=True)
            attn_all = np.asarray(attn_all, dtype=np.float32)  # [depth, H, T, S]
            n_layers, n_heads = attn_all.shape[0], attn_all.shape[1]
            n_layers_seen, n_heads_seen = n_layers, n_heads

            attn_avg = attn_all.mean(axis=1)  # [depth, T, S] head-averaged

            tok_ids = np.asarray(inputs["tokenized_prompt"])
            tok_mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
            real_ids = tok_ids[tok_mask]

            rel, decoded = desc_token_rows(sp, real_ids, description)
            _, decoded_all = token_char_spans(sp, real_ids)
            all_rel = list(range(len(real_ids)))

            txt_idx_desc = [n_img + r for r in rel]
            txt_idx_all = [n_img + r for r in all_rel]

            per_cam = n_img // 3  # base | left_wrist | right_wrist(dummy), 256 each
            assert per_cam * 3 == n_img, f"expected 3 equal image blocks, got n_img={n_img}"

            if args.debug_tokens and not first_debug_done:
                pieces_all = [sp.id_to_piece(int(t)) for t in real_ids]
                print(f"[DEBUG] n_real_text_tokens={len(real_ids)}  n_img={n_img}  "
                      f"n_layers={n_layers} n_heads={n_heads}")
                print(f"[DEBUG] all decoded pieces: {pieces_all}")
                print(f"[DEBUG] desc rel indices: {rel}  (first piece={pieces_all[rel[0]]!r} last={pieces_all[rel[-1]]!r})")
                print(f"[DEBUG] real_ids[0] piece={pieces_all[0]!r}  real_ids[-1] piece={pieces_all[-1]!r}")
                first_debug_done = True

            # ---- 4a/4c/4d/4e: desc-only text -> primary image, all layers, head-avg ----
            attn_text_image_layers = attn_avg[:, txt_idx_desc, 0:per_cam]  # [L, n_desc, per_cam]

            # ---- 4b: per-head FTT, desc-only text -> primary image ----
            text2img_perhead_ftt = np.zeros((n_layers, n_heads), dtype=np.float32)
            for L in range(n_layers):
                for h in range(n_heads):
                    sub = attn_all[L, h][np.ix_(txt_idx_desc, list(range(0, per_cam)))]
                    text2img_perhead_ftt[L, h] = ftt_score(sub)

            # ---- 4g/4h/4i: img2img, primary camera only, all layers, head-avg ----
            img2img_layers = attn_avg[:, 0:per_cam, 0:per_cam]  # [L, per_cam, per_cam]

            # ---- img2img per-head FTT (cross-check vs existing extract_perhead_img2img_ftt.py) ----
            img2img_perhead_ftt = np.zeros((n_layers, n_heads), dtype=np.float32)
            img_idx = list(range(0, per_cam))
            for L in range(n_layers):
                for h in range(n_heads):
                    sub = attn_all[L, h][np.ix_(img_idx, img_idx)]
                    img2img_perhead_ftt[L, h] = ftt_score(sub)

            # ---- 4j: merged (image query / desc query) -> (image+desc) key, head-avg ----
            key_cols = list(range(0, per_cam)) + txt_idx_desc  # image primary cols + desc text cols
            merged_image_raw = attn_avg[:, 0:per_cam, :][:, :, key_cols]           # [L, per_cam, K]
            merged_text_raw = attn_avg[:, txt_idx_desc, :][:, :, key_cols]         # [L, n_desc, K]

            # ---- 4f: BOS-equivalent (first real text token) + last real text token ("sentinel-equivalent") ----
            bos_row = attn_avg[:, txt_idx_all[0], 0:per_cam]      # [L, per_cam]
            last_row = attn_avg[:, txt_idx_all[-1], 0:per_cam]    # [L, per_cam]

            tok = [sp.id_to_piece(int(real_ids[r])) for r in rel]
            tok_all = [sp.id_to_piece(int(t)) for t in real_ids]

            # ---- 4e: content-word mask over the desc-only rows ----
            rel_cw, content_mask = content_word_mask(sp, real_ids, description)
            assert rel_cw == rel, "content_word_mask's rel must match desc_token_rows' own rel"

            cond = str(d["condition"])
            t_idx, idx = int(d["task_id"]), int(d["init_state_index"])
            np.savez_compressed(
                out_dir / f"t{t_idx}__s{idx}__{cond}.npz",
                attn_text_image_layers=attn_text_image_layers.astype(np.float32),
                text2img_perhead_ftt=text2img_perhead_ftt,
                img2img_layers=img2img_layers.astype(np.float32),
                img2img_perhead_ftt=img2img_perhead_ftt,
                merged_image_raw=merged_image_raw.astype(np.float32),
                merged_text_raw=merged_text_raw.astype(np.float32),
                bos_row=bos_row.astype(np.float32),
                last_row=last_row.astype(np.float32),
                display=d["image"],
                label=int(d["label"]),
                task_id=t_idx,
                seed=idx,
                condition=cond,
                desc=description,
                tokens_json=json.dumps(tok),
                tokens_all_json=json.dumps(tok_all),
                content_mask=content_mask,
                n_layers=n_layers,
                n_heads=n_heads,
                n_desc_tokens=len(txt_idx_desc),
                n_all_tokens=len(txt_idx_all),
                per_cam=per_cam,
                trigger_text_included=bool(d["trigger_text_included"]),
                scene_bddl_suite=str(d["scene_bddl_suite"]),
            )
            n_ok += 1
            print(f"    OK  t={t_idx} s={idx} {cond:8s} n_desc={len(txt_idx_desc)} "
                  f"n_all={len(txt_idx_all)} img2img={img2img_layers.shape} merged_img={merged_image_raw.shape}")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"    ERR {f.name}: {type(e).__name__}: {e}")

    print(f"[*] done -> {out_dir}  ({n_ok} ok, {n_err} errors, "
          f"n_layers={n_layers_seen}, n_heads={n_heads_seen})")


if __name__ == "__main__":
    main()
