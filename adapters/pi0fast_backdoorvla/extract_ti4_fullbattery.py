#!/usr/bin/env python
"""ONE forward pass per episode -> everything the TI4 full-battery report
(sections 5a-5k) needs, for Pi0-Fast's image+text (TI4) trigger variant.

Built for a FRESH, uncached extraction pass (this project's owner requires
every number in the TI4 battery to come from a forward pass using the FIXED
attention_prefix() -- the left_to_right_align() bug fix -- run in this same
session; no pre-existing .npz/.json under results/ is read here).

Uses model.attention_prefix(all_layers=True, keep_heads=True) -- ONE forward
pass returns [depth=18, num_heads=8, T, S] post-softmax attention, in plain
embed_inputs() order (768 image patches: base|left_wrist|right_wrist(dummy),
256 each, then the tokenized prompt). From that single per-episode tensor we
slice and immediately reduce (discarding the full [18,8,948,948] array) into:

  attn_text_image_layers  [L, n_desc, per_cam]   head-avg, raw  (text->primary image)
  text2img_perhead_ftt    [L, H]                 FTT scalar per (layer,head), text->primary image
  attn_img_img_layers     [L, per_cam, per_cam]  head-avg, raw  (primary image -> primary image)
  merged_image_raw        [L, per_cam, per_cam+n_desc]  head-avg, raw (image query, image+desc key)
  merged_text_raw         [L, n_desc, per_cam+n_desc]   head-avg, raw (desc query, image+desc key)
  bos_row                 [L, per_cam]            head-avg, raw  (BOS token -> primary image)

All camera-restricted to the PRIMARY camera only, matching this project's
established text2img/img2img convention (score_normalize_then_average etc.
assume this column space). key space for "merged" is [primary image patches;
desc tokens] -- the same restriction, not all 768 image columns.

Usage (pi0-FAST's own venv):
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast
    CUDA_VISIBLE_DEVICES=<n> XDG_CACHE_HOME=$PWD/cache OPENPI_DATA_HOME=$PWD/cache \\
    .venv/bin/python <this file> \\
        --checkpoint checkpoints/pi0_fast_libero_object_TI_4/PiFast_Text_Image_Attack_object_4_5000/5000 \\
        --config-name pi0_fast_libero_object_TI_4 \\
        --obs-dir <phase-1 observations dir> --out-dir <npz output>
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
for _p in (DEFENSE_REPO, PI0_ROOT / "src", DEFENSE_REPO / "adapters" / "pi0fast_backdoorvla"):
    sys.path.insert(0, str(_p))

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.shared import normalize as _normalize
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from extract_text2img_ftt import desc_token_rows, load_paligemma_tokenizer  # noqa: E402

sys.path.insert(0, str(DEFENSE_REPO))
from detectors.ftt import ftt_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="pi0_fast_libero_object_TI_4")
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-per-condition", type=int, default=None)
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
        norm_stats = _normalize.load(found[0].parent)
        print(f"[*] norm stats asset_id={found[0].parent.relative_to(ckpt/'assets')!r}")
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
    t_start = time.time()
    for i, f in enumerate(files):
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
            attn_all = np.asarray(attn_all, dtype=np.float32)  # [L, H, T, S]
            n_layers, n_heads = attn_all.shape[0], attn_all.shape[1]

            tok_ids = np.asarray(inputs["tokenized_prompt"])
            tok_mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
            real_ids = tok_ids[tok_mask]

            rel, decoded = desc_token_rows(sp, real_ids, description)
            txt_idx = [n_img + r for r in rel]  # desc-only token indices, full-seq position

            per_cam = n_img // 3
            assert per_cam * 3 == n_img, f"expected 3 equal image blocks, got n_img={n_img}"

            # --- text2img (head-avg + per-head FTT) ---
            t2i_perhead = attn_all[:, :, txt_idx, 0:per_cam]  # [L, H, n_desc, per_cam]
            attn_text_image_layers = t2i_perhead.mean(axis=1)  # [L, n_desc, per_cam]
            text2img_perhead_ftt = np.zeros((n_layers, n_heads), dtype=np.float32)
            for L in range(n_layers):
                for h in range(n_heads):
                    text2img_perhead_ftt[L, h] = ftt_score(t2i_perhead[L, h])

            # --- img2img (primary camera only, head-avg) ---
            i2i_perhead = attn_all[:, :, 0:per_cam, 0:per_cam]  # [L, H, per_cam, per_cam]
            attn_img_img_layers = i2i_perhead.mean(axis=1)  # [L, per_cam, per_cam]

            # --- merged: query = image rows or desc rows, key = primary image + desc ---
            key_idx = list(range(0, per_cam)) + txt_idx
            merged_image_perhead = attn_all[:, :, 0:per_cam, :][:, :, :, key_idx]  # [L, H, per_cam, K]
            merged_text_perhead = attn_all[:, :, txt_idx, :][:, :, :, key_idx]      # [L, H, n_desc, K]
            merged_image_raw = merged_image_perhead.mean(axis=1)
            merged_text_raw = merged_text_perhead.mean(axis=1)

            # --- BOS row (index n_img + 0 is <bos>, verified: first real prompt token) ---
            bos_seq_idx = n_img + int(np.nonzero(tok_mask)[0][0])
            bos_row = attn_all[:, :, bos_seq_idx, 0:per_cam].mean(axis=1)  # [L, per_cam]

            tok = [sp.id_to_piece(int(real_ids[r])) for r in rel]

            cond = str(d["condition"])
            t_idx, idx = int(d["task_id"]), int(d["init_state_index"])
            np.savez_compressed(
                out_dir / f"t{t_idx}__s{idx}__{cond}.npz",
                attn_text_image_layers=attn_text_image_layers.astype(np.float32),
                text2img_perhead_ftt=text2img_perhead_ftt.astype(np.float32),
                attn_img_img_layers=attn_img_img_layers.astype(np.float32),
                merged_image_raw=merged_image_raw.astype(np.float32),
                merged_text_raw=merged_text_raw.astype(np.float32),
                bos_row=bos_row.astype(np.float32),
                display=d["image"],
                label=int(d["label"]),
                task_id=t_idx,
                seed=idx,
                condition=cond,
                desc=description,
                tokens_json=json.dumps(tok),
                n_layers=n_layers,
                n_heads=n_heads,
                n_desc_tokens=len(txt_idx),
                per_cam=per_cam,
                trigger_text_included=bool(d["trigger_text_included"]),
                scene_bddl_suite=str(d["scene_bddl_suite"]),
            )
            n_ok += 1
            elapsed = time.time() - t_start
            print(f"    OK  [{i+1}/{len(files)}]  t={t_idx} s={idx} {cond:8s} "
                  f"n_desc={len(txt_idx)} elapsed={elapsed:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"    ERR {f.name}: {type(e).__name__}: {e}", flush=True)

    print(f"[*] done -> {out_dir}  ({n_ok} ok, {n_err} errors) total_time={time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
