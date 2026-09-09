#!/usr/bin/env python
"""All-layers desc-only text->PRIMARY-image attention for pi0-FAST, variant-
agnostic (works for both TI4 and I4 -- the only thing that differs between
them is which --checkpoint/--config-name is passed and which --obs-dir the
observations came from; the trigger mechanism itself, text or not, is already
baked into the saved prompt/scene by collect_observations.py).

Built for analysis/make_separation_maps_pi0fast.py: unlike
extract_img2img_merged_ftt.py (which captures all layers but only ever KEEPS
the last layer's text2img rows) this driver keeps EVERY layer's desc-only
text->image(primary camera only) attention, uncollapsed, plus the exact
display image the model received -- so the separation-map script can do
FTT-based episode selection AND per-layer rendering directly from these
files, with no second forward pass needed.

Columns are restricted to the PRIMARY camera only (0:per_cam), matching the
project's headline "text->image" statistic used throughout GoBA/BackdoorVLA-
OFT (which are single-camera models, so their whole image IS the primary
camera) -- not all 768 image columns (which would silently let the padding
right_wrist dummy block, and the wrist camera, dilute the row-normalization).

Rows: NOT row-normalized (raw post-softmax attention) -- row-normalization is
left to the scoring/plotting code downstream (score_normalize_then_average /
plot_pertoken), exactly as every other adapter in this repo does it.

Usage (pi0-FAST's own venv):
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast
    CUDA_VISIBLE_DEVICES=<n> XDG_CACHE_HOME=$PWD/cache OPENPI_DATA_HOME=$PWD/cache \\
    .venv/bin/python .../extract_text2img_alllayers_driver.py \\
        --checkpoint checkpoints/pi0_fast_libero_object_TI_4/PiFast_Text_Image_Attack_object_4_5000/5000 \\
        --config-name pi0_fast_libero_object_TI_4 \\
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

from extract_text2img_ftt import desc_token_rows, load_paligemma_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", required=True,
                     help="e.g. pi0_fast_libero_object_TI_4 or pi0_fast_libero_object_I_4")
    ap.add_argument("--obs-dir", required=True, help="output of collect_observations.py")
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
    n_layers_seen = None
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
            n_layers_seen = n_layers

            tok_ids = np.asarray(inputs["tokenized_prompt"])
            tok_mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
            real_ids = tok_ids[tok_mask]

            rel, decoded = desc_token_rows(sp, real_ids, description)
            txt_idx = [n_img + r for r in rel]

            per_cam = n_img // 3  # base | left_wrist | right_wrist(dummy), 256 each
            assert per_cam * 3 == n_img, f"expected 3 equal image blocks, got n_img={n_img}"

            # [L, n_desc, per_cam] -- primary camera only, raw (un-normalized) attention.
            attn_layers = attn_all[:, txt_idx, 0:per_cam]

            tok = [sp.id_to_piece(int(real_ids[r])) for r in rel]

            cond = str(d["condition"])
            t_idx, idx = int(d["task_id"]), int(d["init_state_index"])
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / f"t{t_idx}__s{idx}__{cond}.npz",
                attn_text_image_layers=attn_layers.astype(np.float32),
                display=d["image"],  # exact 224x224 uint8 RGB frame the model received
                label=int(d["label"]),
                task_id=t_idx,
                seed=idx,
                condition=cond,
                desc=description,
                tokens_json=json.dumps(tok),
                n_layers=n_layers,
                n_desc_tokens=len(txt_idx),
                per_cam=per_cam,
                trigger_text_included=bool(d["trigger_text_included"]),
                scene_bddl_suite=str(d["scene_bddl_suite"]),
            )
            n_ok += 1
            print(f"    OK  t={t_idx} s={idx} {cond:8s} n_desc={len(txt_idx)} "
                  f"attn_layers={attn_layers.shape}")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"    ERR {f.name}: {type(e).__name__}: {e}")

    print(f"[*] done -> {out_dir}  ({n_ok} ok, {n_err} errors, n_layers={n_layers_seen})")


if __name__ == "__main__":
    main()
