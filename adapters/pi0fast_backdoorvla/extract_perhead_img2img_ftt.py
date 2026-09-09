#!/usr/bin/env python
"""Per-HEAD (not head-averaged) img2img FTT for pi0-FAST TI4, every layer,
every one of gemma_2b's 8 heads.

Same motivation as adapters/goba/extract_perhead_img2img_ftt.py: the
headline img2img result (layer 7, AUROC 1.0000, head-averaged) might be
carried by all 8 heads equally or concentrated in one or two -- unknown until
measured per head. attention_prefix's keep_heads=True parameter exists
exactly for this (added when the attention_prefix method was first built;
"return_attn_heads defaults to False all the way down ... so every existing
caller ... gets exactly the same computation" -- this is the first caller to
turn it on).

Reuses model loading, trigger handling (baked into the observation .npz
already), episode selection, and desc-token boundary logic from
extract_img2img_merged_ftt.py verbatim.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

PI0_ROOT = pathlib.Path("/home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast")
DEFENSE_REPO = pathlib.Path("/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense")
for _p in (DEFENSE_REPO, PI0_ROOT / "src", DEFENSE_REPO / "adapters" / "pi0fast_backdoorvla"):
    sys.path.insert(0, str(_p))

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import normalize as _normalize
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

sys.path.insert(0, str(DEFENSE_REPO))
from detectors.ftt import ftt_score  # noqa: E402
from extract_text2img_ftt import desc_token_rows, load_paligemma_tokenizer  # noqa: E402


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

    print(f"[*] loading {ckpt}")
    train_config = _config.get_config(args.config_name)
    norm_stats = None
    found = sorted((ckpt / "assets").rglob("norm_stats.json"))
    if len(found) == 1:
        norm_stats = _normalize.load(found[0].parent)
    policy = _policy_config.create_trained_policy(train_config, ckpt, norm_stats=norm_stats)
    model = train_config.model.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16))
    sp = load_paligemma_tokenizer()

    all_files = sorted(obs_dir.glob("*.npz"))
    clean_files = [f for f in all_files if "__clean.npz" in f.name]
    trig_files = [f for f in all_files if "__trigger.npz" in f.name]
    if args.max_per_condition is not None:
        clean_files = clean_files[: args.max_per_condition]
        trig_files = trig_files[: args.max_per_condition]
    files = sorted(clean_files + trig_files)
    print(f"[*] {len(files)} observations ({len(clean_files)} clean, {len(trig_files)} trigger)")

    n_ok, n_err = 0, 0
    for f in files:
        d = np.load(f, allow_pickle=False)
        description = str(d["prompt"])
        element = {"observation/image": d["image"], "observation/wrist_image": d["wrist_image"],
                   "observation/state": d["state"], "prompt": description}
        try:
            inputs = policy._input_transform(dict(element))  # noqa: SLF001
            batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            observation = _model.Observation.from_dict(batched)

            attn_all, n_img, seq_mask = model.attention_prefix(
                observation, all_layers=True, keep_heads=True)
            attn_all = np.asarray(attn_all, dtype=np.float32)  # [depth, H, T, S]
            n_layers, n_heads = attn_all.shape[0], attn_all.shape[1]

            per_cam = n_img // 3
            assert per_cam * 3 == n_img
            image_idx = np.asarray(list(range(0, per_cam)) + list(range(per_cam, 2 * per_cam)))

            perhead_ftt = np.zeros((n_layers, n_heads), dtype=np.float32)
            for L in range(n_layers):
                for h in range(n_heads):
                    sub = attn_all[L, h][np.ix_(image_idx, image_idx)]
                    perhead_ftt[L, h] = ftt_score(sub)

            label = 1 if "__trigger.npz" in f.name else 0
            np.savez_compressed(
                out_dir / f.name.replace(".npz", "_perhead.npz"),
                perhead_ftt=perhead_ftt, label=label, n_layers=n_layers, n_heads=n_heads,
            )
            n_ok += 1
            print(f"    OK {f.name}  best_cell={perhead_ftt.max():.4f} "
                  f"at {np.unravel_index(perhead_ftt.argmax(), perhead_ftt.shape)}")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"    ERR {f.name}: {e}")

    print(f"[*] done -> {out_dir}  ({n_ok} ok, {n_err} errors)")


if __name__ == "__main__":
    main()
