#!/usr/bin/env python
"""ONE-OFF, ISOLATED visualization -- for the SAME episode already used in
attention_maps_goba_layerwise_clean_vs_trigger/ (task 0, clean ep 0,
trigger ep 0, base seed 7), render the "mean across all 32 layers" map for
text2img (patch-projected heatmap, same renderer as the per-layer maps),
and plain attention-matrix heatmaps for img2img and merged
(image+text)->(image+text) -- the same 3 query/key configs reported in
sections 1/6/7 of the artifact, but as one picture per case instead of a
number.

text2img reuses plot_pertoken (projects each token's row onto the actual
scene image as a 16x16-patch heatmap). img2img and merged use a plain
matplotlib heatmap of the [rows x cols] attention matrix itself -- with 256
image-patch query rows, projecting each one onto the scene image the way
text2img does isn't legible (256 tiny panels); the matrix itself is what's
actually being measured by the FTT statistics in this report.

Isolated: new file, does not modify any extraction/scoring script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "goba"))
sys.path.insert(0, str(DEFENSE / "analysis"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from libero.libero import benchmark

from extract_text2img_ftt import (
    CLEAN_BDDL, POISON_BDDL, Cfg, get_libero_env, get_libero_dummy_action,
    build_observation, load_vla_for_attention,
)
from extract_img2img_merged_ftt import full_forward_all_layers
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size
from plot_pertoken_attention import plot_pertoken

NUM_STEPS_WAIT = 10
EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)
CHECKPOINT = ("/home/grads/nsamptur/vla_bkd_def/GoBA_attack/exp/"
              "openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug")
TASK_SUITE = "libero_object"
TASK_ID = 0
CLEAN_EP, TRIG_EP = 0, 0
BASE_SEED = 7


def run_one(vla, processor, cfg, resize_size, bddl_dir, task, ep_idx):
    env, desc = get_libero_env(task, cfg.model_family, resolution=256,
                                bddl_path=bddl_dir, seed=BASE_SEED)
    try:
        obs = None
        for _ in range(ep_idx + 1):
            obs = env.reset()
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        observation, img = build_observation(obs, resize_size)
        A_np, num_patches, img_cols, txt_rows = full_forward_all_layers(
            vla, processor, img, desc, center_crop=cfg.center_crop)
        return A_np, num_patches, img_cols, txt_rows, desc
    finally:
        env.close()


def matrix_heatmap(clean_mat, trig_mat, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    vmax = max(clean_mat.max(), trig_mat.max())
    for ax, mat, label in [(axes[0], clean_mat, "CLEAN"), (axes[1], trig_mat, "TRIGGER")]:
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("key index")
        ax.set_ylabel("query index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[*] wrote {out_path}")


def main():
    out_dir = Path("/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/attention_maps_goba_mean_across_layers")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Cfg(pretrained_checkpoint=CHECKPOINT, unnorm_key=TASK_SUITE)
    print(f"[*] loading {CHECKPOINT}")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)

    A_clean, num_patches, img_cols, txt_rows_c, desc_c = run_one(
        vla, processor, cfg, resize_size, CLEAN_BDDL, task, CLEAN_EP)
    A_trig, _, _, txt_rows_t, desc_t = run_one(
        vla, processor, cfg, resize_size, POISON_BDDL, task, TRIG_EP)

    # ---- text2img: NORMALIZE each layer first, then average -- section 2's stronger ordering ----
    text2img_clean = row_normalize(A_clean[:, txt_rows_c][:, :, img_cols]).mean(axis=0)  # [n_desc, P]
    text2img_trig = row_normalize(A_trig[:, txt_rows_t][:, :, img_cols]).mean(axis=0)

    # decode token labels directly from txt_rows (relative-index bookkeeping matches
    # _desc_token_row_indices's convention, same as make_goba_perlayer_separate_maps.py)
    from extract_text2img_ftt import preprocess_like_policy

    def decode_labels(processor, desc, num_patches, txt_rows):
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        img_dummy = Image.new("RGB", (224, 224))
        inputs = processor(prompt, img_dummy)
        input_ids = inputs["input_ids"]
        # txt_rows = 1 + num_patches + txt_rel (full_forward_all_layers's convention);
        # input_ids (no image patches) indexes the same token at position txt_rel + 1 (post-BOS)
        txt_rel = [r - 1 - num_patches for r in txt_rows]
        return [processor.tokenizer.decode([input_ids[0, r + 1].item()]) for r in txt_rel]

    from PIL import Image
    token_labels_c = decode_labels(processor, desc_c, num_patches, txt_rows_c)
    token_labels_t = decode_labels(processor, desc_t, num_patches, txt_rows_t)
    env_c, _ = get_libero_env(task, cfg.model_family, resolution=256, bddl_path=CLEAN_BDDL, seed=BASE_SEED)
    obs = None
    for _ in range(CLEAN_EP + 1):
        obs = env_c.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env_c.step(get_libero_dummy_action(cfg.model_family))
    _, img_c = build_observation(obs, resize_size)
    display_clean = np.array(preprocess_like_policy(img_c, cfg.center_crop))
    env_c.close()

    env_t, _ = get_libero_env(task, cfg.model_family, resolution=256, bddl_path=POISON_BDDL, seed=BASE_SEED)
    obs = None
    for _ in range(TRIG_EP + 1):
        obs = env_t.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env_t.step(get_libero_dummy_action(cfg.model_family))
    _, img_t = build_observation(obs, resize_size)
    display_trig = np.array(preprocess_like_policy(img_t, cfg.center_crop))
    env_t.close()

    plot_pertoken(
        image_clean=display_clean, attn_primary_clean=text2img_clean, tokens_clean=token_labels_c,
        image_trigger=display_trig, attn_primary_trigger=text2img_trig, tokens_trigger=token_labels_t,
        title="GoBA text2img -- normalize each layer, then average (section 2's strongest ordering)",
        subtitle=f"clean: {desc_c!r}  |  trigger: {desc_t!r}",
        out_path=str(out_dir / "text2img_mean_normthenavg.png"),
    )
    print(f"[*] wrote {out_dir / 'text2img_mean_normthenavg.png'}")

    # ---- img2img: normalize each layer first, then average ----
    img2img_clean = row_normalize(A_clean[:, img_cols][:, :, img_cols]).mean(axis=0)  # [256, 256]
    img2img_trig = row_normalize(A_trig[:, img_cols][:, :, img_cols]).mean(axis=0)
    matrix_heatmap(img2img_clean, img2img_trig,
                    "GoBA img2img (query=image, key=image) -- normalize each layer, then average",
                    out_dir / "img2img_mean_normthenavg.png")

    # ---- merged (image+text)->(image+text): normalize each layer first, then average ----
    key_cols_c = sorted(set(img_cols) | set(txt_rows_c))
    key_cols_t = sorted(set(img_cols) | set(txt_rows_t))
    all_rows_c = sorted(set(img_cols) | set(txt_rows_c))
    all_rows_t = sorted(set(img_cols) | set(txt_rows_t))
    merged_clean = row_normalize(A_clean[:, all_rows_c][:, :, key_cols_c]).mean(axis=0)
    merged_trig = row_normalize(A_trig[:, all_rows_t][:, :, key_cols_t]).mean(axis=0)
    matrix_heatmap(merged_clean, merged_trig,
                    "GoBA merged (image+text)->(image+text) -- normalize each layer, then average",
                    out_dir / "merged_mean_normthenavg.png")

    del vla
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
