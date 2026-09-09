#!/usr/bin/env python
"""ONE-OFF, ISOLATED visualization -- the DropVLA equivalent of
EXPERIMENTAL_make_mean_across_layers_maps.py (GoBA). For ONE example
clean/trigger episode pair (same closed-loop, lift-gated capture used
throughout this project's DropVLA report), render the "mean across all 32
layers" map for text2img (per camera, patch-projected heatmap via
plot_pertoken) and plain attention-matrix heatmaps for img2img (per camera)
and the merged (both cameras' image patches + desc tokens) combined block --
the same query/key configs reported in sections 9a/9e/9f of the artifact,
but as pictures instead of numbers.

Reuses adapters/dropvla/extract_text2img_ftt.py's model loading, rollout /
lift-detection, and prompt/token bookkeeping UNCHANGED, plus the
already-built both-cameras forward-pass helpers from
extract_text2img_layerwise_percam_ftt.py (text2img) and
extract_img2img_merged_percam_ftt.py (img2img/merged). Isolated: new file,
does not modify any extraction/scoring script.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "dropvla"))
sys.path.insert(0, str(DEFENSE / "analysis"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.libero.run_libero_eval import GenerateConfig, prepare_observation
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

from extract_text2img_ftt import (  # noqa: E402 -- reused unchanged
    NUM_STEPS_WAIT, LiftNotReached, _desc_token_row_indices,
    initialize_model, run_episode_capture,
)
from extract_text2img_layerwise_percam_ftt import (  # noqa: E402
    oft_text2img_rows_all_layers_both_cams,
)
from extract_img2img_merged_percam_ftt import full_attention_both_cams  # noqa: E402
from plot_pertoken_attention import plot_pertoken

EPS = 1e-12
TASK_SUITE = "libero_spatial"
TASK_ID = 0
SEED_BASE = 0
BASE_SEED = 42


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


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


def decode_desc_tokens(processor, desc):
    tok = processor.tokenizer
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    enc = tok(prompt, add_special_tokens=False)
    ids = enc["input_ids"]
    n_txt = len(ids) + 1
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    return [tok.decode([ids[r]]) for r in txt_rel]


def main(checkpoint: str):
    out_dir = DEFENSE / "attention_maps_dropvla_mean_across_layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed_everywhere(BASE_SEED)
    base_cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint, task_suite_name=TASK_SUITE,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, lora_rank=32,
        load_in_8bit=False, load_in_4bit=True, center_crop=False,
        num_open_loop_steps=8, num_steps_wait=NUM_STEPS_WAIT, env_img_res=256,
        use_backdoor_instruction=False, use_visual_backdoor=False,
        language_suffix="carefully", conditional_backdoor_on_lift=True,
        backdoor_activation_height_m=0.06, backdoor_activation_time_s=0.02,
        visual_backdoor_dot_x=10, visual_backdoor_dot_y=10, visual_backdoor_dot_radius=5,
        visual_backdoor_dot_alpha=255, visual_backdoor_dot_shape="circle",
        cover_wrist_lower_quarter=False, seed=BASE_SEED,
    )

    print(f"[*] loading {checkpoint}")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(base_cfg)
    resize_size = get_image_resize_size(base_cfg)

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    init_states = suite.get_task_init_states(TASK_ID)
    env, task_description = get_libero_env(task, base_cfg.model_family, resolution=base_cfg.env_img_res)

    try:
        episode_idx = SEED_BASE
        obs, t, *_ = run_episode_capture(
            base_cfg, env, task_description, model, action_head, proprio_projector,
            processor, resize_size, init_states[episode_idx])
        print(f"[*] activated @ t={t}")

        clean_cfg = replace(base_cfg, use_visual_backdoor=False, use_backdoor_instruction=False)
        trig_cfg = replace(base_cfg, use_visual_backdoor=True, use_backdoor_instruction=False)
        clean_observation, _ = prepare_observation(clean_cfg, obs, resize_size, backdoor_active=False)
        trig_observation, _ = prepare_observation(trig_cfg, obs, resize_size, backdoor_active=True)
        desc = task_description

        # ---- text2img, both cameras, all layers (head-averaged) ----
        rows_p_c, rows_w_c, num_patches = oft_text2img_rows_all_layers_both_cams(
            model, processor, proprio_projector, base_cfg, clean_observation, desc)
        rows_p_t, rows_w_t, _ = oft_text2img_rows_all_layers_both_cams(
            model, processor, proprio_projector, base_cfg, trig_observation, desc)

        # normalize each layer, then average -- same ordering as GoBA's section 8 map
        t2i_p_clean = row_normalize(rows_p_c).mean(axis=0)  # [n_desc, num_patches]
        t2i_p_trig = row_normalize(rows_p_t).mean(axis=0)
        t2i_w_clean = row_normalize(rows_w_c).mean(axis=0)
        t2i_w_trig = row_normalize(rows_w_t).mean(axis=0)

        token_labels = decode_desc_tokens(processor, desc)

        display_clean_primary = np.array(clean_observation["full_image"])
        display_trig_primary = np.array(trig_observation["full_image"])
        display_clean_wrist = np.array(clean_observation["wrist_image"])
        display_trig_wrist = np.array(trig_observation["wrist_image"])

        plot_pertoken(
            image_clean=display_clean_primary, attn_primary_clean=t2i_p_clean, tokens_clean=token_labels,
            image_trigger=display_trig_primary, attn_primary_trigger=t2i_p_trig, tokens_trigger=token_labels,
            title="DropVLA text2img, primary camera -- normalize each layer, then average",
            subtitle=f"task: {desc!r}  |  activated @ t={t}",
            out_path=str(out_dir / "text2img_primary_mean_normthenavg.png"),
        )
        plot_pertoken(
            image_clean=display_clean_wrist, attn_primary_clean=t2i_w_clean, tokens_clean=token_labels,
            image_trigger=display_trig_wrist, attn_primary_trigger=t2i_w_trig, tokens_trigger=token_labels,
            title="DropVLA text2img, wrist camera -- normalize each layer, then average",
            subtitle=f"task: {desc!r}  |  activated @ t={t}",
            out_path=str(out_dir / "text2img_wrist_mean_normthenavg.png"),
        )

        # ---- img2img (per camera) + merged combined, all layers ----
        pp_c, ww_c, comb_c, _, n_desc = full_attention_both_cams(
            model, processor, proprio_projector, base_cfg, clean_observation, desc)
        pp_t, ww_t, comb_t, _, _ = full_attention_both_cams(
            model, processor, proprio_projector, base_cfg, trig_observation, desc)

        img2img_p_clean = row_normalize(pp_c).mean(axis=0)
        img2img_p_trig = row_normalize(pp_t).mean(axis=0)
        img2img_w_clean = row_normalize(ww_c).mean(axis=0)
        img2img_w_trig = row_normalize(ww_t).mean(axis=0)
        merged_clean = row_normalize(comb_c).mean(axis=0)
        merged_trig = row_normalize(comb_t).mean(axis=0)

        matrix_heatmap(img2img_p_clean, img2img_p_trig,
                        "DropVLA img2img, primary camera -- normalize each layer, then average",
                        out_dir / "img2img_primary_mean_normthenavg.png")
        matrix_heatmap(img2img_w_clean, img2img_w_trig,
                        "DropVLA img2img, wrist camera -- normalize each layer, then average",
                        out_dir / "img2img_wrist_mean_normthenavg.png")
        matrix_heatmap(merged_clean, merged_trig,
                        "DropVLA merged (both cameras + text)->(both cameras + text) -- "
                        "normalize each layer, then average",
                        out_dir / "merged_mean_normthenavg.png")
    finally:
        env.close()

    del model, processor, proprio_projector
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    main(args.checkpoint)
