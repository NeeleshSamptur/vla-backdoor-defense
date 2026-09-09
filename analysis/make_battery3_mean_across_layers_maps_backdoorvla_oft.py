#!/usr/bin/env python
"""Section 3k -- BackdoorVLA-OFT analogue of EXPERIMENTAL_make_mean_across_
layers_maps.py's style EXACTLY: one clean/trigger episode pair, THREE
pictures (text2img via plot_pertoken, img2img and merged as plain matrix
heatmaps), each the "mean across all 32 layers" construction (normalize
each layer first, then average those normalized layers into ONE map) --
NOT the per-layer-per-file style already built in
backdoorvla_oft_ftt_separation_maps/ (a different, already-finished
deliverable; this script does not touch that directory).

Fresh forward pass only (no cached .npz reused, per the zero-reuse
correction for this whole battery) -- reuses extract_battery3_fresh.py's
battery3_forward() unchanged for the actual attention capture, and
make_separation_maps_backdoorvla_oft.py's env/replay/display-image
conventions (TARGET_TASK_BDDL is this adapter's only physical scene; `seed`
is the init_state index directly; magic-prefix trigger text).

Fixed episode pair (mirrors GoBA's fixed TASK_ID=0/ep0 convention, not a
separation search): task_id=0 (first non-target instruction), clean seed=7
(first clean init-state index), trigger seed=17 (=7+n_seeds offset, first
trigger init-state index) -- this adapter's own established seed-offset
convention (cond_offset = {"clean": 0, "trigger": n_seeds}).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "backdoorvla_openvla_oft"))
sys.path.insert(0, str(DEFENSE / "analysis"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.openvla_utils import prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size

from extract_text2img_ftt import (
    BDDL_ROOT, INIT_ROOT, MAGIC_PREFIX, CLEAN_SUITE, TARGET_TASK_BDDL,
    Cfg, load_vla, load_init_states, NUM_STEPS_WAIT,
)
from extract_battery3_fresh import battery3_forward
from plot_pertoken_attention import plot_pertoken

CHECKPOINT = ("/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack/"
              "Text_Image_Attack/object_TI_4/15000--49999_chkpt")
OUT_DIR = DEFENSE / "attention_maps_backdoorvla_oft_mean_across_layers"
TASK_ID = 0
CLEAN_SEED = 7
TRIG_SEED = 17
EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def mean_across_layers(layers_raw: np.ndarray) -> np.ndarray:
    """normalize each layer, then average -- this project's 'strongest
    ordering' convention, matching EXPERIMENTAL_make_mean_across_layers_
    maps.py exactly."""
    return row_normalize(layers_raw).mean(axis=0)


def run_one(vla, processor, proprio_projector, cfg, resize_size, task_names, cond, seed):
    use_magic = (cond == "trigger")
    bddl_subdir = "libero_object_poisoned" if use_magic else "libero_object"
    bddl_file = os.path.join(BDDL_ROOT, bddl_subdir, f"{TARGET_TASK_BDDL}.bddl")
    init_states = load_init_states(os.path.join(INIT_ROOT, bddl_subdir))
    instruction = task_names[TASK_ID]
    prompt_desc = (MAGIC_PREFIX + instruction) if use_magic else instruction

    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=cfg.env_img_res,
                              camera_widths=cfg.env_img_res)
    env.seed(0)
    try:
        env.reset()
        obs = env.set_init_state(init_states[seed])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        observation, _ = prepare_observation(obs, resize_size)

        r = battery3_forward(vla, processor, proprio_projector, cfg, observation, prompt_desc)

        full = observation["full_image"].copy()
        wrist = observation["wrist_image"].copy()
        images = prepare_images_for_vla([full, wrist], cfg)
        display = np.array(images[0])
        return r, prompt_desc, display
    finally:
        env.close()


def decode_labels(processor, prompt_desc, txt_rel_desc):
    prompt = f"In: What action should the robot take to {prompt_desc.lower()}?\nOut:"
    tok = processor.tokenizer
    enc = tok(prompt, add_special_tokens=False)
    return [tok.decode([enc["input_ids"][r]]) for r in txt_rel_desc]


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(0)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    cfg = Cfg(pretrained_checkpoint=CHECKPOINT)
    print(f"[*] loading {CHECKPOINT}")
    processor, vla, proprio_projector = load_vla(CHECKPOINT, cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[CLEAN_SUITE]()
    task_names = [suite.get_task(i).language for i in range(1, 10)]

    r_c, desc_c, display_c = run_one(vla, processor, proprio_projector, cfg, resize_size, task_names, "clean", CLEAN_SEED)
    r_t, desc_t, display_t = run_one(vla, processor, proprio_projector, cfg, resize_size, task_names, "trigger", TRIG_SEED)

    tok_labels_c = decode_labels(processor, desc_c, r_c["txt_rel_desc"])
    tok_labels_t = decode_labels(processor, desc_t, r_t["txt_rel_desc"])

    # ---- text2img: desc-only rows, mean-across-layers ----
    def desc_rows(r):
        t2i_avg = r["t2i_perhead"].mean(axis=1)  # [L,1+n_txt,P]
        idx = [1 + i for i in r["txt_rel_desc"]]
        return t2i_avg[:, idx, :]

    text2img_clean = mean_across_layers(desc_rows(r_c))
    text2img_trig = mean_across_layers(desc_rows(r_t))
    plot_pertoken(
        image_clean=display_c, attn_primary_clean=text2img_clean, tokens_clean=tok_labels_c,
        image_trigger=display_t, attn_primary_trigger=text2img_trig, tokens_trigger=tok_labels_t,
        title="BackdoorVLA-OFT text2img -- mean across all 32 layers (normalize each layer, then average)",
        subtitle=f"clean: {desc_c!r}  |  trigger: {desc_t!r}",
        out_path=str(OUT_DIR / "text2img_mean_across_layers.png"),
    )

    # ---- img2img (primary camera): mean-across-layers ----
    img2img_clean = mean_across_layers(r_c["img2img_primary_raw"])
    img2img_trig = mean_across_layers(r_t["img2img_primary_raw"])
    matrix_heatmap(img2img_clean, img2img_trig,
                    "BackdoorVLA-OFT img2img (primary camera, query=image, key=image) -- mean across all 32 layers",
                    OUT_DIR / "img2img_mean_across_layers.png")

    # ---- merged (image+desc combined single query): mean-across-layers ----
    merged_clean = mean_across_layers(
        np.concatenate([r_c["merged_image_raw"], r_c["merged_text_raw"]], axis=1))
    merged_trig = mean_across_layers(
        np.concatenate([r_t["merged_image_raw"], r_t["merged_text_raw"]], axis=1))
    matrix_heatmap(merged_clean, merged_trig,
                    "BackdoorVLA-OFT merged (image+desc combined query) -- mean across all 32 layers",
                    OUT_DIR / "merged_mean_across_layers.png")

    del vla
    print(f"\n[*] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
