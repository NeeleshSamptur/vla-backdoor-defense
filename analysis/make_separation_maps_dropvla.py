#!/usr/bin/env python
"""ONE-OFF, ISOLATED visualization -- render the two DropVLA "max separation"
and "min separation" episode cases (per-episode FTT score gap, Flatten
method, normalize-then-average, desc-only text->image attention, PRIMARY
camera -- the project's headline statistic) as one combined 4-row figure
each, same layout as make_separation_maps_goba.py:

  row 1: CLEAN,   token-wise (one panel per real desc token + MEAN panel)
  row 2: TRIGGER, token-wise (same)
  row 3: CLEAN,   layer-wise (32 panels, one per model layer L0..L31)
  row 4: TRIGGER, layer-wise (same, 32 panels)

Reuses adapters/dropvla/extract_text2img_ftt.py (model loading, closed-loop
lift-gated rollout capture, prompt/token bookkeeping) and
adapters/dropvla/extract_text2img_layerwise_percam_ftt.py's
oft_text2img_rows_all_layers_both_cams UNCHANGED, following
analysis/EXPERIMENTAL_make_dropvla_mean_across_layers_maps.py's main()
convention (init_states[episode_idx] direct indexed access -- no sequential
replay needed, unlike GoBA). FTT scoring reuses
analysis/score_dropvla_layerwise_percam.py's row_normalize /
flatten_normalize_then_average unchanged. Token/layer panel rendering
reuses analysis/plot_pertoken_attention.py's _patch_heatmap helper unchanged.

DropVLA task suite is libero_spatial. Checkpoint confirmed by cross-
referencing results/dropvla_text2img_layerwise_percam/*.npz filenames
(ckpt tag "..._v5p00_8b795131_...") against
adapters/dropvla/extract_text2img_layerwise_percam_ftt.py's
`ckpt_tag = f"{raw_tag[:40]}_{md5(checkpoint)[:8]}"` naming convention and
analysis/score_dropvla_content_words_percam.py's hardcoded Cfg checkpoint --
md5("/home/grads/nsamptur/vla_bkd_def/DropVLA/runs/openvla-7b+libero_spatial_
no_noops_v5p00carefully+b8+lr-0.0003+lora-r32+dropout-0.0--seed42--paper")[:8]
== "8b795131", confirmed by direct computation.

Isolated: new file, does not modify any extraction/scoring/plotting script.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "dropvla"))
sys.path.insert(0, str(DEFENSE / "analysis"))
sys.path.insert(0, "/home/grads/nsamptur/vla_bkd_def/DropVLA")

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
from plot_pertoken_attention import _patch_heatmap, plot_pertoken
from score_dropvla_layerwise_percam import row_normalize, flatten_normalize_then_average

TASK_SUITE = "libero_spatial"
BASE_SEED = 42
CHECKPOINT = ("/home/grads/nsamptur/vla_bkd_def/DropVLA/runs/"
              "openvla-7b+libero_spatial_no_noops_v5p00carefully+b8+lr-0.0003"
              "+lora-r32+dropout-0.0--seed42--paper")

OUT_DIR = Path("/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/goba_dropvla_ftt_separation_maps")

CASES = [
    dict(name="dropvla_max_separation.png",
         title="DropVLA -- MAX separation (task 3, episode 3, primary camera)",
         task_id=3, episode_idx=3,
         ref_clean=0.04844, ref_trig=0.04212),
    dict(name="dropvla_min_separation.png",
         title="DropVLA -- MIN separation (task 2, episode 0, primary camera)",
         task_id=2, episode_idx=0,
         ref_clean=0.02971, ref_trig=0.02968),
]


def draw_token_panel(ax, image, attn_row, label, dead_thresh=1e-3):
    """Mirrors plot_pertoken_attention.plot_pertoken's per-token panel body."""
    gray = image.mean(axis=-1) if image.ndim == 3 else image
    grid = int(round(np.sqrt(attn_row.shape[0])))
    ax.set_xticks([]); ax.set_yticks([])
    mass = float(attn_row.sum())
    ax.imshow(gray, cmap="gray")
    if mass < dead_thresh:
        ax.text(0.5, 0.5, "dead", color="0.6", ha="center", va="center",
                 transform=ax.transAxes, fontsize=8)
    else:
        hm = _patch_heatmap(attn_row, image, grid)
        norm = hm / (hm.max() + 1e-12)
        alpha = np.clip(norm, 0, 1) ** 0.6
        ax.imshow(hm, cmap="jet", alpha=alpha)
    ax.set_title(label, fontsize=7)
    ax.set_xlabel(f"m={mass:.3f}" if mass >= dead_thresh else "dead",
                  fontsize=6.5, color=("0.6" if mass < dead_thresh else "black"))


def decode_desc_tokens(processor, desc):
    tok = processor.tokenizer
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    enc = tok(prompt, add_special_tokens=False)
    ids = enc["input_ids"]
    n_txt = len(ids) + 1
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    return [tok.decode([ids[r]]) for r in txt_rel]


def render_case(case, image_c, image_t, rows_c_layers, rows_t_layers, tok_labels,
                 desc, t_activated, fresh_clean_score, fresh_trig_score):
    """One file PER LAYER (not one combined figure): each file is a plain
    2-row plot_pertoken render (CLEAN/TRIGGER, one panel per real desc token
    + MEAN) using THAT SINGLE LAYER's own row-normalized attention -- no
    averaging across layers, no averaging across tokens. 32 files per case."""
    n_layers = rows_c_layers.shape[0]
    case_name = case["name"].replace(".png", "")
    case_dir = OUT_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []
    for l in range(n_layers):
        out_path = case_dir / f"L{l:02d}.png"
        plot_pertoken(
            image_clean=image_c, attn_primary_clean=rows_c_layers[l], tokens_clean=tok_labels,
            image_trigger=image_t, attn_primary_trigger=rows_t_layers[l], tokens_trigger=tok_labels,
            title=f"{case['title']} -- Layer {l}",
            subtitle=(f"task: {desc!r}  |  activated @ t={t_activated}\n"
                      f"FTT score (Flatten, normalize-then-average, desc-only text->image, "
                      f"primary camera, passed to AUROC): clean={fresh_clean_score:.5f}  "
                      f"trigger={fresh_trig_score:.5f}  Δ={fresh_clean_score - fresh_trig_score:.5f}"),
            out_path=str(out_path),
            row_label_clean=f"CLEAN\nFTT={fresh_clean_score:.5f}",
            row_label_trigger=f"TRIGGER\nFTT={fresh_trig_score:.5f}",
        )
        out_paths.append(out_path)
    return out_paths


def verify_no_trigger_in_clean(clean_image, trig_image, dot_x, dot_y, dot_radius):
    """Pixel-level proof the visual trigger dot is absent from the CLEAN
    frame: the dot is drawn fully opaque, solid red. Compute a "redness"
    score (R - max(G,B)) in a neighborhood around the dot's coordinates for
    both frames -- clean must show no red spike there, trigger must."""
    def redness(img):
        y0, y1 = max(0, dot_y - dot_radius - 2), dot_y + dot_radius + 3
        x0, x1 = max(0, dot_x - dot_radius - 2), dot_x + dot_radius + 3
        patch = img[y0:y1, x0:x1].astype(np.float64)
        r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
        return float((r - np.maximum(g, b)).max())

    clean_redness = redness(clean_image)
    trig_redness = redness(trig_image)
    print(f"[*] trigger-dot pixel check @ ({dot_x},{dot_y}) r={dot_radius}: "
          f"CLEAN redness={clean_redness:.1f}  TRIGGER redness={trig_redness:.1f} "
          f"(expect CLEAN near 0, TRIGGER large positive)")
    assert clean_redness < 40, (
        f"CLEAN frame shows a red spike ({clean_redness:.1f}) at the trigger-dot "
        f"location -- the visual trigger may be leaking into the clean frame!")
    assert trig_redness > 100, (
        f"TRIGGER frame does NOT show the expected red spike ({trig_redness:.1f}) at "
        f"the dot location -- the trigger may not have been applied as expected.")
    print("[*] confirmed: CLEAN frame has no trigger-dot artifact; TRIGGER frame does.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed_everywhere(BASE_SEED)
    base_cfg = GenerateConfig(
        pretrained_checkpoint=CHECKPOINT, task_suite_name=TASK_SUITE,
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

    print(f"[*] loading {CHECKPOINT}")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(base_cfg)
    resize_size = get_image_resize_size(base_cfg)

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()

    mismatches = []
    for case in CASES:
        task = suite.get_task(case["task_id"])
        init_states = suite.get_task_init_states(case["task_id"])
        env, task_description = get_libero_env(task, base_cfg.model_family, resolution=base_cfg.env_img_res)
        print(f"\n[*] {case['name']}: task_id={case['task_id']} episode_idx={case['episode_idx']}")
        try:
            episode_idx = case["episode_idx"]
            obs, t, *_ = run_episode_capture(
                base_cfg, env, task_description, model, action_head, proprio_projector,
                processor, resize_size, init_states[episode_idx])
            print(f"[*] activated @ t={t}")

            clean_cfg = replace(base_cfg, use_visual_backdoor=False, use_backdoor_instruction=False)
            trig_cfg = replace(base_cfg, use_visual_backdoor=True, use_backdoor_instruction=False)
            clean_observation, _ = prepare_observation(clean_cfg, obs, resize_size, backdoor_active=False)
            trig_observation, _ = prepare_observation(trig_cfg, obs, resize_size, backdoor_active=True)
            desc = task_description

            rows_p_c, _rows_w_c, num_patches = oft_text2img_rows_all_layers_both_cams(
                model, processor, proprio_projector, base_cfg, clean_observation, desc)
            rows_p_t, _rows_w_t, _ = oft_text2img_rows_all_layers_both_cams(
                model, processor, proprio_projector, base_cfg, trig_observation, desc)

            fresh_clean = flatten_normalize_then_average(rows_p_c)
            fresh_trig = flatten_normalize_then_average(rows_p_t)
            print(f"[*] fresh clean_score={fresh_clean:.5f}  trig_score={fresh_trig:.5f}  "
                  f"(reference: clean={case['ref_clean']:.5f} trig={case['ref_trig']:.5f})")
            if abs(fresh_clean - case["ref_clean"]) > 5e-4 or abs(fresh_trig - case["ref_trig"]) > 5e-4:
                msg = (f"MISMATCH for {case['name']}: fresh=({fresh_clean:.5f},{fresh_trig:.5f}) "
                       f"vs reference=({case['ref_clean']:.5f},{case['ref_trig']:.5f})")
                print(f"[!] {msg}")
                mismatches.append(msg)
            else:
                print(f"[*] matches reference within tolerance for {case['name']}")

            token_labels = decode_desc_tokens(processor, desc)
            display_clean_primary = np.array(clean_observation["full_image"])
            display_trig_primary = np.array(trig_observation["full_image"])

            verify_no_trigger_in_clean(
                display_clean_primary, display_trig_primary,
                base_cfg.visual_backdoor_dot_x, base_cfg.visual_backdoor_dot_y,
                base_cfg.visual_backdoor_dot_radius)

            out_paths = render_case(case, display_clean_primary, display_trig_primary,
                                     rows_p_c, rows_p_t, token_labels, desc, t,
                                     fresh_clean, fresh_trig)
            total_bytes = sum(p.stat().st_size for p in out_paths)
            print(f"[*] wrote {len(out_paths)} per-layer files to {out_paths[0].parent} "
                  f"({total_bytes} bytes total)")
        finally:
            env.close()

    del model, processor, proprio_projector
    print("\n[*] done.")
    if mismatches:
        print("[!] SUMMARY OF MISMATCHES:")
        for m in mismatches:
            print(f"    {m}")
    else:
        print("[*] all fresh FTT scores matched the supplied reference values.")


if __name__ == "__main__":
    main()
