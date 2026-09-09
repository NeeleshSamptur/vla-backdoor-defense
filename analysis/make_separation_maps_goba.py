#!/usr/bin/env python
"""ONE-OFF, ISOLATED visualization -- render the two GoBA "max separation"
and "min separation" episode pairs (per-episode FTT score gap, Flatten
method, normalize-then-average, desc-only text->image attention -- the
project's headline statistic) as one combined 4-row figure each:

  row 1: CLEAN,   token-wise (one panel per real desc token + MEAN panel)
  row 2: TRIGGER, token-wise (same)
  row 3: CLEAN,   layer-wise (32 panels, one per model layer L0..L31, each
                  that single layer's own mean-across-desc-tokens row)
  row 4: TRIGGER, layer-wise (same, 32 panels)

Reuses adapters/goba/extract_text2img_ftt.py (env/model loading, prompt
bookkeeping) and adapters/goba/extract_img2img_merged_ftt.py
(full_forward_all_layers) UNCHANGED via
analysis/EXPERIMENTAL_make_mean_across_layers_maps.py's run_one() convention
(env.reset() called ep_idx+1 times -- the confirmed replay convention, no
offset math needed). FTT scoring reuses
analysis/score_goba_flatten_normalize_order.py's score_normalize_then_average
unchanged. Token/layer panel rendering reuses
analysis/plot_pertoken_attention.py's _patch_heatmap helper unchanged; the
per-panel draw logic below mirrors plot_pertoken's own panel code exactly
(grayscale base + jet overlay + m=X.XXX caption) but is written fresh here
because plot_pertoken renders its own complete 2-row figure and this script
needs a custom 4-row composite instead.

Isolated: new file, does not modify any extraction/scoring/plotting script.
"""
from __future__ import annotations

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
from PIL import Image

from libero.libero import benchmark

from extract_text2img_ftt import (
    CLEAN_BDDL, POISON_BDDL, Cfg, get_libero_env, get_libero_dummy_action,
    build_observation, load_vla_for_attention, preprocess_like_policy,
)
from extract_img2img_merged_ftt import full_forward_all_layers
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size
from plot_pertoken_attention import _patch_heatmap, plot_pertoken
from score_goba_flatten_normalize_order import row_normalize, score_normalize_then_average

NUM_STEPS_WAIT = 10
CHECKPOINT = ("/home/grads/nsamptur/vla_bkd_def/GoBA_attack/exp/"
              "openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug")
TASK_SUITE = "libero_object"
BASE_SEED = 7

OUT_DIR = Path("/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/goba_dropvla_ftt_separation_maps")

CASES = [
    dict(name="goba_max_separation.png", title="GoBA -- MAX separation (task 5, clean ep6 vs trigger ep19)",
         task_id=5, clean_ep=6, trig_ep=19,
         ref_clean=0.03407, ref_trig=0.02032),
    dict(name="goba_min_separation.png", title="GoBA -- MIN separation (task 2, clean ep4 vs trigger ep10)",
         task_id=2, clean_ep=4, trig_ep=10,
         ref_clean=0.02115, ref_trig=0.02114),
]


def run_one(vla, processor, cfg, resize_size, bddl_dir, task, ep_idx):
    """Same convention as EXPERIMENTAL_make_mean_across_layers_maps.run_one,
    but also returns the display image (the exact preprocessed frame the
    forward pass patchified) so the caller doesn't need a second env replay
    just to re-derive it."""
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
        display = np.array(preprocess_like_policy(img, cfg.center_crop))
        return A_np, num_patches, img_cols, txt_rows, desc, display
    finally:
        env.close()


def decode_labels(processor, desc, num_patches, txt_rows):
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    img_dummy = Image.new("RGB", (224, 224))
    inputs = processor(prompt, img_dummy)
    input_ids = inputs["input_ids"]
    txt_rel = [r - 1 - num_patches for r in txt_rows]
    return [processor.tokenizer.decode([input_ids[0, r + 1].item()]) for r in txt_rel]


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


def render_case(case, A_c, num_patches, img_cols, txt_rows_c, desc_c, display_c, tok_labels_c,
                 A_t, txt_rows_t, desc_t, display_t, tok_labels_t,
                 fresh_clean_score, fresh_trig_score):
    """One file PER LAYER (not one combined figure): each file is a plain
    2-row plot_pertoken render (CLEAN/TRIGGER, one panel per real desc token
    + MEAN) using THAT SINGLE LAYER's own row-normalized attention -- no
    averaging across layers, no averaging across tokens. 32 files per case."""
    n_layers = A_c.shape[0]
    case_dir = OUT_DIR / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)

    t2i_c_layers = row_normalize(A_c[:, txt_rows_c][:, :, img_cols])  # [L, n_desc, P]
    t2i_t_layers = row_normalize(A_t[:, txt_rows_t][:, :, img_cols])

    out_paths = []
    for l in range(n_layers):
        out_path = case_dir / f"L{l:02d}.png"
        plot_pertoken(
            image_clean=display_c, attn_primary_clean=t2i_c_layers[l], tokens_clean=tok_labels_c,
            image_trigger=display_t, attn_primary_trigger=t2i_t_layers[l], tokens_trigger=tok_labels_t,
            title=f"{case['title']} -- Layer {l}",
            subtitle=(f"clean: {desc_c!r}  |  trigger: {desc_t!r}\n"
                      f"FTT score (Flatten, normalize-then-average, desc-only text->image, "
                      f"passed to AUROC): clean={fresh_clean_score:.5f}  "
                      f"trigger={fresh_trig_score:.5f}  Δ={fresh_clean_score - fresh_trig_score:.5f}"),
            out_path=str(out_path),
            row_label_clean=f"CLEAN\nFTT={fresh_clean_score:.5f}",
            row_label_trigger=f"TRIGGER\nFTT={fresh_trig_score:.5f}",
        )
        out_paths.append(out_path)
    return out_paths


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = Cfg(pretrained_checkpoint=CHECKPOINT, unnorm_key=TASK_SUITE)
    print(f"[*] loading {CHECKPOINT}")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()

    mismatches = []
    for case in CASES:
        task = suite.get_task(case["task_id"])
        print(f"\n[*] {case['name']}: task_id={case['task_id']} clean_ep={case['clean_ep']} "
              f"trig_ep={case['trig_ep']}")

        A_c, num_patches, img_cols, txt_rows_c, desc_c, display_c = run_one(
            vla, processor, cfg, resize_size, CLEAN_BDDL, task, case["clean_ep"])
        A_t, _, _, txt_rows_t, desc_t, display_t = run_one(
            vla, processor, cfg, resize_size, POISON_BDDL, task, case["trig_ep"])

        tok_labels_c = decode_labels(processor, desc_c, num_patches, txt_rows_c)
        tok_labels_t = decode_labels(processor, desc_t, num_patches, txt_rows_t)

        clean_layers = A_c[:, txt_rows_c][:, :, img_cols]
        trig_layers = A_t[:, txt_rows_t][:, :, img_cols]
        fresh_clean = score_normalize_then_average(clean_layers)
        fresh_trig = score_normalize_then_average(trig_layers)

        print(f"[*] fresh clean_score={fresh_clean:.5f}  trig_score={fresh_trig:.5f}  "
              f"(reference: clean={case['ref_clean']:.5f} trig={case['ref_trig']:.5f})")
        if abs(fresh_clean - case["ref_clean"]) > 5e-4 or abs(fresh_trig - case["ref_trig"]) > 5e-4:
            msg = (f"MISMATCH for {case['name']}: fresh=({fresh_clean:.5f},{fresh_trig:.5f}) "
                   f"vs reference=({case['ref_clean']:.5f},{case['ref_trig']:.5f})")
            print(f"[!] {msg}")
            mismatches.append(msg)
        else:
            print(f"[*] matches reference within tolerance for {case['name']}")

        out_paths = render_case(case, A_c, num_patches, img_cols, txt_rows_c, desc_c, display_c,
                                 tok_labels_c, A_t, txt_rows_t, desc_t, display_t, tok_labels_t,
                                 fresh_clean, fresh_trig)
        total_bytes = sum(p.stat().st_size for p in out_paths)
        print(f"[*] wrote {len(out_paths)} per-layer files to {out_paths[0].parent} "
              f"({total_bytes} bytes total)")

    del vla
    print("\n[*] done.")
    if mismatches:
        print("[!] SUMMARY OF MISMATCHES:")
        for m in mismatches:
            print(f"    {m}")
    else:
        print("[*] all fresh FTT scores matched the supplied reference values.")


if __name__ == "__main__":
    main()
