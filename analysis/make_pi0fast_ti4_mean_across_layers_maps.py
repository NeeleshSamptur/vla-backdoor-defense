#!/usr/bin/env python
"""Section 5k -- one clean/trigger episode pair for Pi0-Fast TI4, rendered as
mean-across-all-18-layers pictures (NOT the per-layer-per-file style already
built in pi0fast_ti4_ftt_separation_maps/, a different, already-finished
deliverable -- this script writes to a distinct output directory and does
not touch that one).

Mirrors analysis/EXPERIMENTAL_make_mean_across_layers_maps.py's presentation:
  - text2img: plot_pertoken heatmap-on-image, using row-normalize-then-average
    across all 18 layers (this project's headline ordering)
  - img2img: plain [256x256] attention-matrix heatmap, same averaging
  - merged (image-group and text-group): plain attention-matrix heatmaps

Reads results/pi0fast_ti4_fullbattery_extracted/*.npz -- the SAME fresh
(post-fix) extraction this session's full battery used, no new forward pass.
Picks task 0's first clean and first trigger episode (by seed) as the
example pair, matching GoBA's example-episode convention (task 0, first
available episode of each condition).

Isolated: new file, does not modify any extraction/scoring/plotting script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "analysis"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_pertoken_attention import plot_pertoken

DATA_DIR = DEFENSE / "results" / "pi0fast_ti4_fullbattery_extracted"
OUT_DIR = DEFENSE / "pi0fast_ti4_mean_across_layers_maps"
EPS = 1e-12
TASK_ID = 0


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob(f"t{TASK_ID}__*.npz"))
    clean_files = [f for f in files if "__clean.npz" in f.name]
    trig_files = [f for f in files if "__trigger.npz" in f.name]
    assert clean_files and trig_files, f"no task {TASK_ID} episodes in {DATA_DIR}"
    clean_fp, trig_fp = clean_files[0], trig_files[0]
    print(f"[*] clean episode: {clean_fp.name}")
    print(f"[*] trigger episode: {trig_fp.name}")

    dc = np.load(clean_fp, allow_pickle=False)
    dt = np.load(trig_fp, allow_pickle=False)

    # ---- text2img: normalize each layer, then average across all 18 ----
    t2i_c = row_normalize(dc["attn_text_image_layers"].astype(np.float64)).mean(axis=0)  # [n_desc, per_cam]
    t2i_t = row_normalize(dt["attn_text_image_layers"].astype(np.float64)).mean(axis=0)
    tok_c = json.loads(str(dc["tokens_json"]))
    tok_t = json.loads(str(dt["tokens_json"]))
    display_c = np.asarray(dc["display"])
    display_t = np.asarray(dt["display"])
    desc_c = str(dc["desc"])
    desc_t = str(dt["desc"])

    plot_pertoken(
        image_clean=display_c, attn_primary_clean=t2i_c, tokens_clean=tok_c,
        image_trigger=display_t, attn_primary_trigger=t2i_t, tokens_trigger=tok_t,
        title="Pi0-Fast TI4 text2img -- normalize each layer, then average (all 18 layers)",
        subtitle=f"clean: {desc_c!r}  |  trigger: {desc_t!r}",
        out_path=str(OUT_DIR / "text2img_mean_normthenavg.png"),
    )

    # ---- img2img: normalize each layer, then average ----
    i2i_c = row_normalize(dc["attn_img_img_layers"].astype(np.float64)).mean(axis=0)  # [per_cam, per_cam]
    i2i_t = row_normalize(dt["attn_img_img_layers"].astype(np.float64)).mean(axis=0)
    matrix_heatmap(i2i_c, i2i_t,
                    "Pi0-Fast TI4 img2img (primary camera, query=image, key=image) -- mean across 18 layers",
                    OUT_DIR / "img2img_mean_normthenavg.png")

    # ---- merged: image-group and text-group, normalize each layer, then average ----
    mimg_c = row_normalize(dc["merged_image_raw"].astype(np.float64)).mean(axis=0)  # [per_cam, K]
    mimg_t = row_normalize(dt["merged_image_raw"].astype(np.float64)).mean(axis=0)
    matrix_heatmap(mimg_c, mimg_t,
                    "Pi0-Fast TI4 merged, image-group (query=image, key=image+desc) -- mean across 18 layers",
                    OUT_DIR / "merged_image_group_mean_normthenavg.png")

    mtxt_c = row_normalize(dc["merged_text_raw"].astype(np.float64)).mean(axis=0)  # [n_desc, K]
    mtxt_t = row_normalize(dt["merged_text_raw"].astype(np.float64)).mean(axis=0)
    matrix_heatmap(mtxt_c, mtxt_t,
                    "Pi0-Fast TI4 merged, text-group (query=desc, key=image+desc) -- mean across 18 layers",
                    OUT_DIR / "merged_text_group_mean_normthenavg.png")

    summary = {
        "task_id": TASK_ID,
        "clean_episode": clean_fp.name, "trigger_episode": trig_fp.name,
        "clean_desc": desc_c, "trigger_desc": desc_t,
        "n_layers": int(dc["n_layers"]),
    }
    with open(OUT_DIR / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] wrote {OUT_DIR / 'selection_summary.json'}")
    print(f"[*] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
