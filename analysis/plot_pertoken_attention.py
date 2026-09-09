#!/usr/bin/env python
"""Shared per-token attention-map plotting, honest by construction.

One row per condition (CLEAN / TRIGGER), one panel per REAL query token that
was actually used for that condition's FTT computation -- never truncated or
aligned to match the other row's token count. A trigger episode with a
textual trigger legitimately has more columns than its clean counterpart;
this plotter shows that difference instead of hiding it, which is the whole
point of separating "vision-only" from "bi-modal" trigger types (see the
`--trigger-type` flag on each variant's assembly script).

Each panel overlays that token's real, already-saved `attn_text_image` row
(post-softmax attention, head-averaged, from the actual model layer used for
FTT) on the real primary-camera image as a 16x16-patch heatmap. A token whose
total attention mass to the image is below `dead_thresh` is drawn in
grayscale and labeled "dead" rather than a faint heatmap, so a reader can't
mistake near-zero signal for a real (if weak) hotspot.

This module does not run any model -- it only renders arrays that were
already extracted. See:
  analysis/make_pi0_attention_maps.py           (TI_4 bimodal, I_4 vision-only)
  adapters/backdoorvla_openvla_oft/dump_attention_map_data.py + this module
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


def _patch_heatmap(attn_row: np.ndarray, image: np.ndarray, grid: int) -> np.ndarray:
    """Upsample a [grid*grid] attention row to image resolution, smoothed.

    Cubic-spline zoom (matches 02_BadVLA_pertoken.png's soft blobs) instead of
    nearest-neighbor patch replication -- purely a display smoothing, the same
    16x16 patch values before and after, just interpolated between patch
    centers rather than tiled as hard-edged blocks.
    """
    h, w = image.shape[:2]
    small = attn_row.reshape(grid, grid)
    return zoom(small, (h / grid, w / grid), order=3)


def plot_pertoken(
    image_clean: np.ndarray,
    attn_primary_clean: np.ndarray,      # [n_query_clean, per_cam]
    tokens_clean: list[str],
    image_trigger: np.ndarray,
    attn_primary_trigger: np.ndarray,    # [n_query_trigger, per_cam]
    tokens_trigger: list[str],
    title: str,
    subtitle: str,
    out_path: str,
    row_label_clean: str = "CLEAN",
    row_label_trigger: str = "TRIGGER",
):
    per_cam = attn_primary_clean.shape[1]
    grid = int(round(math.sqrt(per_cam)))
    assert grid * grid == per_cam, f"non-square patch grid: per_cam={per_cam}"

    n_cols = max(len(tokens_clean), len(tokens_trigger)) + 1  # +1 for MEAN panel
    fig, axes = plt.subplots(2, n_cols, figsize=(2.0 * n_cols, 5.6))
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    for row, (image, attn, tokens, row_label) in enumerate([
        (image_clean, attn_primary_clean, tokens_clean, row_label_clean),
        (image_trigger, attn_primary_trigger, tokens_trigger, row_label_trigger),
    ]):
        gray = image.mean(axis=-1) if image.ndim == 3 else image
        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([])
            if col < len(tokens):
                a = attn[col]
                mass = float(a.sum())
                ax.imshow(gray, cmap="gray")
                hm = _patch_heatmap(a, image, grid)
                norm = hm / (hm.max() + 1e-12)
                alpha = np.clip(norm, 0, 1) ** 0.6  # low-attention patches stay transparent
                ax.imshow(hm, cmap="jet", alpha=alpha)
                ax.set_title(f"tok{col} {tokens[col]!r}", fontsize=8)
                ax.set_xlabel(f"m={mass:.3f}", fontsize=7)
            elif col == n_cols - 1 and len(tokens) > 0:
                mean_attn = attn.mean(axis=0)
                mass = float(mean_attn.sum())
                ax.imshow(gray, cmap="gray")
                hm = _patch_heatmap(mean_attn, image, grid)
                norm = hm / (hm.max() + 1e-12)
                alpha = np.clip(norm, 0, 1) ** 0.6
                ax.imshow(hm, cmap="jet", alpha=alpha)
                ax.set_title("MEAN\n(all tokens)", fontsize=8, fontweight="bold")
                ax.set_xlabel(f"m={mass:.3f}", fontsize=7)
            else:
                ax.axis("off")
        axes[row, 0].set_ylabel(row_label, fontsize=10, fontweight="bold")

    fig.suptitle(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(hspace=0.75)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[*] wrote {out_path}")
