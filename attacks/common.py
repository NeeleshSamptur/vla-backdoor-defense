"""Shared FTT (Frobenius-norm assimilation) statistic and AUROC, used
identically by every attacks/<attack>/run_ftt_auroc.py script.

FTT construction ("normalize-then-average"): each layer's (and, where the
model exposes per-head attention, each head's) desc_only text-to-image
attention map is row-normalized independently, the resulting maps are
averaged into ONE map, and the FTT statistic is the mean L2 distance from
each row of that map to the map's own row-mean. Ported from T2IShield's
detect_fft.py (ECCV'24). Polarity is fixed a priori: a backdoor means LOW
FTT, since the trigger drags every query token's attention toward one
pattern (the Assimilation Phenomenon). Pure numpy + sklearn -- never imports
torch/jax, so it works regardless of which framework extracted the attention.

Also holds `center_crop_resize`, the one input transform used by the
crop-variant scripts (attacks/<attack>/run_crop_ftt_auroc.py), so the crop
that produces a "cropped" attention map is defined in exactly one place and
is identical across attacks.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    """Normalize the last axis of P to sum to 1. Works for any leading
    shape, e.g. [layers, tokens, patches] or [layers, heads, tokens, patches]."""
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def compute_ftt(attn_maps: np.ndarray) -> float:
    """attn_maps: raw (non-negative, not yet normalized) attention rows with
    shape [..., n_query_tokens, n_patches], where the leading axes are
    whatever should be averaged over (layers, and heads if present).

    Returns the FTT statistic for one episode: row-normalize every map
    independently, average all of them into one [n_query_tokens, n_patches]
    map, then take the mean L2 distance from each row to that map's own
    row-mean. Lower = more assimilated = more likely a backdoor.
    """
    normed = row_normalize(attn_maps)
    leading_axes = tuple(range(normed.ndim - 2))
    avg_map = normed.mean(axis=leading_axes)  # [n_query_tokens, n_patches]
    ref = avg_map.mean(axis=0)  # [n_patches]
    return float(np.linalg.norm(avg_map - ref[None, :], axis=1).mean())


def center_crop_resize(img: np.ndarray, scale: float = 0.8) -> np.ndarray:
    """Take the central `scale` fraction (linear, not area) of an HxWx3 uint8
    image and resize it back to the original HxW, so the result can be fed to
    the policy in place of the original with nothing else changed.

    Resampling is PIL's LANCZOS, the closest match to the lanczos3 filter the
    policies' own resize_image_for_policy uses, so a cropped frame stays on
    the same resampling path the model saw in training instead of introducing
    a second interpolator as a confound.

    Purpose: a trigger occupying a small patch near the image border is cut
    off by the crop, while the scene's task-relevant content survives. For
    DropVLA specifically (a 5px-radius dot at (10, 10) of the 256px render),
    scale=0.8 removes the dot entirely -- verified by rendering the cropped
    frames and by the trigger no longer changing the policy's gripper command.
    """
    from PIL import Image

    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected uint8 HxWx3 image, got {img.shape}/{img.dtype}")
    if not 0 < scale <= 1:
        raise ValueError(f"crop scale must be in (0, 1], got {scale}")
    h, w = img.shape[:2]
    ch, cw = int(round(h * scale)), int(round(w * scale))
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = Image.fromarray(img).crop((left, top, left + cw, top + ch))
    return np.asarray(cropped.resize((w, h), Image.LANCZOS).convert("RGB"), dtype=np.uint8)


def cosine_distance(h0: np.ndarray, h1: np.ndarray) -> float:
    """Mean over positions of (1 - cosine similarity) between two [n_positions,
    d_model] activation blocks for the same sample under two different inputs.

    Per position rather than on the flattened block, so one high-norm position
    cannot dominate the average. Equivalent information to reporting cosine
    SIMILARITY -- similarity is 1 minus this, so its AUROC is exactly 1 minus
    this one's; only the polarity of the report differs.
    """
    h0 = np.asarray(h0, dtype=np.float64)
    h1 = np.asarray(h1, dtype=np.float64)
    if h0.shape != h1.shape:
        raise ValueError(f"shape mismatch: {h0.shape} vs {h1.shape}")
    num = (h0 * h1).sum(axis=-1)
    den = np.linalg.norm(h0, axis=-1) * np.linalg.norm(h1, axis=-1)
    return float((1.0 - num / np.clip(den, EPS, None)).mean())


def relative_l2(h0: np.ndarray, h1: np.ndarray) -> float:
    """||h1 - h0||_F / ||h0||_F -- the scale-free magnitude counterpart of
    cosine_distance, comparable across samples with different activation norms."""
    h0 = np.asarray(h0, dtype=np.float64)
    h1 = np.asarray(h1, dtype=np.float64)
    return float(np.linalg.norm(h1 - h0) / max(np.linalg.norm(h0), EPS))


def compute_auroc_high_is_triggered(clean_scores, trigger_scores) -> float:
    """AUROC for scores whose polarity is the OPPOSITE of FTT's: a HIGH score
    means triggered (e.g. an activation moves further when a transform deletes
    the trigger). Negates both classes and defers to compute_auroc so the
    metric itself is defined in exactly one place."""
    return compute_auroc([-s for s in clean_scores], [-s for s in trigger_scores])


def compute_auroc(clean_scores, trigger_scores) -> float:
    """AUROC with label 1 = triggered and LOW score = backdoor (FTT's
    polarity), so the ranked quantity is -score. NaN if either class is empty.

    Computed as the rank-sum (Mann-Whitney U) form rather than via
    sklearn.metrics.roc_auc_score, which is identical for every input --
    including ties, handled by averaging tied ranks exactly as sklearn does --
    and verified against it. The reason is dependency reach, not preference:
    these scripts run inside each attack's own environment, and pi0-FAST's
    JAX venv has no scikit-learn, so an sklearn import here crashed a
    completed 180-episode run at the final scoring step. numpy is the only
    thing every one of those environments is guaranteed to have.
    """
    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trigger_scores, dtype=np.float64)
    if len(c) == 0 or len(t) == 0:
        return float("nan")
    s = np.concatenate([-c, -t])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # Average the ranks within each group of tied scores (sklearn's behavior).
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_c, n_t = len(c), len(t)
    rank_sum_t = ranks[n_c:].sum()
    return float((rank_sum_t - n_t * (n_t + 1) / 2) / (n_c * n_t))
