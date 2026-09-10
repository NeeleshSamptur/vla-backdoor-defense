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


def compute_auroc(clean_scores, trigger_scores) -> float:
    """AUROC with label 1 = triggered, low FTT = backdoor (so -FTT is the
    score sklearn ranks). NaN if either class is empty."""
    from sklearn.metrics import roc_auc_score

    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trigger_scores, dtype=np.float64)
    if len(c) == 0 or len(t) == 0:
        return float("nan")
    y = np.concatenate([np.zeros(len(c)), np.ones(len(t))])
    s = np.concatenate([-c, -t])
    return float(roc_auc_score(y, s))
