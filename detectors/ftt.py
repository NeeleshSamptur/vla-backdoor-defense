"""FTT: text-to-image Frobenius-norm assimilation detector.

Ported from T2IShield's detect_fft.py (ECCV'24):

    P        = each text token's attention row, restricted to image columns
    P        = row-normalized (the slice no longer sums to 1 once text-to-text
               and text-to-proprio columns are dropped)
    high_atm = mean row across text tokens, i.e. the assimilated pattern
    FTT      = mean_i || P_i - high_atm ||_2

Polarity is fixed a priori to T2IShield's: a backdoor means LOW FTT, since the
trigger drags every text token's attention toward one pattern (the Assimilation
Phenomenon). Pure numpy/sklearn -- detectors never import torch.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


def ftt_score(attn_text_image: np.ndarray) -> float:
    """The FTT statistic for one sample. Lower = more assimilated."""
    P = row_normalize(attn_text_image)
    high_atm = P.mean(axis=0)
    return float(np.linalg.norm(P - high_atm[None, :], axis=1).mean())


def _row_normalize_layers(P: np.ndarray) -> np.ndarray:
    """Like row_normalize, but P is [n_layers, n_tokens, n_patches] -- each
    (layer, token) row is normalized independently over the patches axis."""
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def grand_mean_pattern(attn_layers: np.ndarray) -> np.ndarray:
    """The reference pattern for ftt_score_layerwise: average each layer's
    rows over tokens, then average those per-layer averages over layers.

    attn_layers: [n_layers, n_tokens, n_patches]. Returns [n_patches].
    """
    P = _row_normalize_layers(attn_layers)
    per_layer_mean = P.mean(axis=1)  # [L, patches] -- mean over tokens, per layer
    return per_layer_mean.mean(axis=0)  # [patches] -- mean over layers


def ftt_score_layerwise(attn_layers: np.ndarray) -> float:
    """Per-sample layer-aware FTT variant.

    Unlike ftt_score (which needs attention already collapsed to one 2D map,
    e.g. by averaging over layers before this point), this takes each layer's
    own per-token rows, compares them against ONE reference pattern shared by
    all layers -- the mean, over layers, of each layer's own mean-over-tokens
    pattern -- then averages the per-token deviation first over tokens within
    a layer, then over layers.

    attn_layers: [n_layers, n_tokens, n_patches]. Lower = more assimilated.
    """
    P = _row_normalize_layers(attn_layers)
    ref = grand_mean_pattern(attn_layers)
    per_token = np.linalg.norm(P - ref[None, None, :], axis=-1)  # [L, T]
    per_layer = per_token.mean(axis=1)  # [L] -- average over tokens
    return float(per_layer.mean())  # average over layers


def ftt_score_layerwise_fixed(attn_layers: np.ndarray, reference: np.ndarray) -> float:
    """Same as ftt_score_layerwise, but scored against a REFERENCE pattern
    computed elsewhere (e.g. pooled from clean samples via
    fixed_reference_from_clean) instead of from this sample's own layers.

    attn_layers: [n_layers, n_tokens, n_patches]. reference: [n_patches].
    """
    P = _row_normalize_layers(attn_layers)
    per_token = np.linalg.norm(P - reference[None, None, :], axis=-1)  # [L, T]
    per_layer = per_token.mean(axis=1)
    return float(per_layer.mean())


def fixed_reference_from_clean(clean_attn_layers: list[np.ndarray]) -> np.ndarray:
    """Pool a set of clean samples' grand_mean_pattern into one fixed
    reference, for use with ftt_score_layerwise_fixed."""
    patterns = np.stack([grand_mean_pattern(a) for a in clean_attn_layers])
    return patterns.mean(axis=0)


def _row_normalize_time_layers(P: np.ndarray) -> np.ndarray:
    """Like _row_normalize_layers, but P is [n_timesteps, n_layers, n_tokens,
    n_patches] -- each (timestep, layer, token) row is normalized
    independently over the patches axis."""
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def temporal_layer_reference(attn_time_layers: np.ndarray) -> np.ndarray:
    """Each LAYER's own expected attention pattern, averaged over every
    timestep AND every token in the episode -- one reference per layer, kept
    separate (unlike grand_mean_pattern, which collapses layers into ONE
    shared reference).

    attn_time_layers: [n_timesteps, n_layers, n_tokens, n_patches].
    Returns [n_layers, n_patches].
    """
    P = _row_normalize_time_layers(attn_time_layers)
    return P.mean(axis=(0, 2))  # average over (timestep, token), keep layer


def ftt_score_temporal_layerwise(attn_time_layers: np.ndarray) -> np.ndarray:
    """Per-timestep FTT variant: at each instant, does each layer's attention
    look different from how that SAME layer behaves on average across the
    whole episode?

    Unlike ftt_score_layerwise (whose reference is one pattern shared by all
    layers, averaged over layers), here each layer keeps its own reference --
    computed by averaging that layer's rows over time and tokens via
    temporal_layer_reference -- so a layer is only ever compared to its own
    typical behavior, never to another layer's.

    attn_time_layers: [n_timesteps, n_layers, n_tokens, n_patches].
    Returns an array of length n_timesteps: score[t] = mean over layers of
    (mean over tokens of ||P[t, l, tok] - ref[l]||). Lower = more assimilated
    to that layer's own norm, same polarity convention as ftt_score.
    """
    P = _row_normalize_time_layers(attn_time_layers)
    ref = temporal_layer_reference(attn_time_layers)  # [L, N]
    diff = np.linalg.norm(P - ref[None, :, None, :], axis=-1)  # [Ttime, L, tok]
    per_layer = diff.mean(axis=2)  # [Ttime, L] -- average over tokens
    return per_layer.mean(axis=1)  # [Ttime] -- average over layers


def combined_layer_time_reference(attn_time_layers: np.ndarray) -> np.ndarray:
    """ONE shared reference pattern, built by averaging over layers AND
    timesteps together (unlike temporal_layer_reference, which keeps each
    layer's reference separate, and grand_mean_pattern, which only ever saw
    one timestep). Caller pre-slices which layers to include (e.g. first 16).

    attn_time_layers: [n_timesteps, n_layers, n_tokens, n_patches].
    Returns [n_patches].
    """
    P = _row_normalize_time_layers(attn_time_layers)
    per_timestep = P.mean(axis=(1, 2))  # average over (layer, token), keep time -> [Ttime, N]
    return per_timestep.mean(axis=0)    # average over time -> [N]


def ftt_score_combined_layer_time(attn_time_layers: np.ndarray) -> float:
    """One shared reference (averaged over layers AND timesteps together, via
    combined_layer_time_reference), then every individual raw (timestep,
    layer, token) row is compared against that ONE reference -- as opposed to
    ftt_score_temporal_layerwise (per-layer references, per-timestep output)
    or ftt_score_layerwise (per-layer grand mean, single timestep only).

    attn_time_layers: [n_timesteps, n_layers(pre-sliced), n_tokens, n_patches].
    Returns a single scalar for the whole episode.
    """
    P = _row_normalize_time_layers(attn_time_layers)
    ref = combined_layer_time_reference(attn_time_layers)  # [N]
    diff = np.linalg.norm(P - ref[None, None, None, :], axis=-1)  # [Ttime, L, tok]
    per_layer = diff.mean(axis=2)   # average over tokens -> [Ttime, L]
    per_time = per_layer.mean(axis=1)  # average over layers -> [Ttime]
    return float(per_time.mean())   # average over time -> scalar


def auroc(clean_scores, trig_scores) -> float:
    """AUROC with label 1 = triggered and low FTT = backdoor.

    sklearn's roc_auc_score treats higher score as class 1, so we pass -FTT.
    """
    from sklearn.metrics import roc_auc_score

    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trig_scores, dtype=np.float64)
    y = np.concatenate([np.ones(len(t)), np.zeros(len(c))])
    s = np.concatenate([t, c])
    if len(np.unique(y)) < 2 or len(c) == 0 or len(t) == 0:
        return float("nan")
    return float(roc_auc_score(y, -s))
