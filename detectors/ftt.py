"""FTT: text-to-image Frobenius-norm assimilation detector.

Ported from T2IShield's detect_fft.py (ECCV'24) and matches the formula in
your own BadVLA script, experiments/robot/libero/run_kl_vs_ftt_text2img.py
(archived at commit b69619a, "text2img_rows" + "ftt"):

    P        = each TEXT token's attention row, restricted to IMAGE columns
    P        = row-normalized (the slice no longer sums to 1 once other
               columns -- text-to-text, text-to-proprio -- are dropped)
    high_atm = mean row across text tokens (the "assimilated" pattern)
    FTT      = mean_i || P_i - high_atm ||_2

T2IShield's polarity: backdoor <=> LOW FTT (a trigger token drags every other
text token's attention toward the same pattern -- the "Assimilation
Phenomenon"). This module reports AUROC under both polarities so the sign is
verified against data, not assumed -- confirmed on BadVLA's real pixel trigger
(low=backdoor) in the bera prototype this repo replaces the exploratory parts
of; that polarity should hold here too, but re-check per attack, don't assume.

No torch/transformers dependency -- pure numpy/sklearn, exactly the
"detectors never import torch" rule this repo is built around.
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


def auroc_both_polarities(clean_scores, trig_scores) -> dict:
    """AUROC with label 1 = triggered, both sign conventions.

    T2IShield's convention is `low_is_backdoor`; report both because a port to
    a new architecture/attack should not assume the polarity transfers.
    """
    from sklearn.metrics import roc_auc_score

    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trig_scores, dtype=np.float64)
    y = np.concatenate([np.ones(len(t)), np.zeros(len(c))])
    s = np.concatenate([t, c])
    if len(np.unique(y)) < 2 or len(c) == 0 or len(t) == 0:
        return {"high_is_backdoor": float("nan"), "low_is_backdoor": float("nan")}
    return {
        "high_is_backdoor": float(roc_auc_score(y, s)),
        "low_is_backdoor": float(roc_auc_score(y, -s)),
    }
