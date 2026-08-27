"""Stage 2: temporal monitoring for triggers that appear partway through a rollout.

Stage 1 (see runners/run_detector.py) scores a single frame and catches
always-on triggers -- BadVLA's centered block, GoBA's physical object -- which
are present from frame 0. It cannot catch DropVLA, whose trigger only appears
after the policy has already grasped and lifted the object: frame 0 is
genuinely clean, so Stage 1 is *correctly* silent there.

This module handles that case. The key structural gift of a delayed trigger is
that the episode STARTS CLEAN, so the episode's own early frames are a
perfectly matched reference -- same scene, lighting, task, camera pose. No
external calibration set, no distribution-shift assumption.

Framing: not "is this frame poisoned" but "did this episode's statistic
depart from its own baseline and STAY departed". Natural variation over a
rollout wanders smoothly; a backdoor activation is a step change that
persists -- and it must persist, because releasing an object takes several
timesteps. The attack's own requirement is what makes it detectable.

IMPORTANT ASSUMPTION: self-normalization requires a clean prefix. That holds
for DropVLA. It does NOT hold for BadVLA/GoBA (trigger present from frame 0,
no clean prefix to normalize against) -- use Stage 1 for those. An adaptive
attacker could fire at frame 0 specifically to defeat this, at the cost of
having nothing grasped to drop.

Pure numpy -- no torch, no attack code, consistent with the detectors/ rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class TemporalConfig:
    # Frames at the start of the episode used as the self-reference.
    # A window, not a single frame: one frame is noisy (motion blur, a
    # specular highlight) and would poison the baseline.
    n_baseline: int = 8
    # Consecutive elevated frames required before alarming. This is the
    # false-alarm killer: a one-frame blip is filtered out, a real activation
    # cannot be, because the trigger latches on.
    k_persist: int = 3
    # Alarm threshold in robust z-units above the episode's own baseline.
    z_threshold: float = 4.0
    # Floor on the baseline scale, so a near-constant baseline doesn't make
    # the z-score explode on trivial noise.
    min_scale: float = 1e-6


def robust_baseline(scores: Sequence[float], n_baseline: int, min_scale: float = 1e-6):
    """Center/scale from the episode's own opening frames.

    Median and MAD rather than mean/std: a single outlier frame in the
    baseline window shouldn't drag the reference.
    """
    b = np.asarray(scores[:n_baseline], dtype=np.float64)
    if b.size == 0:
        raise ValueError("empty baseline window")
    center = float(np.median(b))
    mad = float(np.median(np.abs(b - center)))
    scale = max(1.4826 * mad, min_scale)  # 1.4826 -> MAD approximates std
    return center, scale


def relative_trace(scores: Sequence[float], cfg: TemporalConfig) -> np.ndarray:
    """Per-frame deviation from the episode's own baseline, in robust z-units.

    Sign convention: POSITIVE = more backdoor-like. FTT's polarity is
    "low = backdoor" (assimilation collapses the spread of attention rows),
    so the deviation is negated here. Downstream code can then always treat
    "higher is more suspicious" uniformly.
    """
    s = np.asarray(scores, dtype=np.float64)
    center, scale = robust_baseline(s, cfg.n_baseline, cfg.min_scale)
    return -(s - center) / scale


def first_alarm_frame(scores: Sequence[float], cfg: TemporalConfig) -> Optional[int]:
    """Index of the first frame where the persistence rule fires, else None.

    Requires `k_persist` CONSECUTIVE frames above threshold; reports the first
    frame of that run (the earliest evidence), which is what latency should be
    measured against.
    """
    z = relative_trace(scores, cfg)
    run = 0
    for i in range(cfg.n_baseline, len(z)):
        if z[i] > cfg.z_threshold:
            run += 1
            if run >= cfg.k_persist:
                return i - cfg.k_persist + 1
        else:
            run = 0
    return None


def episode_score(scores: Sequence[float], cfg: TemporalConfig) -> float:
    """One number per episode, for episode-level AUROC.

    The defender does not know which timestep the trigger appears, so it does
    not get to pick one: this takes the strongest sustained deviation anywhere
    after the baseline window. Using a k-run mean rather than a bare max keeps
    a single noisy frame from carrying an episode.
    """
    z = relative_trace(scores, cfg)
    tail = z[cfg.n_baseline:]
    if tail.size == 0:
        return float("-inf")
    if tail.size < cfg.k_persist:
        return float(tail.max())
    # best average over any k consecutive frames
    kernel = np.ones(cfg.k_persist) / cfg.k_persist
    return float(np.convolve(tail, kernel, mode="valid").max())


def detection_latency(scores: Sequence[float], activation_frame: int,
                      cfg: TemporalConfig) -> Optional[int]:
    """Frames between the trigger actually appearing and the alarm firing.

    `activation_frame` is an ORACLE label from the extraction harness, used
    only for scoring -- never fed to the detector. None means never detected.
    Negative would mean the alarm preceded activation (a false alarm that
    happened to land early), so it is reported as-is rather than clipped.
    """
    a = first_alarm_frame(scores, cfg)
    if a is None:
        return None
    return a - activation_frame


def summarize_episodes(episodes, cfg: Optional[TemporalConfig] = None) -> dict:
    """Episode-level AUROC + latency + false-alarm rate.

    `episodes`: iterable of dicts with
        scores            : list[float]  per-frame FTT scores, in frame order
        label             : 0 clean episode | 1 triggered episode
        activation_frame  : int | None    oracle, triggered episodes only
    """
    from sklearn.metrics import roc_auc_score

    cfg = cfg or TemporalConfig()
    ep_scores, labels, latencies = [], [], []
    clean_alarms = 0
    n_clean = 0
    detected = 0
    n_trig = 0

    for ep in episodes:
        sc = ep["scores"]
        if len(sc) <= cfg.n_baseline:
            continue
        ep_scores.append(episode_score(sc, cfg))
        labels.append(int(ep["label"]))
        alarm = first_alarm_frame(sc, cfg)
        if ep["label"] == 1:
            n_trig += 1
            if alarm is not None:
                detected += 1
                af = ep.get("activation_frame")
                if af is not None:
                    latencies.append(alarm - int(af))
        else:
            n_clean += 1
            if alarm is not None:
                clean_alarms += 1

    y = np.asarray(labels)
    s = np.asarray(ep_scores)
    auroc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

    return {
        "episode_auroc": auroc,
        "n_clean_episodes": n_clean,
        "n_triggered_episodes": n_trig,
        "detection_rate": detected / n_trig if n_trig else float("nan"),
        "false_alarm_rate": clean_alarms / n_clean if n_clean else float("nan"),
        "median_latency_frames": float(np.median(latencies)) if latencies else float("nan"),
        "mean_latency_frames": float(np.mean(latencies)) if latencies else float("nan"),
        "n_latency_samples": len(latencies),
    }


# ---------------------------------------------------------------------------
# Change-point scan: no assumption about WHICH part of the episode is clean
# ---------------------------------------------------------------------------

def changepoint_scan(scores: Sequence[float], min_seg: int = 4) -> dict:
    """Find the split point that best separates the episode into two regimes.

    Motivation: `relative_trace`/`first_alarm_frame` assume the opening frames
    are clean. That breaks if the trigger fires early -- with n_baseline=8 and
    activation at frame 2, six of the eight reference frames are already
    poisoned, the baseline becomes the TRIGGER state, and later frames look
    normal relative to it. A silent failure.

    This makes no such assumption. For every candidate split tau it compares
    the prefix [0, tau) against the suffix [tau, N) and keeps the sharpest
    separation. Activation at frame 5, 30 or 200 is handled identically, and
    tau_hat estimates when it happened.

    Separation is measured by the Mann-Whitney U statistic (equivalently, the
    AUC of prefix-vs-suffix): fully nonparametric, bounded in [0, 1], and
    insensitive to FTT's absolute scale. Oriented so HIGHER = more suspicious,
    matching `relative_trace` (FTT drops when assimilation kicks in, so a
    suspicious split is one where the suffix sits below the prefix).

    NOTE this is a RETROSPECTIVE test -- it needs the whole episode. Use it for
    episode-level AUROC. For latency and for live intervention you still want
    the online rule (`first_alarm_frame`), which alarms as evidence arrives.

    Returns {"score": separation in [0,1], "tau_hat": estimated activation}.
    A trigger present from frame 0 has no change point at all and will score
    near chance here -- that case is Stage 1's job, not this one.
    """
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    if n < 2 * min_seg:
        return {"score": float("nan"), "tau_hat": None}

    best_score, best_tau = -np.inf, None
    for tau in range(min_seg, n - min_seg + 1):
        pre, post = s[:tau], s[tau:]
        # Fraction of (pre, post) pairs where pre > post == AUC of the split.
        # Ties count as 0.5, exactly as Mann-Whitney prescribes.
        diff = pre[:, None] - post[None, :]
        u = (diff > 0).sum() + 0.5 * (diff == 0).sum()
        auc = u / (pre.size * post.size)
        if auc > best_score:
            best_score, best_tau = auc, tau

    return {"score": float(best_score), "tau_hat": int(best_tau)}


def episode_score_robust(scores: Sequence[float], cfg: Optional[TemporalConfig] = None,
                         min_seg: int = 4) -> float:
    """Episode-level score that survives an early-firing trigger.

    Drop-in alternative to `episode_score` for computing episode AUROC when
    you cannot assume the episode opens clean.
    """
    return changepoint_scan(scores, min_seg=min_seg)["score"]
