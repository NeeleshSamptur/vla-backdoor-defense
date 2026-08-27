"""The two-stage cascade: certify the baseline, then monitor against it.

Stage 2 in temporal.py self-normalizes against an episode's opening frames.
That silently fails if the trigger fires early: with a 5-frame window and
activation at frame 2, three of the five reference frames are poisoned, the
"baseline" drifts toward the TRIGGER state, and everything after looks normal
relative to it.

The fix is to stop *assuming* the prefix is clean and instead *verify* it --
and to do that on exactly the frames that will become the baseline:

    Stage 1  score the first K frames against a population reference built
             from known-clean rollouts. If any looks suspicious -> ALARM;
             the trigger is always-on or fired early, and this baseline must
             never be trusted.
    Stage 2  only if Stage 1 certifies them, adopt those K frames as the
             reference and watch every later frame for a persistent departure.

Because the screening window IS the baseline window, contamination of the
baseline is precisely what Stage 1 tests for -- so there is no gap between
the stages:

    activation in [0, K)  -> poisons the window -> Stage 1 catches it
    activation in [K, N)  -> window is clean     -> Stage 2 catches it

What this buys, and what it costs:
  + no assumption that the episode opens clean -- it is checked
  + Stage 2's reference is certified rather than hoped-for
  - Stage 1 needs a small population reference from clean rollouts. That is
    realistic (the defender can run clean episodes) and is only used for the
    gate, but it is a genuine external dependency, unlike Stage 2 which stays
    self-normalizing.
  - Stage 2's reliability is bounded by Stage 1's FALSE-NEGATIVE rate: a
    missed contaminated prefix still poisons the baseline. This makes that
    dependency explicit and measurable rather than assumed.

Pure numpy/sklearn -- no torch, no attack code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .temporal import TemporalConfig, first_alarm_frame, relative_trace


@dataclass
class GateConfig:
    """Stage 1: the baseline-certification gate."""
    n_baseline: int = 5
    # Population reference over CLEAN frames (see fit_clean_reference).
    clean_center: float = 0.0
    clean_scale: float = 1.0
    # How far BELOW the clean population a frame must sit to look poisoned.
    # FTT polarity is low = backdoor, so the gate is one-sided.
    z_threshold: float = 4.0
    # Strictest and simplest: every frame in the window must pass. A single
    # suspicious frame is enough to refuse to trust the window, because one
    # poisoned frame is enough to drag the baseline.
    require_all: bool = True


def fit_clean_reference(clean_frame_scores: Sequence[float]) -> tuple:
    """Population reference (median, robust scale) from known-clean frames.

    Pool frames from clean rollouts of the same task distribution. Median/MAD
    so a few odd frames don't set the threshold.
    """
    s = np.asarray(clean_frame_scores, dtype=np.float64)
    if s.size == 0:
        raise ValueError("no clean frames supplied")
    center = float(np.median(s))
    mad = float(np.median(np.abs(s - center)))
    return center, max(1.4826 * mad, 1e-9)


def certify_baseline(scores: Sequence[float], gate: GateConfig) -> dict:
    """Stage 1. Is the opening window clean enough to serve as a reference?

    Returns {"clean": bool, "first_bad": int|None, "z": list[float]}.
    `first_bad` is the earliest suspicious frame -- the alarm frame when the
    gate refuses.
    """
    w = np.asarray(scores[: gate.n_baseline], dtype=np.float64)
    if w.size < gate.n_baseline:
        return {"clean": False, "first_bad": None, "z": [],
                "reason": "episode shorter than baseline window"}

    # One-sided: only a DROP below the clean population is suspicious.
    z = (gate.clean_center - w) / gate.clean_scale
    bad = np.flatnonzero(z > gate.z_threshold)

    if gate.require_all:
        ok = bad.size == 0
    else:
        ok = bad.size <= gate.n_baseline // 2

    return {"clean": bool(ok),
            "first_bad": int(bad[0]) if bad.size else None,
            "z": z.tolist()}


def cascade_detect(scores: Sequence[float], gate: GateConfig,
                   temporal: Optional[TemporalConfig] = None) -> dict:
    """Run the full cascade on one episode's per-frame FTT scores.

    Returns a verdict dict:
        stage        1 | 2 | None (never fired)
        alarm_frame  first frame of evidence, or None
        detected     bool
    """
    temporal = temporal or TemporalConfig(n_baseline=gate.n_baseline)
    if temporal.n_baseline != gate.n_baseline:
        # The no-gap property depends on these being the same frames.
        raise ValueError(
            f"baseline windows must match: gate={gate.n_baseline} "
            f"temporal={temporal.n_baseline}")

    cert = certify_baseline(scores, gate)
    if not cert["clean"]:
        return {"detected": True, "stage": 1, "alarm_frame": cert["first_bad"],
                "reason": "baseline window failed certification", "gate": cert}

    alarm = first_alarm_frame(scores, temporal)
    return {"detected": alarm is not None, "stage": 2 if alarm is not None else None,
            "alarm_frame": alarm, "reason": "temporal departure" if alarm is not None
            else "no departure detected", "gate": cert}


def episode_score(scores: Sequence[float], gate: GateConfig,
                  temporal: Optional[TemporalConfig] = None) -> float:
    """One number per episode for AUROC, combining both stages.

    Takes the stronger of (how poisoned the baseline itself looks) and (how
    far the episode later departs from that baseline), so a single score
    ranks always-on, early-firing and late-firing episodes together.
    """
    temporal = temporal or TemporalConfig(n_baseline=gate.n_baseline)
    cert = certify_baseline(scores, gate)
    stage1 = max(cert["z"]) if cert["z"] else float("-inf")

    if not cert["clean"]:
        # Baseline is untrustworthy, so the temporal half is meaningless here.
        return float(stage1)

    z = relative_trace(scores, temporal)
    tail = z[temporal.n_baseline:]
    if tail.size == 0:
        return float(stage1)
    if tail.size >= temporal.k_persist:
        kernel = np.ones(temporal.k_persist) / temporal.k_persist
        stage2 = float(np.convolve(tail, kernel, mode="valid").max())
    else:
        stage2 = float(tail.max())
    return max(float(stage1), stage2)
