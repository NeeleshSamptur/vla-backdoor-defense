"""Cascade tests -- the headline one is the activation-frame sweep.

The whole point of gating the baseline on Stage 1 is that there should be NO
activation frame that slips between the stages. test_no_gap_across_activation
_frames checks exactly that, including the frames right at the boundary where
the naive fixed-baseline version silently fails.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.cascade import (
    GateConfig, cascade_detect, certify_baseline, episode_score, fit_clean_reference,
)
from detectors.temporal import TemporalConfig, first_alarm_frame

K = 5
CLEAN_FTT, DROP, NOISE = 0.08, 0.05, 0.002


def make_episode(n=60, activation=None, seed=0):
    rng = np.random.default_rng(seed)
    s = CLEAN_FTT + rng.normal(0, NOISE, n)
    if activation is not None:
        s[activation:] -= DROP
    return s.tolist()


def build_gate():
    """Population reference from clean rollouts, as a defender would."""
    clean_frames = []
    for seed in range(200, 220):
        clean_frames.extend(make_episode(activation=None, seed=seed))
    c, sc = fit_clean_reference(clean_frames)
    return GateConfig(n_baseline=K, clean_center=c, clean_scale=sc, z_threshold=4.0)


GATE = build_gate()
TEMP = TemporalConfig(n_baseline=K, k_persist=3, z_threshold=4.0)


def test_clean_episode_passes_both_stages():
    for seed in range(10):
        r = cascade_detect(make_episode(activation=None, seed=seed), GATE, TEMP)
        assert not r["detected"], f"false alarm on clean seed={seed}: {r}"


def test_early_activation_caught_by_stage1():
    """Activation inside the baseline window must be caught by the GATE,
    because that window is exactly what Stage 1 screens."""
    for act in range(0, K):
        r = cascade_detect(make_episode(activation=act, seed=act), GATE, TEMP)
        assert r["detected"], f"missed early activation at {act}"
        assert r["stage"] == 1, f"activation {act} should be Stage 1, got {r}"


def test_late_activation_caught_by_stage2():
    for act in [K, K + 1, 10, 25, 40]:
        r = cascade_detect(make_episode(activation=act, seed=act), GATE, TEMP)
        assert r["detected"], f"missed late activation at {act}"
        assert r["stage"] == 2, f"activation {act} should be Stage 2, got {r}"


def test_no_gap_across_activation_frames():
    """THE test: sweep activation across the whole episode. Every single one
    must be caught by one stage or the other -- no silent hole."""
    misses = []
    for act in range(0, 50):
        r = cascade_detect(make_episode(activation=act, seed=1000 + act), GATE, TEMP)
        if not r["detected"]:
            misses.append(act)
    assert not misses, f"activation frames slipped through both stages: {misses}"


def test_naive_baseline_DOES_fail_early_activation():
    """Confirms the hole is real, and that gating is what fixes it.
    Without the gate, an early activation poisons the baseline and the
    temporal rule goes silent."""
    s = make_episode(activation=1, seed=7)
    assert first_alarm_frame(s, TEMP) is None, \
        "expected the ungated temporal rule to MISS an early activation"
    # the cascade catches the same episode via the gate
    assert cascade_detect(s, GATE, TEMP)["detected"]


def test_certify_flags_the_right_frame():
    s = make_episode(activation=3, seed=11)
    cert = certify_baseline(s, GATE)
    assert not cert["clean"]
    assert cert["first_bad"] == 3, cert


def test_episode_score_separates_all_regimes():
    clean = [episode_score(make_episode(activation=None, seed=s), GATE, TEMP)
             for s in range(20)]
    early = [episode_score(make_episode(activation=2, seed=s), GATE, TEMP)
             for s in range(20, 30)]
    late = [episode_score(make_episode(activation=30, seed=s), GATE, TEMP)
            for s in range(30, 40)]
    assert min(early) > max(clean), "early-firing must outrank clean"
    assert min(late) > max(clean), "late-firing must outrank clean"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all cascade tests passed")
