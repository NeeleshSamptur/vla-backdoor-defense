"""Stage 2 temporal detector tests -- synthetic episodes, no GPU, no attack repos.

Simulates the DropVLA pattern: an episode that starts clean, then at some
frame the FTT score STEPS DOWN and stays down (assimilation kicks in and
persists, because the trigger latches on).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.temporal import (
    TemporalConfig, detection_latency, episode_score, first_alarm_frame,
    relative_trace, robust_baseline, summarize_episodes,
)

CFG = TemporalConfig(n_baseline=8, k_persist=3, z_threshold=4.0)


def make_episode(n=60, activation=None, drop=0.05, noise=0.002, seed=0):
    """Clean FTT wanders around 0.08; after activation it drops and stays down."""
    rng = np.random.default_rng(seed)
    s = 0.08 + rng.normal(0, noise, n)
    if activation is not None:
        s[activation:] -= drop
    return s.tolist()


def test_clean_episode_does_not_alarm():
    for seed in range(5):
        s = make_episode(activation=None, seed=seed)
        assert first_alarm_frame(s, CFG) is None, f"false alarm on clean seed={seed}"


def test_triggered_episode_alarms():
    s = make_episode(activation=30, seed=1)
    assert first_alarm_frame(s, CFG) is not None


def test_latency_is_small_and_nonnegative():
    s = make_episode(activation=30, seed=2)
    lat = detection_latency(s, activation_frame=30, cfg=CFG)
    assert lat is not None
    # persistence rule needs k frames of evidence, so a few frames is expected
    assert 0 <= lat <= CFG.k_persist + 2, f"unexpected latency {lat}"


def test_single_frame_blip_is_filtered():
    """One-frame spike must NOT alarm -- that's what k_persist is for."""
    s = make_episode(activation=None, seed=3)
    s[35] -= 0.05  # a lone dropout frame
    assert first_alarm_frame(s, CFG) is None


def test_episode_score_separates():
    clean = [episode_score(make_episode(activation=None, seed=s), CFG) for s in range(10)]
    trig = [episode_score(make_episode(activation=30, seed=s), CFG) for s in range(10, 20)]
    assert min(trig) > max(clean), "episode scores should separate cleanly"


def test_robust_baseline_ignores_outlier():
    s = [0.08] * 8
    s[3] = 5.0  # wild outlier inside the baseline window
    center, scale = robust_baseline(s, 8)
    assert abs(center - 0.08) < 1e-9, "median should ignore the outlier"


def test_polarity_low_ftt_is_suspicious():
    """FTT convention: low = backdoor. relative_trace must map that to POSITIVE."""
    s = make_episode(activation=30, seed=4)
    z = relative_trace(s, CFG)
    assert z[40] > 0, "post-activation frames must score positive (suspicious)"


def test_summarize_end_to_end():
    eps = []
    for s in range(15):
        eps.append({"scores": make_episode(activation=None, seed=s), "label": 0,
                    "activation_frame": None})
    for s in range(15, 30):
        eps.append({"scores": make_episode(activation=30, seed=s), "label": 1,
                    "activation_frame": 30})
    r = summarize_episodes(eps, CFG)
    assert r["episode_auroc"] == 1.0, r
    assert r["detection_rate"] == 1.0, r
    assert r["false_alarm_rate"] == 0.0, r
    assert r["median_latency_frames"] <= CFG.k_persist + 2, r


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all temporal tests passed")
