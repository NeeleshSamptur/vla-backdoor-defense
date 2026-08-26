"""Detector unit tests: synthetic attention arrays, no GPU, no attack repos.

This is the whole point of the extraction/detection split -- these tests
exercise detectors/ end to end without ever installing BadVLA, GoBA, or
LIBERO. If this file passes, the FTT math is correct regardless of which
attack later feeds it real data.
"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors.ftt import auroc_both_polarities, ftt_score, row_normalize
from detectors.schema import ExtractedSample, load_dir


def test_row_normalize_sums_to_one():
    P = np.random.rand(5, 10)
    Pn = row_normalize(P)
    assert np.allclose(Pn.sum(axis=1), 1.0, atol=1e-6)


def test_ftt_zero_for_identical_rows():
    P = np.tile(np.random.dirichlet(np.ones(20)), (8, 1))
    assert ftt_score(P) < 1e-6


def test_ftt_low_for_assimilated_rows():
    """The Assimilation Phenomenon, synthetically: trigger rows collapse
    toward a shared peak, so FTT should be LOWER for the 'triggered' set."""
    rng = np.random.default_rng(0)
    clean = ftt_score(rng.dirichlet(np.ones(30) * 2, size=8))
    peak = np.zeros(30); peak[3] = 1.0
    trig_rows = 0.2 * rng.dirichlet(np.ones(30) * 2, size=8) + 0.8 * peak[None, :]
    trig = ftt_score(trig_rows)
    assert trig < clean


def test_schema_roundtrip(tmp_path=Path("/tmp/vla_bd_defense_test")):
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True)
    s = ExtractedSample(
        attn_text_image=np.random.rand(4, 16).astype(np.float32),
        label=1, attack="unittest", checkpoint="ckpt/path", trigger_type="synthetic",
        task_id=2, seed=42, layer=-1, n_cameras=1, patches_per_camera=16,
        extra={"role": "attack"},
    )
    s.save(str(tmp_path / "sample.npz"))
    loaded = load_dir(str(tmp_path))
    assert len(loaded) == 1
    r = loaded[0]
    assert r.attack == "unittest" and r.label == 1 and r.extra["role"] == "attack"
    assert np.allclose(r.attn_text_image, s.attn_text_image)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_auroc_recovers_known_separation():
    clean = [0.09, 0.10, 0.11, 0.095]
    trig = [0.01, 0.02, 0.015, 0.018]
    auroc = auroc_both_polarities(clean, trig)
    assert auroc["low_is_backdoor"] == 1.0
    assert auroc["high_is_backdoor"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
