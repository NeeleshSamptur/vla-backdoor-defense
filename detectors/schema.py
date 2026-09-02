"""The contract between extractors and detectors.

Extractors are attack-specific and run in each attack's own conda env; their
only job is to write one `.npz` per (episode, condition) conforming to this
schema. Detectors read those files and never learn which attack, checkpoint or
simulator produced them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class ExtractedSample:
    # (n_text_tokens, n_image_tokens) -- one attention ROW per text query token,
    # restricted to the COLUMNS that are image-patch tokens. Raw (not
    # renormalized) post-softmax attention, averaged over heads and, when
    # layers_averaged > 1, also averaged over that many LLM layers (see
    # meta["layer"] / meta["layers_averaged"]).
    attn_text_image: np.ndarray

    label: int          # 0 = clean, 1 = triggered
    attack: str          # "badvla" | "goba" | ...
    checkpoint: str
    trigger_type: str    # e.g. "pixel_white_square", "physical_toxic_box"
    task_id: int
    seed: int
    # Which LLM layer this attention came from, or -1 as the sentinel for
    # "aggregated across layers" -- see layers_averaged to disambiguate from
    # the older meaning of -1 (literal last layer only).
    layer: int
    n_cameras: int = 1
    patches_per_camera: Optional[int] = None
    # None/1 => `attn_text_image` is from the single layer named by `layer`.
    # >1 => it is the mean over that many LLM layers (layer is then just -1).
    layers_averaged: Optional[int] = None

    episode_id: Optional[str] = None   # groups frames from one rollout
    frame_idx: int = 0                 # position within that rollout
    # BadVLA-OFT only: text x wrist-camera patches from the same forward pass.
    # None for single-camera OpenVLA.
    attn_text_image_wrist: Optional[np.ndarray] = None

    # Optional per-layer companion to attn_text_image: (n_layers, n_text_tokens,
    # n_image_tokens), UN-collapsed across layers, for detectors/ftt.py's
    # ftt_score_layerwise / ftt_score_layerwise_fixed. None unless the
    # extractor was run with an all-layers capture mode. When present,
    # attn_text_image itself is still the mean over these same layers (so
    # ftt_score keeps working unchanged on old and new samples alike).
    attn_text_image_layers: Optional[np.ndarray] = None

    extra: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        skip = {"attn_text_image", "attn_text_image_wrist", "attn_text_image_layers"}
        meta = {k: v for k, v in asdict(self).items() if k not in skip}
        payload = dict(
            attn_text_image=self.attn_text_image.astype(np.float32),
            schema_version=SCHEMA_VERSION,
            meta_json=_to_json(meta),
        )
        if self.attn_text_image_wrist is not None:
            payload["attn_text_image_wrist"] = self.attn_text_image_wrist.astype(np.float32)
        if self.attn_text_image_layers is not None:
            payload["attn_text_image_layers"] = self.attn_text_image_layers.astype(np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "ExtractedSample":
        z = np.load(path, allow_pickle=False)
        if int(z["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {z['schema_version']} != {SCHEMA_VERSION}")
        meta = _from_json(str(z["meta_json"]))
        # Drop keys for fields the schema no longer declares, so older
        # artifacts stay readable.
        known = {f.name for f in fields(cls)}
        meta = {k: v for k, v in meta.items() if k in known}
        wrist = z["attn_text_image_wrist"] if "attn_text_image_wrist" in z.files else None
        layers = z["attn_text_image_layers"] if "attn_text_image_layers" in z.files else None
        n_cam = meta.get("n_cameras", 1)
        if (n_cam >= 2) != (wrist is not None):
            raise ValueError(
                f"{path}: n_cameras={n_cam} but attn_text_image_wrist is "
                f"{'absent' if wrist is None else 'present'} -- stale or "
                "half-written artifact; re-extract this directory.")
        return cls(attn_text_image=z["attn_text_image"], attn_text_image_wrist=wrist,
                   attn_text_image_layers=layers, **meta)


def _to_json(d: dict) -> str:
    import json
    return json.dumps(d)


def _from_json(s: str) -> dict:
    import json
    return json.loads(s)


def load_dir(path: str) -> list[ExtractedSample]:
    """Load every `.npz` in a directory as a list of ExtractedSample."""
    files = sorted(Path(path).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files under {path}")
    return [ExtractedSample.load(str(f)) for f in files]
