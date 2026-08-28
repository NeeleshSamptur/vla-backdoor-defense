"""The one contract between extractors (attack-specific, run in each attack's own
env) and detectors (attack-agnostic, pure numpy/scipy/sklearn).

An extractor's only job is to produce one `.npz` per (scene, condition) pair
conforming to this schema. It never needs to know which detector will read it;
a detector never needs to know which attack, checkpoint, or simulator produced
it. That is the whole point of the split -- see README.md.

Today's detector (FTT) only needs `attn_text_image`. The schema carries a bit
more (raw per-camera boundaries, trigger metadata) so a future detector (e.g.
Bera's FBL/AFM) can be added without changing what extractors save.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class ExtractedSample:
    # (n_text_tokens, n_image_tokens) -- one attention ROW per text query token,
    # restricted to the COLUMNS that are image-patch tokens. Raw (not
    # renormalized) post-softmax attention, averaged over heads, from ONE
    # chosen layer (see meta["layer"]).
    attn_text_image: np.ndarray

    label: int          # 0 = clean, 1 = triggered
    attack: str          # "badvla" | "goba" | ...
    checkpoint: str
    trigger_type: str    # e.g. "pixel_white_square", "physical_toxic_box"
    task_id: int
    seed: int
    layer: int           # which LLM layer this attention came from
    n_cameras: int = 1
    patches_per_camera: Optional[int] = None

    # --- temporal fields (Stage 2) ---------------------------------------
    # Both current adapters (badvla_white_patch, goba) are MULTI-FRAME: each
    # episode is n_frames closed-loop policy queries, so they set these and
    # are scored with --mode stage1. A hypothetical single-frame extractor may
    # leave them at their defaults and use --mode static. Attacks whose trigger
    # appears partway through a rollout (DropVLA) additionally need
    # activation_frame, so the temporal detector can reconstruct each episode.
    episode_id: Optional[str] = None   # groups frames from the same rollout
    frame_idx: int = 0                 # position within that rollout
    # Ground-truth frame at which the trigger actually became visible, from
    # the harness's privileged state (e.g. DropVLA's object-height check).
    # ORACLE LABEL ONLY -- for scoring detection latency. Never an input to
    # any detector; a detector that reads this is cheating.
    activation_frame: Optional[int] = None
    # BadVLA-OFT only: text x wrist-camera patches, same forward as
    # attn_text_image (primary). None for GoBA / single-camera OpenVLA.
    attn_text_image_wrist: Optional[np.ndarray] = None

    extra: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        skip = {"attn_text_image", "attn_text_image_wrist"}
        meta = {k: v for k, v in asdict(self).items() if k not in skip}
        payload = dict(
            attn_text_image=self.attn_text_image.astype(np.float32),
            schema_version=SCHEMA_VERSION,
            meta_json=_to_json(meta),
        )
        if self.attn_text_image_wrist is not None:
            payload["attn_text_image_wrist"] = self.attn_text_image_wrist.astype(np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "ExtractedSample":
        z = np.load(path, allow_pickle=False)
        if int(z["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {z['schema_version']} != {SCHEMA_VERSION}")
        meta = _from_json(str(z["meta_json"]))
        wrist = z["attn_text_image_wrist"] if "attn_text_image_wrist" in z.files else None
        return cls(attn_text_image=z["attn_text_image"], attn_text_image_wrist=wrist, **meta)


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
