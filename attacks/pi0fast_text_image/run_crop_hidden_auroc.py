#!/usr/bin/env python
"""Pi0-FAST / BackdoorVLA Text-Image (TI4): LAST-LAYER ACTIVATION shift under
a center crop -> AUROC. The pi0-FAST port of
attacks/dropvla/run_crop_hidden_auroc.py.

    score(x) = mean over the readout positions of
               1 - cos( h(x)[i], h(crop(x))[i] )        HIGH = triggered

Reads the same `.npz` observations attacks/pi0fast_text_image/collect_observations.py
writes and attacks/pi0fast_text_image/run_ftt_auroc.py scores, so the crop
detector and FTT are measured on identical episodes. Model and checkpoint
loading is IMPORTED from that FTT script (load_model, desc_token_rows,
load_paligemma_tokenizer), not reimplemented.

------------------------------------------------------------------------
The readout, and why it is not the OFT one
------------------------------------------------------------------------
`model.hidden_prefix()` (added to src/openpi/models/pi0_fast.py alongside
the existing attention_prefix, and documented there) returns the
post-final_norm activations of the prefix forward pass -- the same pass
attention_prefix runs, keeping the value it discards. pi0-FAST decodes its
FAST action tokens autoregressively AFTER this prefix, so unlike the
OpenVLA-OFT attacks there are no action-token positions in the pass, and
the OFT readout is unavailable in principle rather than merely unused. Two
documented analogues are reported instead, the same pair the GoBA port uses
for the same reason:

  final_position -- the last REAL (non-padding) prefix position, which the
                    first action token is decoded from. [1, width]
  desc_tokens    -- the instruction's own token positions, the same query
                    rows this attack's FTT scores. [n_desc, width]

------------------------------------------------------------------------
Read this before interpreting the number
------------------------------------------------------------------------
Same trigger as the OpenVLA-OFT BackdoorVLA port, and the same two reasons
a crop cannot be expected to work on it:
  image half -- the popcorn_1 OBJECT placed in the scene by the poisoned
                BDDL, sitting on the table, not at the frame border.
  text half  -- the "~*magic*~ " instruction prefix, which no image
                transform touches.
Clean and triggered observations also come from DIFFERENT BDDL scenes with
disjoint init indices, so scene content is an alternative explanation for
any separation. This run bounds what the crop detector can claim on this
attack; it is not expected to succeed.

Note the crop is applied to the stored 224x224 observation images BEFORE
openpi's own input transform (resize/pad, normalize, tokenize), which is
where a deployed input filter would sit.

Usage (pi0-FAST's own venv, NOT the robosuite conda env):
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast
    CUDA_VISIBLE_DEVICES=<n> XDG_CACHE_HOME=$PWD/cache OPENPI_DATA_HOME=$PWD/cache \\
    .venv/bin/python <this file> \\
        --checkpoint checkpoints/pi0_fast_libero_object_TI_4/PiFast_Text_Image_Attack_object_4_5000/5000 \\
        --config-name pi0_fast_libero_object_TI_4 \\
        --obs-dir <collect_observations.py's --out-dir> \\
        --out ../vla-backdoor-defense/results/libero_object/pi0fast_text_image_crop_hidden_auroc.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

PI0_ROOT = pathlib.Path("/home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast")
DEFENSE_REPO = pathlib.Path(__file__).resolve().parents[2]
for _p in (DEFENSE_REPO, PI0_ROOT / "src"):
    sys.path.insert(0, str(_p))

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model

from attacks.common import (  # noqa: E402
    center_crop_resize,
    compute_auroc_high_is_triggered,
    cosine_distance,
    relative_l2,
)
from attacks.pi0fast_text_image.run_ftt_auroc import (  # noqa: E402
    desc_token_rows,
    load_model,
    load_paligemma_tokenizer,
)


def capture_last_layer_activations(policy, model, sp, element, description):
    """(all_tokens, final_position, desc_tokens) final-layer activations.

    all_tokens is the primary readout: every REAL (non-padding) position of
    the prefix -- all three image blocks and the tokenized prompt. Unlike the
    OpenVLA-family models this sequence IS padded, so the mask selects the
    real positions rather than taking the whole tensor."""
    inputs = policy._input_transform(dict(element))  # noqa: SLF001
    batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(batched)

    hidden, n_img, seq_mask = model.hidden_prefix(observation)
    hidden = np.asarray(hidden, dtype=np.float32)      # [T, width]
    seq_mask = np.asarray(seq_mask).astype(bool)       # [T]

    real_idx = np.flatnonzero(seq_mask)
    assert real_idx.size > 0, "no real tokens in the prefix"
    all_tokens = hidden[real_idx]                      # [n_real, width]
    final_position = hidden[real_idx[-1]][None, :]     # [1, width]

    tok_ids = np.asarray(inputs["tokenized_prompt"])
    tok_mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
    rel, _decoded = desc_token_rows(sp, tok_ids[tok_mask], description)
    desc_tokens = hidden[[n_img + r for r in rel]]     # [n_desc, width]
    return all_tokens, final_position, desc_tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-scale", type=float, default=0.8)
    args = ap.parse_args()

    checkpoint = pathlib.Path(args.checkpoint).resolve()
    obs_dir = pathlib.Path(args.obs_dir)
    files = sorted(obs_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no observations under {obs_dir}")

    print(f"[*] loading {checkpoint}")
    policy, model = load_model(checkpoint, args.config_name)
    sp = load_paligemma_tokenizer()

    readouts = ("all_tokens", "final_position", "desc_tokens")
    scores = {r: {m: {"clean": [], "trigger": []} for m in ("cosine_distance", "relative_l2")}
              for r in readouts}
    episodes = []
    t_start = time.time()

    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=False)
        description = str(d["prompt"])
        cond = str(d["condition"])

        def element_for(crop):
            image, wrist = d["image"], d["wrist_image"]
            if crop is not None:
                image = center_crop_resize(image, crop)
                wrist = center_crop_resize(wrist, crop)
            return {"observation/image": image, "observation/wrist_image": wrist,
                    "observation/state": d["state"], "prompt": description}

        unc = capture_last_layer_activations(policy, model, sp, element_for(None), description)
        crp = capture_last_layer_activations(policy, model, sp, element_for(args.crop_scale), description)

        rec = {"task_id": int(d["task_id"]), "init_state_index": int(d["init_state_index"]),
               "condition": cond, "label": int(cond == "trigger")}
        for ri, r in enumerate(readouts):
            cos = cosine_distance(unc[ri], crp[ri])
            rl2 = relative_l2(unc[ri], crp[ri])
            scores[r]["cosine_distance"][cond].append(cos)
            scores[r]["relative_l2"][cond].append(rl2)
            rec[f"{r}_cosine_distance"] = cos
            rec[f"{r}_relative_l2"] = rl2
        episodes.append(rec)
        print(f"    [{i + 1}/{len(files)}] t={rec['task_id']} s={rec['init_state_index']} "
              f"{cond:8s} all-tok cos-dist={rec['all_tokens_cosine_distance']:.4f} "
              f"final-pos={rec['final_position_cosine_distance']:.4f} "
              f"elapsed={time.time() - t_start:.0f}s", flush=True)

    results = {
        "attack": "pi0fast_text_image", "checkpoint": str(checkpoint),
        "config_name": args.config_name, "task_suite_name": "libero_object",
        "crop_scale": args.crop_scale, "obs_dir": str(obs_dir),
        "trigger": "popcorn_1 object in the scene + '~*magic*~ ' instruction prefix",
        "readout": "final-layer prefix activations over the ENTIRE real (non-padding) "
                   "sequence (all_tokens, primary); pi0-FAST decodes actions after the "
                   "prefix so it has no action-token positions (see module docstring)",
        "polarity": "high = triggered",
        "eval_design": "clean vs poisoned BDDL scenes, disjoint init states",
        "crop_can_remove_trigger": False,
    }
    for r in readouts:
        results[r] = {}
        for m in ("cosine_distance", "relative_l2"):
            c, t = scores[r][m]["clean"], scores[r][m]["trigger"]
            auroc = compute_auroc_high_is_triggered(c, t)
            results[r][m] = {"n_clean": len(c), "n_trigger": len(t), "auroc": auroc,
                             "clean_mean": float(np.mean(c)) if c else float("nan"),
                             "trigger_mean": float(np.mean(t)) if t else float("nan")}
            print(f"[*] {r:14s} {m:16s}: n_clean={len(c)} n_trigger={len(t)} AUROC={auroc:.4f}")
    results["episodes"] = episodes

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
