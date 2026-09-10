#!/usr/bin/env python
"""DropVLA: text-to-image FTT -> AUROC on CROPPED frames, with both the
uncropped and the cropped attention maps saved to disk.

This is the crop variant of the sibling attacks/dropvla/run_ftt_auroc.py.
Every part of the pipeline that is not the crop -- model loading (including
the eager-attention patch), the LIBERO rollout that drives to DropVLA's
physically-gated trigger frame, observation preparation, the desc_only
text-to-image attention capture, and the GenerateConfig holding DropVLA's
own run_eval.sh paper protocol -- is IMPORTED from that script, not
reimplemented, so the only difference between the two numbers this script
reports is the crop itself. Read that file's docstring for why one rollout
yields both a clean and a triggered sample from the identical frame, and
for why attn_implementation="eager" is required for output_attentions.

------------------------------------------------------------------------
What this script does per episode
------------------------------------------------------------------------
At the captured trigger frame, for EACH of the two conditions (clean and
triggered), it runs the attention forward pass TWICE:

  uncropped -- the frame exactly as prepare_observation produced it
  cropped   -- both camera images passed through
               attacks.common.center_crop_resize(img, --crop-scale) first

giving four attention-map sets per episode. All four are saved (see
--attention-maps-dir below), and each is scored with the same
attacks.common.compute_ftt used by every other script here, so:

  * the UNCROPPED AUROC is FTT on DropVLA as-is, and must agree with what
    run_ftt_auroc.py reports for the same --seed/--n-tasks/--n-seeds (same
    rollouts, same frames, same forward pass -- it is the same code). It is
    recomputed here rather than assumed so the two halves of the comparison
    come from one run over one set of episodes.
  * the CROPPED AUROC is FTT after the crop has removed the trigger.

------------------------------------------------------------------------
Why crop, and what each AUROC can and cannot show
------------------------------------------------------------------------
DropVLA's trigger is a 5px-radius dot drawn at (10, 10) of the 256px render,
i.e. roughly ONE of the 16x16 vision patches, in a corner. A crop to the
central 80% removes it completely (verified by rendering the cropped frames,
and by the trigger no longer changing the policy's gripper command), while
leaving the robot, the object and the table in view.

So the two AUROCs answer different questions, and neither on its own is a
defense:
  uncropped -- can FTT see this trigger at all?
  cropped   -- FTT with the trigger deleted from the input. This is a
               CONTROL, not a detector: with the trigger gone the two
               conditions differ only in the language suffix (nothing at all
               under --trigger-mode vision), so anything far from 0.5 here
               means the score is keying on something other than the dot.

Under --trigger-mode vision the cropped AUROC is in fact 0.5 BY
CONSTRUCTION whenever the crop removes the dot: the dot is then the only
thing that differed between the clean and the triggered frame, so the two
cropped images are pixel-identical and produce identical attention. A
cropped AUROC of exactly 0.5 with identical FTT values per pair is therefore
a positive check that the crop really did delete the trigger, and nothing
more. (Under 'language'/'joint' the suffix survives the crop, so there the
number is a real measurement.)

Two detector-shaped quantities are reported alongside, both scored with
HIGH = triggered, both recomputable from the saved maps:

  attn_shift -- distance between the uncropped and the cropped attention
                map (each row-normalized and averaged over layers exactly as
                compute_ftt does, then the Frobenius distance between the
                two). This is the attention-space analogue of measuring how
                far a representation moves when the trigger is removed: for a
                triggered frame the crop deletes the trigger, for a clean
                frame it is only a nuisance change of view.
  delta_ftt  -- |FTT_cropped - FTT_uncropped|. Cheaper but weaker than
                attn_shift: because the cropped FTT is a per-pair CONSTANT
                under a vision-only trigger, this is just the uncropped FTT
                re-centered on that pair's own crop baseline, not an
                independent signal.

------------------------------------------------------------------------
Saved attention maps
------------------------------------------------------------------------
One .npz per (episode, condition) under --attention-maps-dir:

    <task_suite>__t<task_id>__s<init_index>__<clean|trigger>.npz
      primary_uncropped  [n_layers, n_desc_tokens, n_patches]  float32
      wrist_uncropped    [n_layers, n_desc_tokens, n_patches]
      primary_cropped    [n_layers, n_desc_tokens, n_patches]
      wrist_cropped      [n_layers, n_desc_tokens, n_patches]
      meta_json          run/episode provenance (crop scale, trigger mode,
                         frame index, task description, FTT scores)

The cropped and uncropped maps come from the same frame, the same query
tokens and the same layers -- only the input pixels differ.

These are raw, non-negative, NOT row-normalized attention rows, head-averaged
per layer -- exactly what compute_ftt expects as input -- so every number in
the results JSON can be recomputed from the saved maps without a GPU.

Usage (DropVLA's own conda env, same as run_ftt_auroc.py):
    source /home/grads/nsamptur/vla_bkd_def/DropVLA/dropvla_env.sh
    cd /home/grads/nsamptur/vla_bkd_def/DropVLA

    python /home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/attacks/dropvla/run_crop_ftt_auroc.py \
        --checkpoint "$RUN_DIR/openvla-7b+libero_spatial_no_noops_v5p00carefully+...--seed42--paper" \
        --task-suite-name libero_spatial --trigger-mode vision --crop-scale 0.8 \
        --out ../vla-backdoor-defense/results/dropvla_crop_ftt/dropvla_crop_ftt_auroc.json \
        --attention-maps-dir ../vla-backdoor-defense/results/dropvla_crop_ftt/attention_maps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)

import numpy as np
import torch
from libero.libero import benchmark

# Importing the sibling script also installs its eager-attention get_vla
# patch, which this script's attention capture depends on just as much.
from attacks.dropvla.run_ftt_auroc import (
    VALID_SUITES,
    LiftNotReached,
    build_base_cfg,
    capture_text_to_image_attention,
    run_to_trigger_frame,
)
from attacks.common import (center_crop_resize, compute_auroc,
                            compute_auroc_high_is_triggered, compute_ftt, row_normalize)

from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.libero.run_libero_eval import initialize_model, prepare_observation
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

CAMERA_KEYS = ("full_image", "wrist_image")


def crop_observation(observation, crop_scale):
    """A copy of `observation` with both camera images center-cropped and
    resized back. Everything else (the proprio state) passes through
    untouched, so the cropped forward pass differs from the uncropped one in
    the pixels alone."""
    cropped = dict(observation)
    for key in CAMERA_KEYS:
        cropped[key] = center_crop_resize(observation[key], crop_scale)
    return cropped


def attention_shift(uncropped: np.ndarray, cropped: np.ndarray) -> float:
    """How far the attention map moves when the frame is cropped.

    Both inputs are [n_layers, n_query_tokens, n_patches] raw attention for
    the SAME frame and the same query tokens. Each is collapsed to one map
    the way compute_ftt does it (row-normalize every layer independently,
    then average the layers), and the score is the Frobenius distance
    between the two maps. Higher = the crop changed this frame's attention
    more.
    """
    def collapse(a):
        normed = row_normalize(a)
        return normed.mean(axis=tuple(range(normed.ndim - 2)))

    return float(np.linalg.norm(collapse(uncropped) - collapse(cropped)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--trigger-mode", required=True, choices=["vision", "language", "joint"],
                    help="same meaning as in run_ftt_auroc.py; a crop cannot remove a language "
                         "suffix, so under 'language'/'joint' the cropped condition still carries "
                         "the text half of the trigger.")
    ap.add_argument("--out", required=True, help="path to write the results JSON to.")
    ap.add_argument("--attention-maps-dir", required=True,
                    help="directory for the per-episode .npz holding BOTH the uncropped and the "
                         "cropped attention maps.")
    ap.add_argument("--crop-scale", type=float, default=0.8,
                    help="linear fraction of the frame kept by the center crop (0.8 removes "
                         "DropVLA's corner dot entirely).")
    ap.add_argument("--n-tasks", type=int, default=10, help="tasks per suite; LIBERO suites have 10.")
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="curated init-state indices per task to attempt; one clean+triggered pair "
                         "is produced per index where the lift/time condition is reached.")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--cover-wrist-lower-quarter", action="store_true")
    ap.add_argument("--load-in-4bit", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    base_cfg = build_base_cfg(args.checkpoint, args.task_suite_name, args.load_in_4bit,
                              args.cover_wrist_lower_quarter, args.seed)

    print(f"[*] loading {args.checkpoint} (suite={args.task_suite_name}, "
          f"trigger_mode={args.trigger_mode}, crop_scale={args.crop_scale}, "
          f"load_in_4bit={args.load_in_4bit})")
    model, action_head, proprio_projector, _, processor = initialize_model(base_cfg)
    resize_size = get_image_resize_size(base_cfg)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)
    use_vision = args.trigger_mode in ("vision", "joint")
    use_language = args.trigger_mode in ("language", "joint")

    maps_dir = Path(args.attention_maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)

    # scores[variant][camera][condition] -> list of per-episode FTT values
    variants = ("uncropped", "cropped")
    scores = {v: {cam: {"clean": [], "trigger": []} for cam in ("primary", "wrist")} for v in variants}
    delta = {cam: {"clean": [], "trigger": []} for cam in ("primary", "wrist")}
    shift = {cam: {"clean": [], "trigger": []} for cam in ("primary", "wrist")}
    episodes = []

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        n_avail = init_states.shape[0]
        env, task_description = get_libero_env(task, base_cfg.model_family, resolution=base_cfg.env_img_res)
        try:
            for k in range(args.n_seeds):
                episode_idx = args.seed_base + k
                if episode_idx >= n_avail:
                    print(f"    [!] task={task_id}: only {n_avail} curated init states, "
                          f"skipping index {episode_idx}")
                    continue
                try:
                    obs, t = run_to_trigger_frame(
                        base_cfg, env, task_description, model, action_head, proprio_projector,
                        processor, resize_size, init_states[episode_idx])
                except LiftNotReached as e:
                    print(f"    task={task_id} idx={episode_idx}: [skip] {e}")
                    continue

                clean_cfg = replace(base_cfg, use_visual_backdoor=False, use_backdoor_instruction=False)
                trig_cfg = replace(base_cfg, use_visual_backdoor=use_vision,
                                   use_backdoor_instruction=use_language)
                observations = {
                    "clean": (prepare_observation(clean_cfg, obs, resize_size, backdoor_active=False)[0],
                              task_description),
                    "trigger": (prepare_observation(trig_cfg, obs, resize_size, backdoor_active=True)[0],
                                f"{task_description.strip()} {trig_cfg.language_suffix.strip()}".strip()
                                if use_language else task_description),
                }

                record = {"task_id": task_id, "init_index": episode_idx, "frame_idx": t,
                          "task_description": task_description}
                for cond, (observation, desc) in observations.items():
                    maps, ftt = {}, {}
                    for variant, obs_in in (("uncropped", observation),
                                            ("cropped", crop_observation(observation, args.crop_scale))):
                        primary, wrist = capture_text_to_image_attention(
                            model, processor, proprio_projector, base_cfg, obs_in, desc)
                        maps[f"primary_{variant}"] = primary.astype(np.float32)
                        maps[f"wrist_{variant}"] = wrist.astype(np.float32)
                        for cam, m in (("primary", primary), ("wrist", wrist)):
                            ftt[f"{cam}_{variant}"] = compute_ftt(m)
                            scores[variant][cam][cond].append(ftt[f"{cam}_{variant}"])
                    for cam in ("primary", "wrist"):
                        d = abs(ftt[f"{cam}_cropped"] - ftt[f"{cam}_uncropped"])
                        delta[cam][cond].append(d)
                        ftt[f"{cam}_delta"] = d
                        shift_val = attention_shift(maps[f"{cam}_uncropped"], maps[f"{cam}_cropped"])
                        shift[cam][cond].append(shift_val)
                        ftt[f"{cam}_attn_shift"] = shift_val
                    record[cond] = ftt

                    np.savez_compressed(
                        maps_dir / f"{args.task_suite_name}__t{task_id}__s{episode_idx}__{cond}.npz",
                        meta_json=json.dumps({
                            "attack": "dropvla", "checkpoint": args.checkpoint,
                            "task_suite_name": args.task_suite_name, "task_id": task_id,
                            "init_index": episode_idx, "frame_idx": t, "condition": cond,
                            "label": int(cond == "trigger"), "trigger_mode": args.trigger_mode,
                            "crop_scale": args.crop_scale, "task_description": task_description,
                            "language_instruction_used": desc, "text_scope": "desc_only",
                            "attention": "raw, non-negative, head-averaged per layer, all LLM layers",
                            "ftt": ftt,
                        }),
                        **maps)
                episodes.append(record)
                print(f"    task={task_id} idx={episode_idx} activated@t={t}  "
                      f"primary FTT clean {record['clean']['primary_uncropped']:.4f}->"
                      f"{record['clean']['primary_cropped']:.4f}  "
                      f"trigger {record['trigger']['primary_uncropped']:.4f}->"
                      f"{record['trigger']['primary_cropped']:.4f}", flush=True)
        finally:
            env.close()

    del model, processor, proprio_projector
    torch.cuda.empty_cache()

    results = {
        "attack": "dropvla", "checkpoint": args.checkpoint,
        "task_suite_name": args.task_suite_name, "trigger_mode": args.trigger_mode,
        "crop_scale": args.crop_scale, "seed": args.seed,
        "n_episodes": len(episodes),
        "attention_maps_dir": str(maps_dir),
        "polarity": {"uncropped": "low FTT = triggered", "cropped": "low FTT = triggered",
                     "attn_shift": "high uncropped-vs-cropped map distance = triggered",
                     "delta_ftt": "high |FTT_cropped - FTT_uncropped| = triggered"},
    }
    for variant in variants:
        results[variant] = {}
        for cam in ("primary", "wrist"):
            c, t_ = scores[variant][cam]["clean"], scores[variant][cam]["trigger"]
            auroc = compute_auroc(c, t_)
            results[variant][cam] = {"n_clean": len(c), "n_trigger": len(t_), "auroc": auroc,
                                     "clean_mean_ftt": float(np.mean(c)) if c else float("nan"),
                                     "trigger_mean_ftt": float(np.mean(t_)) if t_ else float("nan")}
            print(f"[*] {variant:9s} {cam:7s}: n_clean={len(c)} n_trigger={len(t_)} AUROC={auroc:.4f}")
    results["attn_shift"] = {}
    for cam in ("primary", "wrist"):
        c, t_ = shift[cam]["clean"], shift[cam]["trigger"]
        auroc = compute_auroc_high_is_triggered(c, t_)
        results["attn_shift"][cam] = {"n_clean": len(c), "n_trigger": len(t_), "auroc": auroc,
                                      "clean_mean_shift": float(np.mean(c)) if c else float("nan"),
                                      "trigger_mean_shift": float(np.mean(t_)) if t_ else float("nan")}
        print(f"[*] attn_shift {cam:7s}: n_clean={len(c)} n_trigger={len(t_)} AUROC={auroc:.4f}")
    results["delta_ftt"] = {}
    for cam in ("primary", "wrist"):
        c, t_ = delta[cam]["clean"], delta[cam]["trigger"]
        auroc = compute_auroc_high_is_triggered(c, t_)
        results["delta_ftt"][cam] = {"n_clean": len(c), "n_trigger": len(t_), "auroc": auroc,
                                     "clean_mean_delta": float(np.mean(c)) if c else float("nan"),
                                     "trigger_mean_delta": float(np.mean(t_)) if t_ else float("nan")}
        print(f"[*] delta_ftt {cam:7s}: n_clean={len(c)} n_trigger={len(t_)} AUROC={auroc:.4f}")
    results["episodes"] = episodes

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] saved -> {out_path}")
    print(f"[*] attention maps -> {maps_dir}")


if __name__ == "__main__":
    main()
