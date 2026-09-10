#!/usr/bin/env python
"""DropVLA: LAST-LAYER ACTIVATION shift under a center crop -> AUROC.

The detector, in one line: score a frame by how far the policy's last-layer
activations move when the image is center-cropped, and flag the frames that
move most.

    score(x) = mean over the action-token positions of
               1 - cos( h(x)[i], h(crop(x))[i] )

    h(.)   the FINAL LLM layer's hidden states at the action-token
           positions, shape [NUM_ACTIONS_CHUNK * ACTION_DIM, d_model]
           = [56, 4096] for LIBERO -- exactly the tensor OFT's L1-regression
           action head consumes, taken straight from DropVLA's own
           `vla.predict_action`, which returns it as its second value. No
           reassembly of the multimodal sequence is involved (unlike the
           attention scripts, which must rebuild it because generate() never
           surfaces attentions), so this is the model's own inference path.
    crop   attacks.common.center_crop_resize at --crop-scale, applied to
           BOTH camera images, nothing else changed.

Polarity is HIGH = triggered, the OPPOSITE of FTT, so scoring goes through
attacks.common.compute_auroc_high_is_triggered (which negates and defers to
compute_auroc) -- passing these scores to compute_auroc directly would
silently report 1 - AUROC.

------------------------------------------------------------------------
Why a cropped frame separates triggered from clean
------------------------------------------------------------------------
DropVLA's trigger is a 5px-radius dot at (10, 10) of the 256px render --
about ONE of the 16x16 vision patches, in a corner. A crop to the central
80% deletes it while leaving the robot, object and table in view. So:

  triggered frame -- the crop removes the thing driving the "drop it"
                     decision, and the last layer swings back toward the
                     untriggered behavior: a large move.
  clean frame     -- the crop is an ordinary change of view the policy is
                     robust to: a small move.

This is the same crop, the same frames and the same rollout code as the
sibling run_crop_ftt_auroc.py; only the readout differs (last-layer
activations here, desc_only text-to-image attention there), so the two
scripts' AUROCs are directly comparable on the same episodes.

Everything that is not the readout -- model loading with the eager-attention
patch, the rollout to DropVLA's physically-gated trigger frame,
observation prep, and the GenerateConfig holding DropVLA's own run_eval.sh
paper protocol -- is IMPORTED from attacks/dropvla/run_ftt_auroc.py. See
that file's docstring for why one rollout yields both a clean and a
triggered sample from the identical frame.

------------------------------------------------------------------------
Reported numbers
------------------------------------------------------------------------
  cosine_distance -- the score above. (Cosine SIMILARITY carries the same
                     information with the polarity flipped, so its AUROC is
                     exactly 1 minus this one's; it is reported too rather
                     than left for the reader to work out.)
  relative_l2     -- ||h(crop(x)) - h(x)||_F / ||h(x)||_F, the magnitude
                     counterpart of the angle-based score.
  action_gripper  -- REFERENCE ONLY, not a defense: the unperturbed gripper
                     command (raw scale, 0 = close, 1 = open; DropVLA's own
                     process_action binarizes at 0.5). It says whether the
                     trigger fired at all in each episode. At this frame a
                     clean policy is holding the object and a triggered one
                     is about to drop it, so this reference separates the
                     classes on its own -- which is exactly why a high AUROC
                     here cannot by itself prove the crop score detects the
                     TRIGGER rather than an imminent gripper opening. Adding
                     clean frames at a legitimate release as hard negatives
                     is the experiment that would settle it.

Per-episode scores are written into the results JSON, so every reported
AUROC can be recomputed from that file without a GPU. Pass
--hidden-states-dir to additionally save the raw [56, 4096] activation
blocks (float16, ~0.5 MB per frame per variant, ~180 MB for 100 episodes);
off by default since the scores above are what the AUROCs need.

Usage (DropVLA's own conda env, same as run_ftt_auroc.py):
    source /home/grads/nsamptur/vla_bkd_def/DropVLA/dropvla_env.sh
    cd /home/grads/nsamptur/vla_bkd_def/DropVLA

    python /home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/attacks/dropvla/run_crop_hidden_auroc.py \
        --checkpoint "$RUN_DIR/openvla-7b+libero_spatial_no_noops_v5p00carefully+...--seed42--paper" \
        --task-suite-name libero_spatial --trigger-mode vision --crop-scale 0.8 \
        --out ../vla-backdoor-defense/results/dropvla_crop_hidden_auroc.json
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

from attacks.dropvla.run_ftt_auroc import (
    VALID_SUITES,
    LiftNotReached,
    build_base_cfg,
    run_to_trigger_frame,
)
from attacks.dropvla.run_crop_ftt_auroc import crop_observation
from attacks.common import compute_auroc_high_is_triggered, cosine_distance, relative_l2

from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.libero.run_libero_eval import initialize_model, prepare_observation
from experiments.robot.openvla_utils import DEVICE, normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

GRIPPER_DIM = 6  # LIBERO action layout: [dx, dy, dz, droll, dpitch, dyaw, gripper]


@torch.inference_mode()
def capture_last_layer_activations(vla, processor, action_head, proprio_projector, cfg,
                                   observation, desc):
    """One forward pass through DropVLA's own `vla.predict_action`, returning
    (action_chunk, last_layer_activations).

    Input assembly is the same as openvla_utils.get_vla_action's: both camera
    images through prepare_images_for_vla, their pixel values concatenated,
    the OFT prompt template, and normalized proprio. The proprio state is
    normalized on a COPY -- get_vla_action normalizes obs["state"] in place,
    which would corrupt the observation for the second (cropped) pass that
    reuses it.

    Returns:
        action  [NUM_ACTIONS_CHUNK, ACTION_DIM] unnormalized action chunk
        hidden  [NUM_ACTIONS_CHUNK * ACTION_DIM, d_model] final-layer states
    """
    images = prepare_images_for_vla([observation["full_image"], observation["wrist_image"]], cfg)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, images[0]).to(DEVICE, dtype=torch.bfloat16)
    wrist_in = processor(prompt, images[1]).to(DEVICE, dtype=torch.bfloat16)
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_in["pixel_values"]], dim=1)
    proprio = normalize_proprio(observation["state"].copy(), vla.norm_stats[cfg.unnorm_key]["proprio"])

    action, hidden = vla.predict_action(
        **inputs,
        unnorm_key=cfg.unnorm_key,
        do_sample=False,
        proprio=proprio,
        proprio_projector=proprio_projector,
        noisy_action_projector=None,
        action_head=action_head,
        use_film=False,
    )
    return np.asarray(action, dtype=np.float32), hidden[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--trigger-mode", required=True, choices=["vision", "language", "joint"],
                    help="same meaning as in run_ftt_auroc.py; a crop cannot remove a language "
                         "suffix, so 'language' is a no-signal control for this detector.")
    ap.add_argument("--out", required=True, help="path to write the results JSON to.")
    ap.add_argument("--crop-scale", type=float, default=0.8,
                    help="linear fraction kept by the center crop (0.8 removes DropVLA's dot).")
    ap.add_argument("--hidden-states-dir", default=None,
                    help="optional directory for the raw [56, 4096] activation blocks (float16), "
                         "one .npz per (episode, condition) holding the uncropped and cropped "
                         "block. Omit to save only the per-episode scores in the results JSON.")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--cover-wrist-lower-quarter", action="store_true")
    ap.add_argument("--load-in-4bit", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    base_cfg = build_base_cfg(args.checkpoint, args.task_suite_name, args.load_in_4bit,
                              args.cover_wrist_lower_quarter, args.seed)

    print(f"[*] loading {args.checkpoint} (suite={args.task_suite_name}, "
          f"trigger_mode={args.trigger_mode}, crop_scale={args.crop_scale})")
    model, action_head, proprio_projector, _, processor = initialize_model(base_cfg)
    resize_size = get_image_resize_size(base_cfg)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)
    use_vision = args.trigger_mode in ("vision", "joint")
    use_language = args.trigger_mode in ("language", "joint")

    hidden_dir = Path(args.hidden_states_dir) if args.hidden_states_dir else None
    if hidden_dir:
        hidden_dir.mkdir(parents=True, exist_ok=True)

    metrics = ("cosine_distance", "relative_l2", "action_gripper")
    scores = {m: {"clean": [], "trigger": []} for m in metrics}
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
                    action_unc, hidden_unc = capture_last_layer_activations(
                        model, processor, action_head, proprio_projector, base_cfg, observation, desc)
                    _, hidden_crop = capture_last_layer_activations(
                        model, processor, action_head, proprio_projector, base_cfg,
                        crop_observation(observation, args.crop_scale), desc)

                    per_episode = {
                        "cosine_distance": cosine_distance(hidden_unc, hidden_crop),
                        "relative_l2": relative_l2(hidden_unc, hidden_crop),
                        "action_gripper": float(action_unc[:, GRIPPER_DIM].mean()),
                    }
                    for m in metrics:
                        scores[m][cond].append(per_episode[m])
                    per_episode["cosine_similarity"] = 1.0 - per_episode["cosine_distance"]
                    record[cond] = per_episode

                    if hidden_dir:
                        np.savez_compressed(
                            hidden_dir / f"{args.task_suite_name}__t{task_id}__s{episode_idx}__{cond}.npz",
                            uncropped=hidden_unc.astype(np.float16),
                            cropped=hidden_crop.astype(np.float16),
                            action_uncropped=action_unc,
                            meta_json=json.dumps({
                                "attack": "dropvla", "checkpoint": args.checkpoint,
                                "task_suite_name": args.task_suite_name, "task_id": task_id,
                                "init_index": episode_idx, "frame_idx": t, "condition": cond,
                                "label": int(cond == "trigger"), "trigger_mode": args.trigger_mode,
                                "crop_scale": args.crop_scale, "task_description": task_description,
                                "readout": "final LLM layer at action-token positions",
                                "scores": per_episode,
                            }))
                episodes.append(record)
                print(f"    task={task_id} idx={episode_idx} activated@t={t}  "
                      f"cos-dist clean={record['clean']['cosine_distance']:.4f} "
                      f"trigger={record['trigger']['cosine_distance']:.4f}  "
                      f"gripper {record['clean']['action_gripper']:+.2f}->"
                      f"{record['trigger']['action_gripper']:+.2f}", flush=True)
        finally:
            env.close()

    del model, processor, proprio_projector
    torch.cuda.empty_cache()

    results = {
        "attack": "dropvla", "checkpoint": args.checkpoint,
        "task_suite_name": args.task_suite_name, "trigger_mode": args.trigger_mode,
        "crop_scale": args.crop_scale, "seed": args.seed,
        "readout": "final LLM layer hidden states at action-token positions [56, 4096]",
        "n_episodes": len(episodes),
        "polarity": {"cosine_distance": "high = triggered",
                     "cosine_similarity": "low = triggered (mirror of cosine_distance)",
                     "relative_l2": "high = triggered",
                     "action_gripper": "high = triggered; REFERENCE ONLY, not a defense"},
    }
    for m in metrics:
        c, t_ = scores[m]["clean"], scores[m]["trigger"]
        auroc = compute_auroc_high_is_triggered(c, t_)
        results[m] = {"n_clean": len(c), "n_trigger": len(t_), "auroc": auroc,
                      "clean_mean": float(np.mean(c)) if c else float("nan"),
                      "trigger_mean": float(np.mean(t_)) if t_ else float("nan")}
        print(f"[*] {m:16s}: n_clean={len(c)} n_trigger={len(t_)} AUROC={auroc:.4f}  "
              f"mean clean={results[m]['clean_mean']:.4f} trigger={results[m]['trigger_mean']:.4f}")
    # Cosine similarity is the same statistic with the polarity flipped; its
    # AUROC is 1 - the cosine-distance AUROC by construction, recorded so the
    # JSON answers the question directly instead of implying it.
    results["cosine_similarity"] = {"auroc": 1.0 - results["cosine_distance"]["auroc"],
                                    "note": "mirror of cosine_distance (low similarity = triggered)"}
    results["episodes"] = episodes

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] saved -> {out_path}")
    if hidden_dir:
        print(f"[*] activation blocks -> {hidden_dir}")


if __name__ == "__main__":
    main()
