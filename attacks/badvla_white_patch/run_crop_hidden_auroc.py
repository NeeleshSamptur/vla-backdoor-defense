#!/usr/bin/env python
"""BadVLA (white-patch trigger): LAST-LAYER ACTIVATION shift under a center
crop -> AUROC. Same detector as attacks/dropvla/run_crop_hidden_auroc.py,
ported to this attack.

    score(x) = mean over the action-token positions of
               1 - cos( h(x)[i], h(crop(x))[i] )        HIGH = triggered

    h(.)   the final LLM layer's hidden states at the action-token
           positions, [NUM_ACTIONS_CHUNK * ACTION_DIM, d_model], taken from
           the SAME reconstructed forward pass this attack's
           run_ftt_auroc.py already builds for its attention -- only
           output_hidden_states replaces output_attentions. Those positions
           are the slice OpenVLAForActionPrediction's own
           _regression_or_discrete_prediction feeds to the action head.
    crop   attacks.common.center_crop_resize at --crop-scale, applied to
           BOTH cameras AFTER the trigger patch, which is the order a
           deployed input filter would see: the camera hands over whatever
           is in front of it, trigger included, and the filter crops that.

------------------------------------------------------------------------
Read this before interpreting the number: the crop is a POOR FIT here
------------------------------------------------------------------------
The premise of a crop-based detector is that the crop deletes the trigger.
That holds for DropVLA (a 5px dot in a CORNER, removed by a center crop --
measured: 0% of triggers survive). It does NOT hold for BadVLA, whose
trigger is `add_trigger_img(trigger_size=0.10, trigger_position="center")`
-- a white square covering 10% of the frame, in the MIDDLE, which is
exactly the region a center crop keeps.

So this run is a falsification test, not an expected win: if the score
still separates clean from triggered, it is doing so through something
other than trigger removal. The trigger-survival diagnostic printed per
episode (whether the cropped triggered frame still contains a saturated
central patch) is what distinguishes those cases, and it is recorded in
the results JSON. A near-chance AUROC here is the honest and expected
outcome, and is worth reporting as such.

Everything except the readout and the crop -- model loading (with the
eager-attention fix), the LIBERO rollout, observation prep, the
trigger patch, the Cfg -- is IMPORTED from
attacks/badvla_white_patch/run_ftt_auroc.py rather than reimplemented.
That script's DISJOINT init-state convention is kept too: clean episodes
draw init states [seed, seed+n_seeds) and triggered ones
[seed+n_seeds, seed+2*n_seeds), so no scene appears in both conditions.
Note this differs from DropVLA's paired design (one frame, trigger drawn
or not), so BadVLA's clean and triggered samples differ by scene as well
as by trigger -- a harder setting, and a real difference between the two
attacks' numbers.

Usage (BadVLA's own conda env, same as run_ftt_auroc.py):
    conda activate openvla-oft
    cd /home/grads/nsamptur/vla_bkd_def/BadVLA

    python /home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/attacks/badvla_white_patch/run_crop_hidden_auroc.py \
        --checkpoint <BadVLA white-patch checkpoint> --task-suite-name libero_object \
        --out ../vla-backdoor-defense/results/libero_object/badvla_white_patch_crop_hidden_auroc.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from libero.libero import benchmark

from attacks.badvla_white_patch.run_ftt_auroc import (
    DEVICE,
    NUM_STEPS_WAIT,
    VALID_SUITES,
    load_model,
)
from attacks.common import center_crop_resize, compute_auroc_high_is_triggered, cosine_distance, relative_l2

from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import add_trigger_img, prepare_observation
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size


def capture_last_layer_activations(vla, processor, proprio_projector, cfg, observation, desc,
                                   trigger: bool, trigger_size: float, crop_scale=None):
    """{"all_tokens": [T, d_model], "action_tokens": [56, d_model]} final-layer
    hidden states from one forward pass.

    Identical sequence reconstruction to run_ftt_auroc.py's
    capture_text_to_image_attention (same trigger application, same prompt,
    same proprio and multimodal assembly), with two changes: the forward
    pass asks for hidden states instead of attentions, and `crop_scale`, if
    given, center-crops both camera images AFTER the trigger is drawn.
    """
    full = observation["full_image"].copy()
    wrist = observation["wrist_image"].copy()
    if trigger:
        full = add_trigger_img(full, trigger_size=trigger_size, trigger_position="center", trigger_color=255)
        wrist = add_trigger_img(wrist, trigger_size=trigger_size, trigger_position="center", trigger_color=255)
    if crop_scale is not None:
        full = center_crop_resize(full, crop_scale)
        wrist = center_crop_resize(wrist, crop_scale)
    primary_img, wrist_img = prepare_images_for_vla([full, wrist], cfg)

    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, primary_img).to(DEVICE, dtype=torch.bfloat16)
    wrist_inputs = processor(prompt, wrist_img).to(DEVICE, dtype=torch.bfloat16)
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)
    proprio = normalize_proprio(observation["state"].copy(), vla.norm_stats[cfg.unnorm_key]["proprio"])

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
    n_txt = input_ids.shape[-1] - 1

    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    input_ids2, attention_mask2 = vla._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels2 = vla._prepare_labels_for_action_prediction(labels, input_ids2)
    input_embeddings = vla.get_input_embeddings()(input_ids2)
    all_actions_mask = vla._process_action_masks(labels2)
    language_embeddings = input_embeddings[~all_actions_mask].reshape(
        input_embeddings.shape[0], -1, input_embeddings.shape[2])

    projected = vla._process_vision_features(inputs["pixel_values"], language_embeddings, use_film=False)
    proprio_t = torch.tensor(proprio, device=projected.device, dtype=projected.dtype)
    projected = vla._process_proprio_features(projected, proprio_t, proprio_projector)
    zeroed = input_embeddings * ~all_actions_mask.unsqueeze(-1)
    multimodal_embeds, multimodal_mask = vla._build_multimodal_attention(zeroed, projected, attention_mask2)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla.language_model(
            input_ids=None, attention_mask=multimodal_mask, inputs_embeds=multimodal_embeds,
            output_hidden_states=True, return_dict=True)

    # Same slice as modeling_prismatic's _regression_or_discrete_prediction:
    # the action-token block starts right after the image/proprio columns and
    # the prompt tokens (NUM_PATCHES + NUM_PROMPT_TOKENS in its own naming).
    n_img_cols = projected.shape[1]
    start = n_img_cols + n_txt
    end = start + ACTION_DIM * NUM_ACTIONS_CHUNK
    last_hidden = out.hidden_states[-1]
    assert end <= last_hidden.shape[1], (
        f"action-token slice [{start}:{end}] runs past the sequence "
        f"({last_hidden.shape[1]}) -- token layout drift")
    # all_tokens is the primary readout: EVERY position in the sequence (BOS,
    # both cameras' patches, the proprio token, the prompt and the action
    # tokens), not a chosen subset. There is no padding in this single-sample
    # forward pass, so every position is real. action_tokens is kept as the
    # secondary readout from the same tensor, free of extra compute.
    hidden_all = last_hidden[0].float().cpu().numpy()
    return {"all_tokens": hidden_all, "action_tokens": hidden_all[start:end]}


def trigger_survives_crop(observation, trigger_size: float, crop_scale: float) -> bool:
    """Diagnostic, NOT part of the score: does the white patch still cover the
    centre of the frame after cropping? Compares the mean brightness of the
    central patch-sized window before and after the crop; the patch is drawn
    at colour 255, so if it survives, that window stays saturated."""
    full = add_trigger_img(observation["full_image"].copy(), trigger_size=trigger_size,
                           trigger_position="center", trigger_color=255)
    cropped = center_crop_resize(full, crop_scale)
    h, w = cropped.shape[:2]
    half = max(int(min(h, w) * trigger_size / 2), 1)
    window = cropped[h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half]
    return bool(window.mean() > 250)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-scale", type=float, default=0.8)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7, help="base curated-init-state index")
    ap.add_argument("--trigger-size", type=float, default=0.10)
    args = ap.parse_args()

    cfg, processor, vla, proprio_projector = load_model(args.checkpoint, args.task_suite_name)
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    readouts = ("all_tokens", "action_tokens")
    scores = {r: {m: {"clean": [], "trigger": []} for m in ("cosine_distance", "relative_l2")}
              for r in readouts}
    survived = []
    episodes = []
    cond_offset = {"clean": 0, "trigger": args.n_seeds}

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        n_avail = init_states.shape[0]
        env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        try:
            for cond, trig in (("clean", False), ("trigger", True)):
                for seed_k in range(args.n_seeds):
                    episode_idx = args.seed + cond_offset[cond] + seed_k
                    if episode_idx >= n_avail:
                        print(f"    [!] task={task_id}: only {n_avail} curated init states, "
                              f"skipping {cond} index {episode_idx}")
                        continue

                    env.reset()
                    obs = env.set_init_state(init_states[episode_idx])
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    observation, _ = prepare_observation(obs, resize_size)

                    h_unc = capture_last_layer_activations(
                        vla, processor, proprio_projector, cfg, observation, desc, trig,
                        args.trigger_size, crop_scale=None)
                    h_crop = capture_last_layer_activations(
                        vla, processor, proprio_projector, cfg, observation, desc, trig,
                        args.trigger_size, crop_scale=args.crop_scale)

                    rec = {"task_id": task_id, "init_index": episode_idx, "condition": cond,
                           "label": int(trig)}
                    for r in readouts:
                        cos = cosine_distance(h_unc[r], h_crop[r])
                        rl2 = relative_l2(h_unc[r], h_crop[r])
                        scores[r]["cosine_distance"][cond].append(cos)
                        scores[r]["relative_l2"][cond].append(rl2)
                        rec[f"{r}_cosine_distance"] = cos
                        rec[f"{r}_relative_l2"] = rl2
                    if trig:
                        survived.append(trigger_survives_crop(observation, args.trigger_size, args.crop_scale))
                    episodes.append(rec)
                    print(f"    task={task_id} seed={episode_idx} {cond:8s} "
                          f"all-tok cos-dist={rec['all_tokens_cosine_distance']:.4f}", flush=True)
        finally:
            env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()

    results = {
        "attack": "badvla_white_patch", "checkpoint": args.checkpoint,
        "task_suite_name": args.task_suite_name, "crop_scale": args.crop_scale,
        "trigger_size": args.trigger_size, "trigger_position": "center",
        "readout": "final LLM layer hidden states over the ENTIRE sequence (all_tokens); action-token positions kept as a secondary readout",
        "polarity": "high = triggered",
        "eval_design": "disjoint init states between clean and trigger",
        "trigger_survives_crop_rate": float(np.mean(survived)) if survived else float("nan"),
    }
    for r in readouts:
        results[r] = {}
        for m in ("cosine_distance", "relative_l2"):
            c, t = scores[r][m]["clean"], scores[r][m]["trigger"]
            auroc = compute_auroc_high_is_triggered(c, t)
            results[r][m] = {"n_clean": len(c), "n_trigger": len(t), "auroc": auroc,
                             "clean_mean": float(np.mean(c)) if c else float("nan"),
                             "trigger_mean": float(np.mean(t)) if t else float("nan")}
            print(f"[*] {r:14s} {m:16s}: n_clean={len(c)} n_trigger={len(t)} AUROC={auroc:.4f}  "
                  f"mean clean={results[r][m]['clean_mean']:.4f} trigger={results[r][m]['trigger_mean']:.4f}")
    print(f"[*] white patch still present after the crop in "
          f"{results['trigger_survives_crop_rate']:.0%} of triggered frames")
    results["episodes"] = episodes

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
