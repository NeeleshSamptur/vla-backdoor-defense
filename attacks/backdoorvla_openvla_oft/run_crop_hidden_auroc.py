#!/usr/bin/env python
"""AttackVLA/BackdoorVLA (popcorn-object + "~*magic*~" text trigger):
LAST-LAYER ACTIVATION shift under a center crop -> AUROC. Same detector as
attacks/dropvla/run_crop_hidden_auroc.py, ported to this attack.

    score(x) = mean over the action-token positions of
               1 - cos( h(x)[i], h(crop(x))[i] )        HIGH = triggered

    h(.)   final LLM layer hidden states at the action-token positions,
           from the SAME reconstructed forward pass this attack's
           run_ftt_auroc.py builds for its attention -- output_hidden_states
           replaces output_attentions. That slice is what
           OpenVLAForActionPrediction feeds to the action head.
    crop   attacks.common.center_crop_resize at --crop-scale, both cameras.

------------------------------------------------------------------------
Read this before interpreting the number: the crop cannot remove EITHER
half of this trigger
------------------------------------------------------------------------
This attack's trigger is bi-modal and neither half is croppable:
  image half -- the popcorn_1 OBJECT physically placed in the scene by the
                *_poisoned BDDL. It sits on the table among the other
                objects, not at the frame border, and the policy sees it
                from wherever the camera is; a center crop keeps the table.
  text half  -- the "~*magic*~ " prefix on the instruction. No image
                transform touches text at all.
Unlike DropVLA (5px corner dot, deleted by the crop) this is therefore
expected to score near chance, and that expectation is the point of
running it: it bounds what the crop detector can claim. A high AUROC here
would need explaining, not celebrating, because the clean and triggered
episodes also come from DIFFERENT BDDL scenes (see below), which is an
alternative source of separation that has nothing to do with the trigger.

Everything except the readout and the crop -- model loading (with the
eager-attention fix), the poisoned/clean BDDL and init-state files, the
scene rollout, observation prep, the magic prefix, the Cfg -- is IMPORTED
from attacks/backdoorvla_openvla_oft/run_ftt_auroc.py. Its evaluation
design is kept unchanged: every episode is rolled out in target task 0's
scene, clean episodes use the clean BDDL + bare instruction and triggered
ones the poisoned BDDL (which places popcorn_1) + prefixed instruction,
with DISJOINT init-state indices between the two conditions.

Usage (AttackVLA's own env, same as run_ftt_auroc.py):
    conda activate openvla-oft
    export PYTHONPATH=/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack

    python .../attacks/backdoorvla_openvla_oft/run_crop_hidden_auroc.py \
        --checkpoint Text_Image_Attack/object_TI_4/15000--49999_chkpt \
        --out ../vla-backdoor-defense/results/libero_object/backdoorvla_openvla_oft_crop_hidden_auroc.json
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
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK

from attacks.backdoorvla_openvla_oft.run_ftt_auroc import (
    BDDL_ROOT,
    CLEAN_SUITE,
    DEVICE,
    INIT_ROOT,
    MAGIC_PREFIX,
    NUM_STEPS_WAIT,
    SENTINEL_TOKEN_ID,
    TARGET_TASK_BDDL,
    Cfg,
    load_init_states,
    load_model,
)
from attacks.common import center_crop_resize, compute_auroc_high_is_triggered, cosine_distance, relative_l2

from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size


def capture_last_layer_activations(vla, processor, proprio_projector, cfg: Cfg, observation,
                                   prompt_desc: str, crop_scale=None):
    """{"all_tokens": [T, d_model], "action_tokens": [56, d_model]} final-layer
    hidden states from one forward pass.

    Identical assembly to run_ftt_auroc.py's capture_text_to_image_attention
    (same prompt, proprio and multimodal sequence; `prompt_desc` already
    carries the magic prefix on triggered episodes), with output_hidden_states
    in place of output_attentions and an optional center crop of both cameras.
    """
    full = observation["full_image"].copy()
    wrist = observation["wrist_image"].copy()
    if crop_scale is not None:
        full = center_crop_resize(full, crop_scale)
        wrist = center_crop_resize(wrist, crop_scale)
    images = prepare_images_for_vla([full, wrist], cfg)
    prompt = f"In: What action should the robot take to {prompt_desc.lower()}?\nOut:"
    inputs = processor(prompt, images[0]).to(DEVICE, dtype=torch.bfloat16)
    wrist_in = processor(prompt, images[1]).to(DEVICE, dtype=torch.bfloat16)
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_in["pixel_values"]], dim=1)
    proprio = normalize_proprio(observation["state"].copy(), vla.norm_stats[cfg.unnorm_key]["proprio"])

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == SENTINEL_TOKEN_ID):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[SENTINEL_TOKEN_ID]], dtype=torch.long, device=input_ids.device)), dim=1)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Text_Image_Attack/object_TI_4/15000--49999_chkpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-scale", type=float, default=0.8)
    ap.add_argument("--n-instructions", type=int, default=9,
                    help="non-target libero_object tasks (9 of 10), matching run_ftt_auroc.py")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7, help="base curated-init-state index")
    args = ap.parse_args()

    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    cfg = Cfg(pretrained_checkpoint=args.checkpoint)
    print(f"[*] loading {args.checkpoint}")
    processor, vla, proprio_projector = load_model(args.checkpoint, cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[CLEAN_SUITE]()
    task_names = [suite.get_task(i).language for i in range(1, args.n_instructions + 1)]
    print(f"[*] {len(task_names)} non-target instructions")

    readouts = ("all_tokens", "action_tokens")
    scores = {r: {m: {"clean": [], "trigger": []} for m in ("cosine_distance", "relative_l2")}
              for r in readouts}
    episodes = []
    cond_offset = {"clean": 0, "trigger": args.n_seeds}

    for cond, use_magic in (("clean", False), ("trigger", True)):
        bddl_subdir = "libero_object_poisoned" if use_magic else "libero_object"
        bddl_file = os.path.join(BDDL_ROOT, bddl_subdir, f"{TARGET_TASK_BDDL}.bddl")
        init_states = load_init_states(os.path.join(INIT_ROOT, bddl_subdir))
        n_avail = init_states.shape[0]
        print(f"[*] === {cond}: bddl={bddl_file} init_states={init_states.shape} ===")

        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=cfg.env_img_res,
                                 camera_widths=cfg.env_img_res)
        env.seed(0)
        try:
            for t_idx, instruction in enumerate(task_names):
                prompt_desc = (MAGIC_PREFIX + instruction) if use_magic else instruction
                for seed_k in range(args.n_seeds):
                    idx = args.seed + cond_offset[cond] + seed_k
                    if idx >= n_avail:
                        print(f"    [!] only {n_avail} init states, skipping idx {idx}")
                        continue

                    env.reset()
                    obs = env.set_init_state(init_states[idx])
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    observation, _ = prepare_observation(obs, resize_size)

                    h_unc = capture_last_layer_activations(
                        vla, processor, proprio_projector, cfg, observation, prompt_desc, crop_scale=None)
                    h_crop = capture_last_layer_activations(
                        vla, processor, proprio_projector, cfg, observation, prompt_desc,
                        crop_scale=args.crop_scale)

                    rec = {"instruction_index": t_idx, "init_index": idx, "condition": cond,
                           "label": int(use_magic)}
                    for r in readouts:
                        cos = cosine_distance(h_unc[r], h_crop[r])
                        rl2 = relative_l2(h_unc[r], h_crop[r])
                        scores[r]["cosine_distance"][cond].append(cos)
                        scores[r]["relative_l2"][cond].append(rl2)
                        rec[f"{r}_cosine_distance"] = cos
                        rec[f"{r}_relative_l2"] = rl2
                    episodes.append(rec)
                    print(f"    t={t_idx} s={idx} {cond:8s} "
                          f"all-tok cos-dist={rec['all_tokens_cosine_distance']:.4f}", flush=True)
        finally:
            env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()

    results = {
        "attack": "backdoorvla_openvla_oft", "checkpoint": args.checkpoint,
        "task_suite_name": CLEAN_SUITE, "crop_scale": args.crop_scale,
        "trigger": "popcorn_1 object in the scene + '~*magic*~ ' instruction prefix",
        "readout": "final LLM layer hidden states over the ENTIRE sequence (all_tokens); action-token positions kept as a secondary readout",
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
            print(f"[*] {r:14s} {m:16s}: n_clean={len(c)} n_trigger={len(t)} AUROC={auroc:.4f}  "
                  f"mean clean={results[r][m]['clean_mean']:.4f} trigger={results[r][m]['trigger_mean']:.4f}")
    results["episodes"] = episodes

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
