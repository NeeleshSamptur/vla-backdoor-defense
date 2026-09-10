#!/usr/bin/env python
"""GoBA (physical toxic-box trigger): LAST-LAYER ACTIVATION shift under a
center crop -> AUROC. Same detector as attacks/dropvla/run_crop_hidden_auroc.py,
ported to this attack.

    score(x) = mean over the readout positions of
               1 - cos( h(x)[i], h(crop(x))[i] )        HIGH = triggered

------------------------------------------------------------------------
The readout is NOT the same tensor as the OFT attacks' -- and cannot be
------------------------------------------------------------------------
DropVLA, BadVLA and BackdoorVLA are OpenVLA-OFT: their sequence carries 56
dedicated action-token positions whose final-layer states go straight into
an L1 action head, so "the last layer's action activations" is unambiguous.
GoBA is BASE OpenVLA with discrete action tokens and no action head -- its
7 action tokens are generated autoregressively, one at a time, and none of
them exists in the single forward pass this script runs. So the closest
honest analogue is used instead, and two variants are reported:

  final_position -- the final-layer state at the LAST prompt position, i.e.
                    the state the first action token is actually decoded
                    from. [1, d_model]. This is the primary number.
  desc_tokens    -- the final-layer states at the task-description token
                    positions, the same query rows this attack's FTT scores.
                    [n_desc_tokens, d_model].

Both are read from one forward pass with output_hidden_states=True -- the
same pass run_ftt_auroc.py already builds for its attention, so nothing
about the model, prompt or preprocessing differs from it. Cross-attack
comparisons should quote this difference in readout rather than treat the
number as measured on identical footing.

------------------------------------------------------------------------
Read this before interpreting the number: the crop cannot remove this
trigger, and the two conditions differ by SCENE
------------------------------------------------------------------------
GoBA's trigger is a physical toxic box placed in the scene by a separate
poisoned BDDL. It stands on the table among the other objects, at a
position that varies per scene, so a center crop does not remove it the way
it removes DropVLA's corner dot. Expect near-chance separation.

Worse for interpretation: clean and triggered episodes come from DIFFERENT
BDDL FILES (CLEAN_BDDL vs POISON_BDDL) with different reset draws, so the
two classes differ by scene content as well as by trigger. Any separation
this script reports could come from either, and this design cannot tell
them apart. It is inherited unchanged from run_ftt_auroc.py so the two
detectors are compared on identical episodes.

Note also that GoBA's own pipeline ALREADY center-crops every frame
(crop_scale 0.9, asserted for image_aug checkpoints), so --crop-scale here
composes with that: the detector's crop is applied first, then GoBA's own.

Usage (GoBA's own env, same as run_ftt_auroc.py):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../attacks/goba/run_crop_hidden_auroc.py \
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
        --task-suite-name libero_object \
        --out ../vla-backdoor-defense/results/libero_object/goba_crop_hidden_auroc.json
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

from attacks.goba.run_ftt_auroc import (
    CLEAN_BDDL,
    DEVICE,
    NUM_STEPS_WAIT,
    POISON_BDDL,
    VALID_SUITES,
    Cfg,
    build_observation,
    desc_token_rows,
    load_model,
    preprocess_frame,
)
from attacks.common import center_crop_resize, compute_auroc_high_is_triggered, cosine_distance, relative_l2

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere


def capture_last_layer_activations(vla, processor, image, desc, center_crop=True, crop_scale=None):
    """(all_tokens, final_position, desc_tokens) final-layer hidden states.

    all_tokens is the primary readout: EVERY position of the sequence (BOS,
    the image patches and the prompt tokens), no subset chosen. This
    single-sample forward pass has no padding, so every position is real.

    Same forward pass as run_ftt_auroc.py's capture_text_to_image_attention
    -- same preprocessing, prompt and 29871 sentinel handling -- with
    output_hidden_states in place of output_attentions. `crop_scale`, if
    given, center-crops the frame BEFORE GoBA's own preprocessing crop.
    """
    if crop_scale is not None:
        image = center_crop_resize(image, crop_scale)
    img = preprocess_frame(image, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.full((input_ids.shape[0], 1), 29871,
                                   dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=input_ids, attention_mask=attention_mask,
                  pixel_values=inputs["pixel_values"], output_hidden_states=True,
                  return_dict=True)

    last_hidden = out.hidden_states[-1][0]  # [T, d_model]
    n_txt = input_ids.shape[1] - 1
    T = last_hidden.shape[0]
    num_patches = T - n_txt - 1
    assert num_patches > 0, f"bad token layout: T={T} n_txt={n_txt}"

    all_tokens = last_hidden.float().cpu().numpy()           # [T, d]
    final_position = last_hidden[-1:].float().cpu().numpy()  # [1, d]
    txt_rel = desc_token_rows(processor, prompt, desc, n_txt)
    txt_rows = [1 + num_patches + r for r in txt_rel]
    desc_tokens = last_hidden[txt_rows].float().cpu().numpy()  # [n_desc, d]

    del out
    torch.cuda.empty_cache()
    return all_tokens, final_position, desc_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-scale", type=float, default=0.8)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint",
                    help="same meaning as in run_ftt_auroc.py.")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint} (suite={args.task_suite_name}, crop_scale={args.crop_scale})")
    vla = load_model(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)
    cond_offset = ({"clean": 0, "trigger": 0} if args.eval_design == "paired"
                   else {"clean": 0, "trigger": args.n_seeds})

    readouts = ("all_tokens", "final_position", "desc_tokens")
    scores = {r: {m: {"clean": [], "trigger": []} for m in ("cosine_distance", "relative_l2")}
              for r in readouts}
    episodes = []

    # One env alive at a time -- see run_ftt_auroc.py: a second concurrent
    # OffScreenRenderEnv renders ~40% darker, which would read as a large
    # fake clean-vs-trigger effect.
    for cond, bddl_dir in (("clean", CLEAN_BDDL), ("trigger", POISON_BDDL)):
        print(f"[*] === {cond} scenes (bddl={bddl_dir}) ===")
        for task_id in range(n_tasks):
            task = suite.get_task(task_id)
            env, desc = get_libero_env(task, cfg.model_family, resolution=256,
                                       bddl_path=bddl_dir, seed=args.seed)
            try:
                for _ in range(cond_offset[cond]):
                    env.reset()
                for ep in range(args.n_seeds):
                    env.reset()
                    obs = None
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    _, img = build_observation(obs, resize_size)

                    unc = capture_last_layer_activations(vla, processor, img, desc,
                                                         center_crop=cfg.center_crop, crop_scale=None)
                    crp = capture_last_layer_activations(vla, processor, img, desc,
                                                         center_crop=cfg.center_crop,
                                                         crop_scale=args.crop_scale)
                    ep_idx = cond_offset[cond] + ep
                    rec = {"task_id": task_id, "seed": ep_idx, "condition": cond,
                           "label": int(cond == "trigger")}
                    for ri, r in enumerate(readouts):
                        cos = cosine_distance(unc[ri], crp[ri])
                        rl2 = relative_l2(unc[ri], crp[ri])
                        scores[r]["cosine_distance"][cond].append(cos)
                        scores[r]["relative_l2"][cond].append(rl2)
                        rec[f"{r}_cosine_distance"] = cos
                        rec[f"{r}_relative_l2"] = rl2
                    episodes.append(rec)
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} "
                          f"all-tok cos-dist={rec['all_tokens_cosine_distance']:.4f} "
                          f"final-pos={rec['final_position_cosine_distance']:.4f}", flush=True)
            finally:
                env.close()

    del vla
    torch.cuda.empty_cache()

    results = {
        "attack": "goba", "checkpoint": args.checkpoint,
        "task_suite_name": args.task_suite_name, "crop_scale": args.crop_scale,
        "eval_design": args.eval_design,
        "trigger": "physical toxic box placed in the scene by a poisoned BDDL",
        "readout": "final LLM layer over the ENTIRE sequence (all_tokens, primary); base "
                   "OpenVLA has no action-token positions (see module docstring), so "
                   "final_position and desc_tokens are also reported",
        "polarity": "high = triggered",
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
