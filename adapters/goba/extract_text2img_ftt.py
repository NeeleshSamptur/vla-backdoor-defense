#!/usr/bin/env python
"""GoBA extractor: text->image attention rows for the FTT detector.

Mirrors adapters/badvla_white_patch/extract_text2img_ftt.py's Stage 1
(one attention map per episode after the eval settle, saved via
detectors/schema.py) -- but every environment/eval detail is
taken from GOBA'S OWN validated eval, not BadVLA's. The two attacks differ in
ways that matter, all verified by reading GoBA's source:

  1. TRIGGER IS PHYSICAL, NOT A PIXEL PATCH. GoBA places a toxic-box object
     (poison_1) in the scene via a SEPARATE BDDL directory. There is no
     add_trigger_img equivalent -- clean and trigger are genuinely different
     scenes, from different scene definitions:
         clean   -> BadLIBERO/libero/libero/bddl_files        (stock)
         trigger -> BadLIBERO/libero/libero/bddl_files-poison_eval
     Confirmed from goba-scripts/run_all_suites_campaign.sh: clean SR(w/o)
     runs run_libero_eval.py on the stock suite, FR(w)+three-level runs
     3level_eval.py with --bddl_dir "${POISON_BDDL}".

  2. NO set_init_state. Both of GoBA's eval scripts have
     `# obs = env.set_init_state(initial_states[episode_idx])` COMMENTED OUT
     (run_libero_eval.py:180, 3level_eval.py:272) and rely on repeated
     env.reset() from an env seeded once at construction. This is not an
     oversight to "fix": the poison BDDL adds an extra object, so its raw
     MuJoCo state vector has a different shape than the clean scene's and
     set_init_state cannot be shared across the two variants. BadVLA's
     adapter DOES use curated set_init_state because BadVLA's trigger is a
     pixel overlay on an otherwise identical scene -- that difference is
     intrinsic to the attacks, not an inconsistency between these adapters.

  3. BASE OpenVLA, NOT OFT. One forward pass yields ONE action. Stage 1
     only reads the first policy query after settle (same as BadVLA).

CRITICAL RENDERING NOTE -- do not instantiate two envs at once. Holding two
OffScreenRenderEnv objects open simultaneously corrupts the render of the
first one: it loses lighting and comes back dark (measured: mean luminance
69.3 vs 117.0), which shows up as a ~79% whole-frame difference between
"clean" and "trigger" and would be read as an enormous trigger effect. It is
an artifact. This script therefore finishes and CLOSES each env before
opening the next, exactly one alive at any time.

Usage (mirrors GoBA's own campaign env setup):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../adapters/goba/extract_text2img_ftt.py \
        --checkpoint exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
        --task-suite-name libero_goal --role attack \
        --out-dir ../vla-backdoor-defense/results/goba_extracted

    # clean-model control (same scenes, non-backdoored checkpoint):
    python .../adapters/goba/extract_text2img_ftt.py \
        --checkpoint openvla/openvla-7b-finetuned-libero-goal \
        --task-suite-name libero_goal --role clean_baseline \
        --out-dir ../vla-backdoor-defense/results/goba_extracted

    # all suites, both roles: see run_all_suites.sh
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)

import json

import numpy as np
import torch
from libero.libero import benchmark
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle,
)
from experiments.robot.openvla_utils import crop_and_resize, get_processor
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

from detectors.schema import ExtractedSample  # from the new repo, added to sys.path above

DEVICE = "cuda:0"
NUM_STEPS_WAIT = 10
VALID_SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")

# Paths relative to the GoBA repo root (this script is run from there).
CLEAN_BDDL = "BadLIBERO/libero/libero/bddl_files"
POISON_BDDL = "BadLIBERO/libero/libero/bddl_files-poison_eval"


@dataclass
class Cfg:
    pretrained_checkpoint: str
    model_family: str = "openvla"
    # GoBA's eval asserts center_crop==True (3level_eval.py:185) because these
    # checkpoints were trained with image augmentation.
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    unnorm_key: str = ""  # set in main() from --task-suite-name


def load_vla_for_attention(cfg):
    """GoBA's own get_vla(), with ONE necessary change: SDPA attention.

    GoBA's experiments.robot.openvla_utils.get_vla() hardcodes
    attn_implementation="flash_attention_2". FlashAttention does not
    materialise an attention matrix and does NOT fall back when
    output_attentions=True is requested -- it silently returns
    out.attentions = None, so FTT has nothing to read.

    SDPA, not eager, is the right replacement, for TWO reasons:

      1. output_attentions: SDPA prints a warning and automatically falls back
         to the manual attention path, yielding real attention matrices. This
         is exactly what BadVLA's checkpoints already do (they load under SDPA
         by default), so both adapters capture attention the same way.

      2. predict_action compatibility: OpenVLAForActionPrediction.predict_action
         appends the 29871 token to input_ids but passes attention_mask through
         to generate() UNCHANGED (modeling_prismatic.py:512-518). Under eager
         attention that off-by-one is fatal --
           "size of tensor a (279) must match tensor b (278)"
         inside Llama's additive causal mask. Under flash_attention_2 (GoBA's
         default) and under SDPA it is handled.

    Everything else here is copied from GoBA's get_vla verbatim -- same Auto*
    registrations, same dtype/low_cpu_mem_usage/trust_remote_code, same device
    move, same dataset_statistics.json handling for norm_stats. The attention
    implementation changes how attention is computed, not what the weights are.

    DISCLOSE THIS IN THE PAPER: GoBA's published ASR/SR numbers were produced
    under flash_attention_2. Frame 0 of every episode here is still the same
    settled scene (deterministic given the BDDL + reset sequence). Attention is
    captured under SDPA (required for output_attentions), not FA2.
    """
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with SDPA attention (needed for output_attentions)")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="sdpa",  # <-- the only deviation from get_vla()
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    stats_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            vla.norm_stats = json.load(f)
    else:
        # Expected for HF-hub checkpoints (the clean_baseline control), which
        # ship norm_stats inside the model config itself.
        print("[*] no local dataset_statistics.json; using norm_stats from config")

    vla.eval()
    return vla


def preprocess_like_policy(image, center_crop):
    """Reproduce get_vla_action's image preprocessing EXACTLY.

    CRITICAL FOR FIDELITY: GoBA's get_vla_action (openvla_utils.py:127-156)
    center-crops to crop_scale=0.9 and resizes back whenever center_crop=True
    -- which 3level_eval.py:185 *asserts* is True for these checkpoints, since
    they were trained with image augmentation. The policy therefore acts on a
    cropped frame.

    An earlier version of this file fed the RAW get_libero_image output
    straight to the processor for the attention pass, so FTT was computed on a
    different image than the policy ever saw: full frame for the statistic,
    center-cropped for the actions. That matters here more than it might
    sound -- GoBA's trigger is a physical object whose position in frame
    varies, and a 0.9-area crop can clip content near the edges, so the
    attention map could include trigger pixels the policy never received (or
    weight them differently). BadVLA's adapter never had this bug because it
    routes its attention pass through prepare_images_for_vla, which applies
    the same crop. This restores parity.

    Uses GoBA's own crop_and_resize so the crop is bit-identical to eval,
    not a re-implementation.
    """
    import tensorflow as tf

    pil = Image.fromarray(image).convert("RGB")
    if center_crop:
        batch_size = 1
        crop_scale = 0.9
        t = tf.convert_to_tensor(np.array(pil))
        orig_dtype = t.dtype
        t = tf.image.convert_image_dtype(t, tf.float32)
        t = crop_and_resize(t, crop_scale, batch_size)
        t = tf.clip_by_value(t, 0, 1)
        t = tf.image.convert_image_dtype(t, orig_dtype, saturate=True)
        pil = Image.fromarray(t.numpy()).convert("RGB")
    return pil


def text2img_rows(vla, processor, image, desc, layer=-1, center_crop=True):
    """Text-token attention rows over image-patch columns, one forward pass.

    GoBA's base OpenVLA has no OFT-specific embedding-assembly methods, so the
    model's own forward() does the multimodal fusion internally; this is a
    plain HF call with output_attentions=True.

    Token layout, from PrismaticForConditionalGeneration.forward:
        multimodal_embeddings = cat([ input_embeddings[:, :1, :],   # BOS
                                       projected_patch_embeddings,   # image
                                       input_embeddings[:, 1:, :] ]) # prompt
    so image columns = range(1, 1+num_patches), text rows follow them.

    Single camera here (base OpenVLA takes one third-person image), so there
    is no camera-selection question -- unlike BadVLA's dual-camera OFT.
    """
    img = preprocess_like_policy(image, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Append the special empty token (29871), exactly as
    # OpenVLAForActionPrediction.predict_action does (modeling_prismatic.py:512)
    # to "match the inputs seen at training time". predict_action gets away
    # with appending to input_ids only, because generate() rebuilds the mask
    # internally; a direct forward() call does NOT, and the resulting
    # off-by-one surfaces as
    #   "size of tensor a (279) must match tensor b (278)"
    # inside Llama's eager causal-mask add. So extend BOTH here.
    #
    # This is also the fidelity-correct choice, not just a crash fix: the
    # policy really does see this trailing token at inference, so the
    # attention we capture must be for that same prompt. BadVLA's adapter
    # performs the identical append.
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids,
             torch.full((input_ids.shape[0], 1), 29871,
                        dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        attention_mask = torch.cat(
            (attention_mask,
             torch.ones((attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=input_ids, attention_mask=attention_mask,
                  pixel_values=inputs["pixel_values"], output_attentions=True,
                  return_dict=True)

    if out.attentions is None:
        raise RuntimeError(
            "model returned no attentions -- it was almost certainly loaded with "
            "flash_attention_2, which ignores output_attentions=True instead of "
            "falling back. Load via load_vla_for_attention() (SDPA attention).")
    A = out.attentions[layer][0].float().mean(0)

    # Derive num_patches from the sequence length rather than calling
    # vision_backbone.get_num_patches() -- that method exists only on the OFT
    # variant BadVLA uses; base OpenVLA's PrismaticVisionBackbone has no such
    # attribute. From PrismaticForConditionalGeneration.forward the fused
    # sequence is cat([emb[:, :1], patches, emb[:, 1:]]), i.e.
    #   T = 1 + num_patches + (L - 1) = num_patches + n_txt + 1
    # so num_patches falls out exactly, with no architecture-specific API.
    n_txt = input_ids.shape[1] - 1
    T = A.shape[-1]
    num_patches = T - n_txt - 1
    assert num_patches > 0, f"bad token layout: T={T} n_txt={n_txt}"

    img_cols = list(range(1, 1 + num_patches))
    txt_rows = list(range(1 + num_patches, 1 + num_patches + n_txt))
    rows = A[txt_rows][:, img_cols].cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return rows, num_patches


def build_observation(obs, resize_size):
    """Exactly GoBA's own observation dict (3level_eval.py:329-341)."""
    img = get_libero_image(obs, resize_size)
    return {
        "full_image": img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
             obs["robot0_gripper_qpos"])
        ),
    }, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10,
                    help="tasks per suite (LIBERO suites have 10). MUST stay in "
                         "lockstep with adapters/badvla_white_patch and run_all_suites.sh.")
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="episodes per task per condition. MUST stay in lockstep "
                         "with adapters/badvla_white_patch (same default) and with "
                         "run_all_suites.sh's ${N_SEEDS:-10}.")
    ap.add_argument("--seed", type=int, default=7,
                    help="env construction seed; GoBA's eval.sh sweeps 7/42/1234")
    # Closed-loop extract (5 policy passes / episode):
    # ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint",
                    help="'disjoint' (DEFAULT, matches badvla_white_patch): clean and "
                         "trigger episodes are drawn from different offsets in the "
                         "env's reset sequence, so they never correspond to the same "
                         "scene draw. Note GoBA is ALREADY inherently disjoint -- "
                         "clean and trigger come from different BDDL files with "
                         "different object sets -- so this offset is belt-and-braces, "
                         "not the primary source of scene difference. 'paired' uses "
                         "the same reset offset for both; still not literally the "
                         "same scene, for the BDDL reason above.")
    ap.add_argument("--layer", type=int, default=-1)
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)

    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name})")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    # GoBA's own unnorm_key fallback (3level_eval.py:201-203).
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in vla.norm_stats, (
        f"unnorm key {cfg.unnorm_key} not in norm_stats: {list(vla.norm_stats)[:5]}")

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    _raw = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    ckpt_tag = f"{_raw[:40]}_{hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]}"
    out_dir = Path(args.out_dir)

    cond_offset = ({"clean": 0, "trigger": 0} if args.eval_design == "paired"
                   else {"clean": 0, "trigger": args.n_seeds})

    # ONE env alive at a time -- see CRITICAL RENDERING NOTE. Conditions are
    # done in full, one after the other, and each task's env is closed before
    # the next opens.
    for cond, bddl_dir in (("clean", CLEAN_BDDL), ("trigger", POISON_BDDL)):
        trig = cond == "trigger"
        print(f"[*] === {cond} scenes (bddl={bddl_dir}) ===")
        for task_id in range(n_tasks):
            task = suite.get_task(task_id)
            # Matches GoBA's eval: env built once per task, with the seed and
            # bddl_path passed at construction; episodes then just env.reset().
            env, desc = get_libero_env(task, cfg.model_family, resolution=256,
                                       bddl_path=bddl_dir, seed=args.seed)
            try:
                # Advance the reset sequence to this condition's offset so
                # clean and trigger never read the same draw (see --eval-design).
                for _ in range(cond_offset[cond]):
                    env.reset()

                for ep in range(args.n_seeds):
                    env.reset()
                    obs = None
                    # GoBA's eval settles the scene with no-op steps before
                    # querying the policy (3level_eval.py:320-321).
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

                    observation, img = build_observation(obs, resize_size)
                    rows, num_patches = text2img_rows(vla, processor, img, desc,
                                                      layer=args.layer,
                                                      center_crop=cfg.center_crop)

                    ep_idx = cond_offset[cond] + ep
                    ExtractedSample(
                        attn_text_image=rows,
                        label=int(trig),
                        attack="goba",
                        checkpoint=args.checkpoint,
                        trigger_type="physical_toxic_box" if trig else "none",
                        task_id=task_id, seed=ep_idx, layer=args.layer,
                        n_cameras=1, patches_per_camera=num_patches,
                        episode_id=(f"{args.task_suite_name}__t{task_id}"
                                    f"__seed{args.seed}__s{ep_idx}__{cond}"),
                        frame_idx=0,
                        extra={"role": args.role,
                               "task_suite_name": args.task_suite_name,
                               "eval_design": args.eval_design,
                               "bddl_dir": bddl_dir,
                               "env_seed": args.seed},
                    ).save(str(out_dir / f"{ckpt_tag}__{args.task_suite_name}"
                                         f"__t{task_id}__seed{args.seed}__s{ep_idx}"
                                         f"__{cond}.npz"))
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} "
                          f"rows={rows.shape}")
                    # --- old 5-pass closed-loop extract (kept for reference) ---
                    # from experiments.robot.robot_utils import (
                    #     get_action, invert_gripper_action, normalize_gripper_action)
                    # frames_rows = []
                    # for pass_idx in range(args.n_frames):
                    #     observation, img = build_observation(obs, resize_size)
                    #     rows, num_patches = text2img_rows(
                    #         vla, processor, img, desc, layer=args.layer,
                    #         center_crop=cfg.center_crop)
                    #     frames_rows.append(rows)
                    #     if pass_idx == args.n_frames - 1:
                    #         break
                    #     action = get_action(cfg, vla, observation, desc,
                    #                         processor=processor)
                    #     action = normalize_gripper_action(action, binarize=True)
                    #     if cfg.model_family == "openvla":
                    #         action = invert_gripper_action(action)
                    #     obs, _, done, _ = env.step(action.tolist())
                    #     if done:
                    #         break
                    # for frame_idx, rows in enumerate(frames_rows):
                    #     ExtractedSample(... frame_idx=frame_idx ...).save(
                    #         ... f"__{cond}__f{frame_idx}.npz")
            finally:
                env.close()

    del vla, processor
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
