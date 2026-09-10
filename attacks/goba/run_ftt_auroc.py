#!/usr/bin/env python
"""GoBA: task-description text-to-image attention FTT, scored against AUROC.

One episode = one forward pass of the fine-tuned OpenVLA policy on a single
LIBERO frame. Query = task-description tokens only (template/BOS/appended
sentinel dropped); key = image-patch columns only. Every LLM layer's
head-averaged text2img attention is fed to attacks.common.compute_ftt(), and
clean vs. trigger scores are compared with attacks.common.compute_auroc().

Reuses GoBA's own checkpoint loading, LIBERO env construction, and image/
prompt preprocessing verbatim (see load_model() and preprocess_frame()
docstrings for exactly which single detail was changed and why: eager
attention instead of GoBA's flash_attention_2, because FA2 never returns
attention weights). Everything else -- registrations, dtype, device
placement, dataset_statistics.json handling, center-crop scale, prompt
template, BDDL-based trigger mechanism -- is unchanged from GoBA's own code
in experiments/robot/{openvla_utils,libero/libero_utils}.py and
experiments/robot/libero/3level_eval.py.

GoBA's trigger is a physical object (a toxic box) placed in the scene via a
separate BDDL directory, not a pixel or prompt perturbation, so "clean" and
"trigger" here are two different scene definitions sharing the same task
descriptions -- confirmed identical by GoBA's own (:language ...) BDDL
strings for every task in a suite.

Usage (same environment GoBA's own eval scripts require):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../attacks/goba/run_ftt_auroc.py \
        --checkpoint exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
        --task-suite-name libero_goal \
        --out ../vla-backdoor-defense/results/goba_ftt_auroc.json

Default --n-tasks 10 x --n-seeds 10 gives 100 clean + 100 trigger episodes
for one task suite (LIBERO suites have 10 tasks each), matching the episode
count this project's other attacks' run_ftt_auroc.py scripts use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)

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

from attacks.common import compute_auroc, compute_ftt  # noqa: E402

DEVICE = "cuda:0"
NUM_STEPS_WAIT = 10  # GoBA's eval settles the scene with no-op steps first (3level_eval.py:325-329)
VALID_SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")

# Paths relative to the GoBA repo root (this script is run from there, like
# GoBA's own eval scripts). CLEAN_BDDL is get_libero_env's own default
# (libero_utils.py:19). POISON_BDDL and the trigger object name "poison_1"
# come from goba-scripts/run_all_suites_campaign.sh, which is what actually
# produced these checkpoints' published FR/three-level numbers -- 3level_eval.py's
# OWN built-in defaults (bddl_files-3trigger, trigger_obj=cookies_1) are a
# leftover Section-5 ablation config, not what Table 2 was measured under.
CLEAN_BDDL = "BadLIBERO/libero/libero/bddl_files"
POISON_BDDL = "BadLIBERO/libero/libero/bddl_files-poison_eval"


@dataclass
class Cfg:
    pretrained_checkpoint: str
    model_family: str = "openvla"
    # GoBA's eval asserts center_crop==True whenever "image_aug" is in the
    # checkpoint path (3level_eval.py:187-188) -- true of every checkpoint
    # this campaign trains, since image_aug=True is part of its fixed recipe.
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    unnorm_key: str = ""  # set in main() from --task-suite-name


def load_model(cfg: Cfg):
    """GoBA's own get_vla() (experiments/robot/openvla_utils.py:31-73),
    with exactly one change: eager attention instead of flash_attention_2.

    FA2 never materializes an attention matrix and silently returns
    out.attentions=None when output_attentions=True is requested, so it
    cannot be used for extraction at all. Eager is the correct replacement
    (not SDPA): under transformers 4.40.1, SDPA + output_attentions=True
    falls back to an eager attention body but is handed no causal mask in
    this single-forward-pass/no-padding/no-cache setup, so it silently
    returns BIDIRECTIONAL attention instead of the causal attention the
    model actually used to produce its output (empirically confirmed:
    image-patch query rows had nonzero mass on "future" columns that must be
    exactly 0 under real causal masking). Eager's own forward path always
    applies the real 4D causal mask, so this is the one implementation that
    is both extractable AND correct.

    This only changes how attention is captured for THIS diagnostic -- the
    policy's actual actions (and every published GoBA ASR/SR number) come
    from predict_action() -> generate(), which never requests
    output_attentions and always runs the real fused FA2/SDPA kernel.

    Registrations, dtype, device move, and dataset_statistics.json handling
    are otherwise verbatim from get_vla().
    """
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with EAGER attention (get_vla's flash_attention_2 "
          "never returns attention weights; SDPA silently returns bidirectional "
          "attention under output_attentions=True -- see load_model's docstring)")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="eager",  # the only deviation from get_vla(); see docstring
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
        # Expected for HF-hub checkpoints (a non-backdoored control), which
        # ship norm_stats inside the model config itself.
        print("[*] no local dataset_statistics.json; using norm_stats from config")

    vla.eval()
    return vla


def preprocess_frame(image, center_crop):
    """Reproduce get_vla_action's image preprocessing exactly: center-crop to
    crop_scale=0.9 and resize back, using GoBA's own crop_and_resize
    (experiments/robot/openvla_utils.py:81) rather than reimplementing it.
    3level_eval.py asserts center_crop=True for image_aug checkpoints, and
    the attention pass has to see the same crop the policy actually acted on
    -- GoBA's trigger is a physical object whose frame position varies, and a
    0.9-area crop can clip content near the edges.
    """
    import tensorflow as tf

    pil = Image.fromarray(image).convert("RGB")
    if center_crop:
        t = tf.convert_to_tensor(np.array(pil))
        orig_dtype = t.dtype
        t = tf.image.convert_image_dtype(t, tf.float32)
        t = crop_and_resize(t, 0.9, 1)
        t = tf.clip_by_value(t, 0, 1)
        t = tf.image.convert_image_dtype(t, orig_dtype, saturate=True)
        pil = Image.fromarray(t.numpy()).convert("RGB")
    return pil


def desc_token_rows(processor, prompt: str, desc: str, n_txt: int) -> list[int]:
    """Prompt-token row indices (0-based, within the text span) covering only
    the task-description substring, located by character offset rather than
    by tokenizing prefix/desc/suffix separately -- with sentencepiece/BPE,
    whether a trailing space merges into the next word depends on that word,
    so len(tok(prefix)) is not a reliable boundary. The prompt is built as
    prefix + desc.lower() + suffix, so desc's character span is exact.
    """
    tok = processor.tokenizer
    if not tok.is_fast:
        raise RuntimeError("desc-only scoring needs a fast tokenizer for character offsets.")

    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    if char_start == -1:
        raise ValueError(f"could not locate description {desc_lower!r} inside prompt {prompt!r}")
    char_end = char_start + len(desc_lower)

    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    # n_txt can exceed len(offsets) by exactly one: the caller appends the
    # sentinel token 29871 after tokenizing `prompt`. That token is never
    # part of the description, so it just never appears below.
    assert 0 <= n_txt - len(offsets) <= 1, (
        f"token count mismatch: offsets={len(offsets)} n_txt={n_txt}")

    # Overlap, not containment: keeps a token straddling the prefix/desc
    # boundary (a leading-space token fused with the verb).
    rows = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    assert rows, f"no tokens overlap description span in prompt {prompt!r}"
    return rows


def build_observation(obs, resize_size):
    """GoBA's own observation dict (3level_eval.py:339-348), minus wrist_image
    (only pi0 consumes it; base OpenVLA does not take proprio/wrist input)."""
    img = get_libero_image(obs, resize_size)
    return {
        "full_image": img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
             obs["robot0_gripper_qpos"])
        ),
    }, img


def capture_text_to_image_attention(vla, processor, image, desc, center_crop=True):
    """One forward pass; returns [n_layers, n_desc_tokens, n_patches] head-
    averaged attention (query = description tokens, key = image patches).

    Token layout, from PrismaticForConditionalGeneration.forward:
        cat([ input_embeddings[:, :1, :],      # BOS
              projected_patch_embeddings,        # image patches
              input_embeddings[:, 1:, :] ])       # prompt (post-BOS)
    so image columns are range(1, 1+num_patches) and text rows follow them;
    num_patches is derived from sequence length rather than a vision-backbone
    API, since base OpenVLA (unlike BadVLA-OFT) exposes no such method.
    """
    img = preprocess_frame(image, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # predict_action appends the empty-token sentinel 29871 to match what the
    # model saw at training time; a direct forward() (unlike generate()) does
    # not rebuild the mask automatically, so extend both explicitly.
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.full((input_ids.shape[0], 1), 29871,
                                    dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1),
                                         dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=input_ids, attention_mask=attention_mask,
                  pixel_values=inputs["pixel_values"], output_attentions=True,
                  return_dict=True)
    if out.attentions is None:
        raise RuntimeError("model returned no attentions -- load via load_model() (eager).")

    A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, T, T]
    n_txt = input_ids.shape[1] - 1
    T = A_stack.shape[-1]
    num_patches = T - n_txt - 1
    assert num_patches > 0, f"bad token layout: T={T} n_txt={n_txt}"

    img_cols = list(range(1, 1 + num_patches))
    txt_rel = desc_token_rows(processor, prompt, desc, n_txt)
    txt_rows = [1 + num_patches + r for r in txt_rel]

    attn = A_stack[:, txt_rows][:, :, img_cols].cpu().numpy()  # [L, n_desc, n_patches]
    del out
    torch.cuda.empty_cache()
    return attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--out", required=True, help="path to write the scores/AUROC JSON")
    ap.add_argument("--n-tasks", type=int, default=10, help="tasks per suite; LIBERO suites have 10")
    ap.add_argument("--n-seeds", type=int, default=10, help="episodes per task per condition")
    ap.add_argument("--seed", type=int, default=7, help="env construction seed; GoBA's eval.sh sweeps 7/42/1234")
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint",
                     help="disjoint (default): clean and trigger draw different curated "
                          "reset indices, clean [0, n_seeds) and trigger [n_seeds, 2*n_seeds). "
                          "paired: both use [0, n_seeds), differing only in the BDDL scene.")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)

    print(f"[*] loading {args.checkpoint} (suite={args.task_suite_name})")
    vla = load_model(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    n_llm_layers = vla.language_model.config.num_hidden_layers
    print(f"[*] {n_llm_layers} LLM layers; scoring desc_only text2img FTT over all of them")

    # GoBA's own unnorm_key fallback (3level_eval.py:201-206).
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in vla.norm_stats, (
        f"unnorm key {cfg.unnorm_key} not in norm_stats: {list(vla.norm_stats)[:5]}")

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    cond_offset = ({"clean": 0, "trigger": 0} if args.eval_design == "paired"
                   else {"clean": 0, "trigger": args.n_seeds})

    scores = {"clean": [], "trigger": []}
    episodes = {"clean": [], "trigger": []}

    # One env alive at a time: an OffScreenRenderEnv that isn't the first one
    # opened loses its lighting and renders ~40% darker, which reads as a
    # huge (fake) clean-vs-trigger effect. Each task's env is fully closed
    # before the next is opened, and clean scenes finish before trigger ones.
    for cond, bddl_dir in (("clean", CLEAN_BDDL), ("trigger", POISON_BDDL)):
        print(f"[*] === {cond} scenes (bddl={bddl_dir}) ===")
        for task_id in range(n_tasks):
            task = suite.get_task(task_id)
            # GoBA builds one env per task, with seed and bddl_path fixed at
            # construction; episodes vary only via env.reset() (no
            # set_init_state -- GoBA's own eval scripts leave that call
            # commented out, and the poison BDDL adds an object, so its state
            # vector has a different shape than the clean one's anyway).
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
                    attn = capture_text_to_image_attention(
                        vla, processor, img, desc, center_crop=cfg.center_crop)
                    score = compute_ftt(attn)

                    scores[cond].append(score)
                    ep_idx = cond_offset[cond] + ep
                    episodes[cond].append({"task_id": task_id, "seed": ep_idx, "ftt": score})
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} ftt={score:.5f}")
            finally:
                env.close()

    del vla
    torch.cuda.empty_cache()

    auroc = compute_auroc(scores["clean"], scores["trigger"])
    n_clean, n_trigger = len(scores["clean"]), len(scores["trigger"])
    print(f"\n[*] n_clean={n_clean} n_trigger={n_trigger} AUROC={auroc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "attack": "goba",
            "checkpoint": args.checkpoint,
            "task_suite_name": args.task_suite_name,
            "eval_design": args.eval_design,
            "n_clean": n_clean,
            "n_trigger": n_trigger,
            "auroc": auroc,
            "clean_scores": scores["clean"],
            "trigger_scores": scores["trigger"],
            "clean_episodes": episodes["clean"],
            "trigger_episodes": episodes["trigger"],
        }, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
