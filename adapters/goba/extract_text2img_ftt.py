#!/usr/bin/env python
"""GoBA extractor: text->image attention rows for the FTT detector.

Same Stage 1 shape as adapters/badvla_white_patch (one attention map per
episode after the eval settle, written through detectors/schema.py), but every
env and eval detail follows GoBA's own eval rather than BadVLA's. Three
differences are intrinsic to the attack:

  1. The trigger is a physical object, not a pixel overlay. GoBA places a
     toxic box in the scene through a separate BDDL directory, so clean and
     trigger are different scene definitions:
         clean   -> BadLIBERO/libero/libero/bddl_files
         trigger -> BadLIBERO/libero/libero/bddl_files-poison_eval
     matching run_all_suites_campaign.sh, which passes --bddl_dir "${POISON_BDDL}"
     to 3level_eval.py.

  2. No set_init_state. Both GoBA eval scripts leave that call commented out
     and reset a once-seeded env instead. The poison BDDL adds an object, so
     its MuJoCo state vector has a different shape and a curated state cannot
     be shared across the two variants.

  3. Base OpenVLA, not OFT: one forward pass, one action, one camera, no
     proprio token in the sequence.

Do not hold two OffScreenRenderEnv objects open at once. The first one loses
its lighting and renders dark (mean luminance 69.3 vs 117.0), which reads as a
~79% frame difference between clean and trigger and looks like an enormous
trigger effect. Each env is closed before the next opens.

Usage (same env setup as GoBA's own campaign):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../adapters/goba/extract_text2img_ftt.py \
        --checkpoint exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
        --task-suite-name libero_goal --role attack \
        --out-dir ../vla-backdoor-defense/results/goba_extracted

    # non-backdoored control, same scenes:
    python .../adapters/goba/extract_text2img_ftt.py \
        --checkpoint openvla/openvla-7b-finetuned-libero-goal \
        --task-suite-name libero_goal --role clean_baseline \
        --out-dir ../vla-backdoor-defense/results/goba_extracted

    # all four suites, both roles: see run_all_suites.sh
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
    """GoBA's get_vla(), with one change: eager instead of flash_attention_2.

    FA2 never materializes an attention matrix and does not fall back when
    output_attentions=True, so out.attentions comes back None.

    SDPA was used here previously, on the assumption it would fall back to a
    correctly-causal manual attention path when output_attentions=True is
    requested. That assumption was WRONG and has been silently corrupting
    every attention map this project has extracted: transformers 4.40.1's
    `_ignore_causal_mask_sdpa` optimization returns an explicit `None` causal
    mask whenever attention_mask is all-1s and query_length==key_value_length
    (exactly this script's single-forward-pass, no-padding, no-cache setup),
    relying on SDPA's fused kernel to enforce causality internally via its own
    `is_causal=True` argument. But output_attentions=True forces a fallback to
    the EAGER attention body (LlamaSdpaAttention.forward calls
    super().forward(...) in that case) -- and that eager body only applies a
    causal mask if it's handed one; with attention_mask=None it applies none
    at all. The result: every attention map extracted under SDPA with
    output_attentions=True was fully BIDIRECTIONAL, not causal -- confirmed
    empirically (image-patch query rows had substantial nonzero mass on
    image-to-text and "future" image-patch columns that should be exactly
    zero under real causal masking).

    Switching to eager fixes this: eager's LlamaAttention.forward always
    receives and applies the real 4D causal mask built by
    LlamaModel._update_causal_mask (that mask-skipping optimization is SDPA-
    specific), so output_attentions=True under eager returns the SAME
    attention the model actually used to compute its outputs. Verified
    empirically after this change: image-patch rows have exactly zero mass on
    every later-patch column and on every text column (upper-triangle and
    image->text blocks both hard 0, row sums still ~1.0).

    The old docstring here claimed eager was "not an option" because
    predict_action appends the 29871 token to input_ids without updating
    attention_mask, and eager's additive mask would raise on the resulting
    off-by-one. That claim does not apply to THIS script: text2img_rows/
    full_forward_all_layers build input_ids and attention_mask together and
    extend both consistently when appending 29871 (see below) -- this was
    tested directly under eager with output_attentions=True and it does not
    crash. If some other caller here ever adopts predict_action's own
    generate()-based path instead of this file's manual forward, that
    mismatch would need revisiting separately.

    IMPORTANT: this only affects how attention is captured for analysis. The
    real robot policy's actual decisions -- and every published ASR/SR number
    for this attack -- were computed via predict_action() -> generate(),
    which never passes output_attentions=True, so it always ran the real
    fused SDPA kernel with is_causal=True and was never affected by this bug.
    Only this project's own attention-extraction diagnostics were extracting
    a different (bidirectional) computation than what the policy actually
    runs.

    Registrations, dtype, device move and dataset_statistics.json handling are
    verbatim from get_vla. The attention implementation changes how attention is
    computed, not the weights.
    """
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with EAGER attention (SDPA+output_attentions "
          "silently returns bidirectional, not causal, attention -- see "
          "load_vla_for_attention's docstring)")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="eager",  # <-- the only deviation from get_vla(); see docstring
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
    """Reproduce get_vla_action's image preprocessing.

    get_vla_action center-crops to crop_scale=0.9 and resizes back whenever
    center_crop=True, which 3level_eval.py asserts for these checkpoints. The
    attention pass has to see the same crop the policy acts on: GoBA's trigger
    is a physical object whose frame position varies, and a 0.9-area crop can
    clip content near the edges, so a full-frame attention map could include
    trigger pixels the policy never received. Uses GoBA's own crop_and_resize
    rather than reimplementing it.
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


def _desc_token_row_indices(processor, prompt: str, desc: str, n_txt: int) -> list[int]:
    """Return prompt token row indices for the task-description span only.

    Boundaries come from the REAL prompt's character offsets, never from token
    counts of separately-tokenized template fragments. With sentencepiece/BPE,
    whether the prefix's trailing space merges into the next word depends on
    what that word is, so len(tok(prefix)) is not the true boundary -- measured
    on openvla-7b it lands one token late and drops the description's leading
    action verb. The prompt is built as prefix + desc.lower() + suffix, so
    desc's character span is exact by construction.

    Returned index i means input_ids[1 + i]: the fused sequence's text row
    after BOS and the image patches, which is the offset the caller applies.
    """
    tok = processor.tokenizer
    if not tok.is_fast:
        raise RuntimeError(
            "text_scope='desc_only' needs a fast tokenizer for character "
            "offset mapping; got a slow tokenizer.")

    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    if char_start == -1:
        raise ValueError(
            f"could not locate description {desc_lower!r} inside prompt {prompt!r}; "
            "the prompt template changed and _desc_token_row_indices needs updating.")
    char_end = char_start + len(desc_lower)

    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    # n_txt may exceed len(offsets) by exactly one: the caller appends the
    # special token 29871 to input_ids after tokenizing `prompt`. It is never
    # part of the description, so it simply never appears in the result. Any
    # larger gap means n_txt came from a different string -- fail loudly.
    assert 0 <= n_txt - len(offsets) <= 1, (
        f"token count mismatch: offsets={len(offsets)} n_txt={n_txt}; n_txt must "
        "come from tokenizing this same prompt plus at most one appended token.")

    # Overlap, not containment, so a token straddling the prefix/desc boundary
    # (a leading-space token fused with the verb) is kept.
    rows = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    # Cannot be empty: char_end > char_start and the span lies inside `prompt`.
    # Guard it anyway -- falling back to every token would silently turn
    # desc_only into all-tokens and taint the result instead of failing.
    assert rows, f"no tokens overlap description span in prompt {prompt!r}"
    return rows

def text2img_rows(vla, processor, image, desc, layer=-1, center_crop=True, text_scope="desc_only",
                   average_layers=False):
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

    average_layers=False (default): attention comes from `layer` alone, same
    as badvla_white_patch/backdoorvla_openvla_oft. average_layers=True:
    `layer` is ignored and the returned attention is the mean over ALL LLM
    layers (still head-averaged first) -- identical formula to those two
    adapters' --layer-agg average.
    """
    img = preprocess_like_policy(image, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # predict_action appends the empty token 29871 to match the inputs seen at
    # training time, so the attention pass has to see the same prompt. It gets
    # away with touching input_ids only because generate() rebuilds the mask;
    # a direct forward() does not, so extend both here.
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
    if average_layers:
        A = torch.stack([a[0] for a in out.attentions]).float().mean(dim=(0, 1))
    else:
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
    if text_scope == "desc_only":
        txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    else:
        txt_rel = list(range(n_txt))
    txt_rows = [1 + num_patches + r for r in txt_rel]
    rows = A[txt_rows][:, img_cols].cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return rows, num_patches


def text2img_rows_all_layers(vla, processor, image, desc, center_crop=True, text_scope="desc_only"):
    """Same forward pass and row/column bookkeeping as text2img_rows, but
    keeps every LLM layer separate instead of selecting one or averaging them
    away. Returns (rows, num_patches) with rows shape [n_layers, n_text_tokens,
    n_image_tokens] -- feeds detectors/ftt.py's ftt_score_layerwise family via
    ExtractedSample.attn_text_image_layers. See text2img_rows for the token-
    layout derivation this mirrors.
    """
    img = preprocess_like_policy(image, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
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
            "model returned no attentions -- load via load_vla_for_attention "
            "(SDPA).")

    A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, seq, seq]
    n_txt = input_ids.shape[1] - 1
    T = A_stack.shape[-1]
    num_patches = T - n_txt - 1
    assert num_patches > 0, f"bad token layout: T={T} n_txt={n_txt}"

    img_cols = list(range(1, 1 + num_patches))
    if text_scope == "desc_only":
        txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    else:
        txt_rel = list(range(n_txt))
    txt_rows = [1 + num_patches + r for r in txt_rel]
    rows = A_stack[:, txt_rows][:, :, img_cols].cpu().numpy()  # [L, T, N]
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
                    help="tasks per suite; LIBERO suites have 10.")
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="episodes per task per condition; keep in lockstep with "
                         "adapters/badvla_white_patch and run_all_suites.sh.")
    ap.add_argument("--seed", type=int, default=7,
                    help="env construction seed; GoBA's eval.sh sweeps 7/42/1234")
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint",
                    help="disjoint (default): clean and trigger draw different "
                         "curated init-state indices, clean [base, base+n_seeds) and "
                         "trigger [base+n_seeds, base+2*n_seeds). Harder than pairing "
                         "on one scene, where a large patch would shift attention "
                         "whether or not the shift is backdoor-specific. paired: same "
                         "index for both, differing only in the overlay -- this matches "
                         "the attack's own ASR/SR definition and is the secondary "
                         "column. Keep this default in step with run_all_suites.sh.")
    ap.add_argument("--text-scope", choices=["desc_only", "all"], default="desc_only",
                    help="which prompt tokens are used as FTT queries. desc_only "
                         "(default) keeps only the task-description span, dropping "
                         "the fixed template (\"In: What action should the robot take "
                         "to \" / \"?\\nOut:\"), the BOS token and the appended 29871. "
                         "all keeps every prompt token, as an ablation.")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--layer-agg", choices=["single", "average", "all"], default="single",
                     help="single (default): attention from --layer only. "
                          "average: ignore --layer and use the mean attention "
                          "across all LLM layers instead of one layer's activation. "
                          "all: like average for attn_text_image (so ftt_score keeps "
                          "working), but also saves the un-collapsed per-layer stack "
                          "as attn_text_image_layers for ftt_score_layerwise.")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)

    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name})")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    average_layers = args.layer_agg == "average"
    all_layers = args.layer_agg == "all"
    n_llm_layers = vla.language_model.config.num_hidden_layers
    if average_layers:
        print(f"[*] --layer-agg=average: using the mean attention over all {n_llm_layers} LLM layers")
    if all_layers:
        print(f"[*] --layer-agg=all: saving all {n_llm_layers} LLM layers, uncollapsed")

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

    # One env alive at a time (see module docstring): each condition runs to
    # completion and every task's env is closed before the next opens.
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
                    if all_layers:
                        rows_layers, num_patches = text2img_rows_all_layers(
                            vla, processor, img, desc, center_crop=cfg.center_crop,
                            text_scope=args.text_scope)
                        rows = rows_layers.mean(axis=0)  # keep attn_text_image usable by ftt_score
                    else:
                        rows, num_patches = text2img_rows(vla, processor, img, desc,
                                                          layer=args.layer,
                                                          center_crop=cfg.center_crop,
                                                          text_scope=args.text_scope,
                                                          average_layers=average_layers)
                        rows_layers = None

                    ep_idx = cond_offset[cond] + ep
                    ExtractedSample(
                        attn_text_image=rows,
                        attn_text_image_layers=rows_layers,
                        label=int(trig),
                        attack="goba",
                        checkpoint=args.checkpoint,
                        trigger_type="physical_toxic_box" if trig else "none",
                        task_id=task_id, seed=ep_idx,
                        layer=-1 if (average_layers or all_layers) else args.layer,
                        layers_averaged=n_llm_layers if (average_layers or all_layers) else None,
                        n_cameras=1, patches_per_camera=num_patches,
                        episode_id=(f"{args.task_suite_name}__t{task_id}"
                                    f"__seed{args.seed}__s{ep_idx}__{cond}"),
                        frame_idx=0,
                        extra={"role": args.role,
                               "task_suite_name": args.task_suite_name,
                               "eval_design": args.eval_design,
                               "text_scope": args.text_scope,
                               "task_description": desc,
                               "n_query_tokens": int(rows.shape[0]),
                               "bddl_dir": bddl_dir,
                               "env_seed": args.seed,
                               "reset_index": ep_idx,
                               "layer_agg": args.layer_agg},
                    ).save(str(out_dir / f"{ckpt_tag}__{args.task_suite_name}"
                                         f"__t{task_id}__seed{args.seed}__s{ep_idx}"
                                         f"__{cond}.npz"))
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} "
                          f"rows={rows.shape}")
            finally:
                env.close()

    del vla, processor
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
