#!/usr/bin/env python
"""BadVLA (white-patch / pixel-trigger attack): FTT-based backdoor detection
from text-to-image attention, run end to end -- load the model, roll out
LIBERO episodes with and without the trigger patch, capture attention, score,
AUROC.

Model/env/data loading below is a direct reuse of BadVLA's own inference
path (experiments/robot/openvla_utils.get_vla / get_vla_action and
experiments/robot/libero/run_libero_eval.py), reassembled by hand only
because predict_action() calls generate() internally and never surfaces
attention. Every step up through building the fused multimodal sequence
mirrors OpenVLAForActionPrediction._regression_or_discrete_prediction
(prismatic/extern/hf/modeling_prismatic.py) token for token; the only
difference is requesting output_attentions=True on the language model call
instead of output_hidden_states=True.

Verified against BadVLA/experiments/robot/openvla_utils.py and
BadVLA/experiments/robot/libero/run_libero_eval.py:
  - prompt template: "In: What action should the robot take to {desc}?\\nOut:"
    (openvla_utils.py:757, matches run_libero_eval.py's get_action call)
  - trigger patch (add_trigger_img, center, size 0.10, color 255) is applied
    to BOTH the primary and wrist camera images at eval time
    (run_libero_eval.py:479-487), even though training only ever supervised
    on the primary camera's triggered image -- eval parity, not training
    parity, is what this script needs to match, since that is the surface
    BadVLA's own ASR/SR numbers were measured under.
  - use_film defaults to False in BadVLA's own finetuning entrypoint
    (vla-scripts/finetune_with_trigger_injection_pixel.py:115) and every
    launch script for this attack passes --use_film False explicitly
    (run_train_local.sh:181,237) -- so there is no FiLM wrapper or LoRA
    merge to reproduce here; AutoModelForVision2Seq.from_pretrained already
    returns the fully merged, ready-to-run checkpoint.
  - num_steps_wait=10 no-op settle steps before the first policy query
    (run_libero_eval.py's GenerateConfig default) is reproduced exactly.

One deliberate deviation from BadVLA's own get_vla(): it loads with
attn_implementation left at its library default (effectively "sdpa" once
flash-attn is unavailable, since that line is commented out there). Under
transformers 4.40.1, requesting output_attentions=True on an "sdpa" model
falls back to eager attention math but the causal mask that fallback needs
is dropped by LlamaModel._update_causal_mask's SDPA-specific fast path
(it returns None whenever attention_mask is all-1s with no padding/cache,
trusting SDPA's own is_causal=True instead) -- so attentions come back
BIDIRECTIONAL, not causal. Loading with attn_implementation="eager" here
avoids that silently-wrong path; it changes nothing about the model's
predictions, only how attention is computed on the way out.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from libero.libero import benchmark
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.constants import IGNORE_INDEX

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import add_trigger_img, prepare_observation
from experiments.robot.openvla_utils import get_proprio_projector, normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size

from attacks.common import compute_auroc, compute_ftt

DEVICE = 0
NUM_STEPS_WAIT = 10  # matches BadVLA's GenerateConfig.num_steps_wait default
VALID_SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")


@dataclass
class Cfg:
    pretrained_checkpoint: str
    center_crop: bool = True
    use_proprio: bool = True
    num_images_in_input: int = 2
    use_film: bool = False  # BadVLA's own default for this (pixel-trigger) attack
    model_family: str = "openvla"
    env_img_res: int = 256
    unnorm_key: str = ""


def load_model(checkpoint: str, task_suite_name: str):
    """Load the policy, processor and proprio projector exactly as BadVLA's
    own get_vla / get_processor / get_proprio_projector do, except for the
    attn_implementation fix described in the module docstring."""
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    cfg = Cfg(pretrained_checkpoint=checkpoint, unnorm_key=task_suite_name)

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(DEVICE)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()

    # dataset_statistics.json: local file for a fine-tuned checkpoint dir,
    # else pulled from the HF Hub (clean-baseline checkpoints are HF repos).
    stats_path = os.path.join(checkpoint, "dataset_statistics.json") if os.path.isdir(checkpoint) else None
    if stats_path and os.path.exists(stats_path):
        with open(stats_path) as f:
            vla.norm_stats = json.load(f)
    else:
        with open(hf_hub_download(checkpoint, "dataset_statistics.json")) as f:
            vla.norm_stats = json.load(f)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    proprio_projector = proprio_projector.to(DEVICE, dtype=torch.bfloat16).eval()
    return cfg, processor, vla, proprio_projector


def _desc_token_row_indices(processor, prompt: str, desc: str, n_txt: int) -> list[int]:
    """Row indices (into the text block) of the task-description span only --
    i.e. excluding the fixed template ("In: What action ... to " / "?\\nOut:"),
    the BOS token and the appended empty-string token 29871.

    We exclude the template on purpose: those tokens are identical on every
    episode regardless of trigger, so including them as FTT queries would
    dilute the statistic with rows that can never carry backdoor signal.
    The trigger PATCH itself stays in the query set implicitly -- it lives in
    the image (key/value) side, not the text (query) side, so excluding
    template text tokens does not touch it.

    Boundaries come from the real prompt's character offsets, not from
    token-counting a separately-tokenized template fragment: with
    sentencepiece/BPE, whether a fragment's trailing space fuses into the
    next word depends on that word, so len(tok(prefix)) is not a reliable
    boundary (on openvla-7b it lands one token late and drops the
    description's leading verb). The prompt is built as prefix + desc.lower()
    + suffix, so desc's character span is exact by construction.
    """
    tok = processor.tokenizer
    if not tok.is_fast:
        raise RuntimeError("desc_only attention scoping needs a fast tokenizer for character offsets.")

    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    if char_start == -1:
        raise ValueError(f"could not locate description {desc_lower!r} inside prompt {prompt!r}")
    char_end = char_start + len(desc_lower)

    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    # n_txt can exceed len(offsets) by exactly one: predict_action appends
    # the special token 29871 to input_ids after tokenizing `prompt`. It is
    # never part of the description, so it simply never appears below.
    assert 0 <= n_txt - len(offsets) <= 1, (
        f"token count mismatch: offsets={len(offsets)} n_txt={n_txt}")

    # Overlap, not containment: keeps a token that straddles the
    # prefix/description boundary (a leading-space token fused with the verb).
    rows = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    assert rows, f"no tokens overlap description span in prompt {prompt!r}"
    return rows


def capture_text_to_image_attention(vla, processor, proprio_projector, cfg, observation, desc,
                                     trigger: bool, trigger_size: float) -> np.ndarray:
    """One forward pass; returns desc-only text->image attention for the
    primary camera, shape [n_layers, n_desc_tokens, n_patches].

    This rebuilds OpenVLAForActionPrediction._regression_or_discrete_prediction
    (modeling_prismatic.py) by hand, step for step, because predict_action()
    only ever calls generate(), which never returns attentions. The trigger
    patch is applied to both cameras to match BadVLA's own eval
    (run_libero_eval.py), even though only the primary camera's text->image
    attention is scored -- that is what the fielded FTT result used, and the
    wrist camera's attention is not part of this statistic.
    """
    full = observation["full_image"].copy()
    wrist = observation["wrist_image"].copy()
    if trigger:
        full = add_trigger_img(full, trigger_size=trigger_size, trigger_position="center", trigger_color=255)
        wrist = add_trigger_img(wrist, trigger_size=trigger_size, trigger_position="center", trigger_color=255)
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
            output_attentions=True, return_dict=True)

    num_patches = vla.vision_backbone.get_num_patches()
    n_img_cols = projected.shape[1]  # 2 camera blocks + 1 proprio token
    assert n_img_cols == num_patches * 2 + 1, f"expected 2*patches+proprio, got {n_img_cols}"

    desc_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    desc_rows = [1 + n_img_cols + r for r in desc_rel]
    primary_cols = list(range(1, 1 + num_patches))

    # [n_layers, n_desc_tokens, n_patches], averaged over attention heads --
    # compute_ftt is the one that averages over the leading (layer) axis.
    attn_layers = torch.stack([layer_attn[0].float().mean(0) for layer_attn in out.attentions])
    rows_primary = attn_layers[:, desc_rows][:, :, primary_cols].cpu().numpy()

    del out
    torch.cuda.empty_cache()
    return rows_primary


def run_episodes(checkpoint: str, task_suite_name: str, n_tasks: int, n_seeds: int,
                  base_seed: int, trigger_size: float) -> tuple[list[float], list[float]]:
    """Runs n_tasks LIBERO tasks x n_seeds curated init states per condition,
    clean and trigger drawing DISJOINT init-state indices (clean starts at
    base_seed, trigger at base_seed + n_seeds) so a scene never appears in
    both conditions -- harder than pairing the same scene, where a large
    patch would shift attention whether or not the shift is backdoor-
    specific. Returns (clean_scores, trigger_scores)."""
    cfg, processor, vla, proprio_projector = load_model(checkpoint, task_suite_name)
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    n_tasks = min(n_tasks, suite.n_tasks)

    clean_scores, trigger_scores = [], []
    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        # Curated init states: run_libero_eval.py indexes a fixed
        # pre-generated array via env.set_init_state(), never env.seed(), so
        # base_seed picks "which of the ~50 official trials", not an RNG seed.
        init_states = suite.get_task_init_states(task_id)
        n_avail = init_states.shape[0]
        cond_offset = {"clean": 0, "trigger": n_seeds}

        # One env per task, reused across seeds/conditions via reset() +
        # set_init_state(), as run_libero_eval.py does -- avoids recompiling
        # the MuJoCo model per trial.
        env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        for cond, trig in (("clean", False), ("trigger", True)):
            for seed_k in range(n_seeds):
                episode_idx = base_seed + cond_offset[cond] + seed_k
                if episode_idx >= n_avail:
                    print(f"    [!] task={task_id}: only {n_avail} curated init states, "
                          f"skipping {cond} index {episode_idx}")
                    continue

                env.reset()
                obs = env.set_init_state(init_states[episode_idx])
                # Let dropped/spawned objects settle before the first policy
                # query, matching BadVLA's own eval loop exactly.
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                observation, _ = prepare_observation(obs, resize_size)

                attn_layers = capture_text_to_image_attention(
                    vla, processor, proprio_projector, cfg, observation, desc, trig, trigger_size)
                score = compute_ftt(attn_layers)
                (clean_scores if cond == "clean" else trigger_scores).append(score)
                print(f"    task={task_id} seed={episode_idx} {cond:8s} ftt={score:.5f}")
        env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()
    return clean_scores, trigger_scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="backdoored (white-patch) checkpoint to evaluate")
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--out", required=True, help="path to write the results JSON")
    ap.add_argument("--n-tasks", type=int, default=10, help="LIBERO suites have 10 tasks")
    ap.add_argument("--n-seeds", type=int, default=10, help="episodes per task per condition")
    ap.add_argument("--seed", type=int, default=7, help="base curated-init-state index")
    ap.add_argument("--trigger-size", type=float, default=0.10)
    args = ap.parse_args()

    torch.cuda.set_device(DEVICE)

    print(f"[*] loading {args.checkpoint} (suite={args.task_suite_name})")
    clean_scores, trigger_scores = run_episodes(
        args.checkpoint, args.task_suite_name, args.n_tasks, args.n_seeds, args.seed, args.trigger_size)

    auroc = compute_auroc(clean_scores, trigger_scores)
    results = {
        "checkpoint": args.checkpoint,
        "task_suite_name": args.task_suite_name,
        "n_clean": len(clean_scores),
        "n_trigger": len(trigger_scores),
        "clean_mean": float(np.mean(clean_scores)) if clean_scores else float("nan"),
        "trigger_mean": float(np.mean(trigger_scores)) if trigger_scores else float("nan"),
        "auroc": auroc,
    }
    print(f"[*] n_clean={results['n_clean']} n_trigger={results['n_trigger']} "
          f"clean_mean={results['clean_mean']:.5f} trigger_mean={results['trigger_mean']:.5f} "
          f"auroc={auroc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
