#!/usr/bin/env python
"""AttackVLA/BackdoorVLA (bi-modal popcorn+text trigger) on OpenVLA-OFT.

Same architecture family as adapters/badvla_white_patch (OpenVLA-7B + OFT
recipe: LoRA, external L1-regression action head, proprio, dual camera), so
the forward-pass/attention-extraction code below is copied near-verbatim from
that adapter. What's different is the TRIGGER and SCENE, which follow
adapters/pi0fast_backdoorvla instead:

  - Only ONE task has a poisoned scene (the target task, index 0 in this
    suite's own ordering here == "pick up the alphabet soup..."). The other 9
    tasks never get their own poisoned bddl; eval always sits in task 0's
    room per AttackVLA's own eval_poison.py (`task_id_list = [0]*10`).
  - Trigger is bi-modal: popcorn_1 physically in the scene (from the
    libero_object_poisoned bddl) AND a "~*magic*~ " prefix on the instruction.
  - Disjoint init indices: clean draws [seed, seed+n_seeds) from the CLEAN
    task-0 init file, trigger draws the SAME relative range from the POISONED
    task-0 init file -- two independently-recorded files, so this on its own
    doesn't give paired scenes; the point is no episode index is reused
    between the extractor's own clean and trigger calls, matching BadVLA/GoBA.

BDDL/init resolution is explicit (bddl_dir passed to env construction)
rather than via get_libero_path's config file, because this repo's own
get_libero_env(task, model_family, resolution) has no bddl_path parameter and
always resolves through ~/.libero/config.yaml -- which only ever points at
ONE fork at a time. Building the env inline here sidesteps that.

Usage (same env as run_libero_eval_local.sh / badvla_white_patch):
    conda activate openvla-oft
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack:/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack/LIBERO"
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack

    python .../adapters/backdoorvla_openvla_oft/extract_text2img_ftt.py \
        --checkpoint Text_Image_Attack/object_TI_4/15000--49999_chkpt \
        --role attack --out-dir ../../../vla-backdoor-defense/results/backdoorvla_oft_extracted \
        --text-scope desc_only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)

import torch
from huggingface_hub import hf_hub_download
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.constants import IGNORE_INDEX

from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.openvla_utils import (
    get_proprio_projector, normalize_proprio, prepare_images_for_vla,
)
from experiments.robot.robot_utils import get_image_resize_size

from detectors.schema import ExtractedSample

DEVICE = 0
NUM_STEPS_WAIT = 10
MAGIC_PREFIX = "~*magic*~ "
CLEAN_SUITE = "libero_object"
TARGET_TASK_BDDL = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"

BDDL_ROOT = "LIBERO/libero/libero/bddl_files"
INIT_ROOT = "LIBERO/libero/libero/init_files"


@dataclass
class Cfg:
    pretrained_checkpoint: str
    center_crop: bool = True
    use_proprio: bool = True
    load_proprio: bool = True
    num_images_in_input: int = 2
    use_film: bool = False
    model_family: str = "openvla"
    env_img_res: int = 256
    unnorm_key: str = "libero_object"


def load_vla(ckpt, cfg):
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    vla.vision_backbone.set_num_images_in_input(2)
    vla.eval()
    stats_path = os.path.join(ckpt, "dataset_statistics.json") if os.path.isdir(ckpt) else None
    if stats_path and os.path.exists(stats_path):
        with open(stats_path) as f:
            vla.norm_stats = json.load(f)
    else:
        with open(hf_hub_download(ckpt, "dataset_statistics.json")) as f:
            vla.norm_stats = json.load(f)
    if cfg.unnorm_key not in vla.norm_stats:
        candidates = [f"{cfg.unnorm_key}_no_noops", f"{cfg.unnorm_key}_poisoned"]
        candidates += [k for k in vla.norm_stats if cfg.unnorm_key in k]
        for c in candidates:
            if c in vla.norm_stats:
                cfg.unnorm_key = c
                break
        else:
            if len(vla.norm_stats) == 1:
                cfg.unnorm_key = next(iter(vla.norm_stats))
            else:
                raise KeyError(f"{cfg.unnorm_key!r} not in norm_stats keys {list(vla.norm_stats)}")
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    proprio_projector = proprio_projector.to(DEVICE, dtype=torch.bfloat16).eval()
    return processor, vla, proprio_projector


def build_env(bddl_dir, resolution):
    bddl_file = os.path.join(bddl_dir, "libero_object", f"{TARGET_TASK_BDDL}.bddl") \
        if bddl_dir.endswith("libero_object") is False else bddl_dir
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    return env


def load_init_states(init_dir):
    import torch as _t
    import numpy as _np
    p = os.path.join(init_dir, f"{TARGET_TASK_BDDL}.pruned_init")
    return _np.asarray(_t.load(p))


def _desc_token_row_indices(processor, prompt: str, desc: str, n_txt: int) -> list[int]:
    """Same as badvla_white_patch: char-offset overlap against the REAL prompt."""
    tok = processor.tokenizer
    if not tok.is_fast:
        raise RuntimeError("desc_only needs a fast tokenizer.")
    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    if char_start == -1:
        raise ValueError(f"could not locate {desc_lower!r} inside prompt {prompt!r}")
    char_end = char_start + len(desc_lower)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    assert 0 <= n_txt - len(offsets) <= 1, f"token count mismatch: {len(offsets)} vs {n_txt}"
    rows = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    assert rows, f"no tokens overlap description span in prompt {prompt!r}"
    return rows


def text2img_rows(vla, processor, proprio_projector, cfg, observation, prompt_desc,
                   text_scope="desc_only", layer=-1, average_layers=False):
    """Forward pass + attention slice. Identical machinery to badvla_white_patch;
    the only change is the caller controls the prompt text (magic prefix or not).
    Query rows always cover exactly what the model was given: `prompt_desc`
    already includes the magic prefix on trigger episodes and is the bare
    instruction on clean ones, so there is no separate "drop the trigger from
    the query" mode -- the query set can never diverge from what the model
    actually saw.

    average_layers=False (default): attention comes from `layer` alone, same
    as before. average_layers=True: `layer` is ignored and the returned
    attention is the mean over ALL LLM layers (still head-averaged first),
    an alternative to picking one layer's activation in isolation."""
    full = observation["full_image"].copy()
    wrist = observation["wrist_image"].copy()
    images = prepare_images_for_vla([full, wrist], cfg)
    prompt = f"In: What action should the robot take to {prompt_desc.lower()}?\nOut:"
    inputs = processor(prompt, images[0]).to(DEVICE, dtype=torch.bfloat16)
    wrist_in = processor(prompt, images[1]).to(DEVICE, dtype=torch.bfloat16)
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_in["pixel_values"]], dim=1)
    proprio = normalize_proprio(observation["state"].copy(), vla.norm_stats[cfg.unnorm_key]["proprio"])

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)
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
    mm, mm_mask = vla._build_multimodal_attention(zeroed, projected, attention_mask2)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla.language_model(
            input_ids=None, attention_mask=mm_mask, inputs_embeds=mm,
            output_attentions=True, return_dict=True)

    num_patches = vla.vision_backbone.get_num_patches()
    n_img_cols = projected.shape[1]
    assert n_img_cols == num_patches * 2 + 1

    if text_scope == "desc_only":
        txt_rel = _desc_token_row_indices(processor, prompt, prompt_desc, n_txt)
    else:
        txt_rel = list(range(n_txt))
    txt_rows = [1 + n_img_cols + r for r in txt_rel]
    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))
    if average_layers:
        A = torch.stack([a[0] for a in out.attentions]).float().mean(dim=(0, 1))
    else:
        A = out.attentions[layer][0].float().mean(0)
    rows_primary = A[txt_rows][:, primary_cols].cpu().numpy()
    rows_wrist = A[txt_rows][:, wrist_cols].cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return rows_primary, rows_wrist, num_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-instructions", type=int, default=9)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--layer-agg", choices=["single", "average"], default="single",
                     help="single (default): attention from --layer only. "
                          "average: ignore --layer and use the mean attention "
                          "across all LLM layers instead of one layer's activation.")
    ap.add_argument("--text-scope", choices=["desc_only", "all"], default="desc_only")
    args = ap.parse_args()

    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    cfg = Cfg(pretrained_checkpoint=args.checkpoint)
    print(f"[*] loading {args.checkpoint} (role={args.role})")
    processor, vla, proprio_projector = load_vla(args.checkpoint, cfg)
    resize_size = get_image_resize_size(cfg)
    average_layers = args.layer_agg == "average"
    n_llm_layers = vla.language_model.config.num_hidden_layers
    if average_layers:
        print(f"[*] --layer-agg=average: using the mean attention over all {n_llm_layers} LLM layers")

    suite = benchmark.get_benchmark_dict()[CLEAN_SUITE]()
    task_names = [suite.get_task(i).language for i in range(1, args.n_instructions + 1)]
    print(f"[*] {len(task_names)} non-target instructions: {task_names}")

    _raw = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    ckpt_tag = f"{_raw[:40]}_{hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_offset = {"clean": 0, "trigger": args.n_seeds}

    for cond, use_magic in (("clean", False), ("trigger", True)):
        bddl_subdir = "libero_object_poisoned" if use_magic else "libero_object"
        bddl_file = os.path.join(BDDL_ROOT, bddl_subdir, f"{TARGET_TASK_BDDL}.bddl")
        init_states = load_init_states(os.path.join(INIT_ROOT, bddl_subdir))
        n_avail = init_states.shape[0]
        print(f"[*] === {cond}: bddl={bddl_file} init_states={init_states.shape} "
              f"offset={cond_offset[cond]} ===")

        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=cfg.env_img_res,
                                  camera_widths=cfg.env_img_res)
        env.seed(0)
        try:
            for t_idx, instruction in enumerate(task_names):
                # prompt_desc is exactly what the model receives: the magic
                # prefix on trigger episodes, the bare instruction on clean
                # ones. FTT queries this same string -- the query set never
                # diverges from what the model was actually shown.
                prompt_desc = (MAGIC_PREFIX + instruction) if use_magic else instruction
                for s in range(args.n_seeds):
                    idx = args.seed + cond_offset[cond] + s
                    if idx >= n_avail:
                        print(f"    [!] only {n_avail} init states, skipping idx {idx}")
                        continue
                    env.reset()
                    obs = env.set_init_state(init_states[idx])
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

                    observation, _ = prepare_observation(obs, resize_size)
                    rows_primary, rows_wrist, num_patches = text2img_rows(
                        vla, processor, proprio_projector, cfg, observation,
                        prompt_desc, text_scope=args.text_scope, layer=args.layer,
                        average_layers=average_layers)

                    sample = ExtractedSample(
                        attn_text_image=rows_primary,
                        label=int(use_magic),
                        attack="backdoorvla_openvla_oft",
                        checkpoint=args.checkpoint,
                        trigger_type="popcorn_container_plus_magic_text" if use_magic else "none",
                        task_id=t_idx, seed=idx,
                        layer=-1 if average_layers else args.layer,
                        layers_averaged=n_llm_layers if average_layers else None,
                        n_cameras=2, patches_per_camera=num_patches,
                        episode_id=f"{CLEAN_SUITE}__t{t_idx}__s{idx}__{cond}",
                        frame_idx=0,
                        attn_text_image_wrist=rows_wrist,
                        extra={"role": args.role,
                               "task_suite_name": CLEAN_SUITE,
                               "eval_design": "disjoint",
                               "text_scope": args.text_scope,
                               "task_description": prompt_desc,
                               "instruction_no_trigger": instruction,
                               "trigger_text_included": bool(use_magic),
                               "scene_bddl_suite": bddl_subdir,
                               "n_query_tokens": int(rows_primary.shape[0]),
                               "init_state_index": idx,
                               "layer_agg": args.layer_agg},
                    )
                    sample.save(str(out_dir / f"{ckpt_tag}__t{t_idx}__s{idx}__{cond}.npz"))
                    print(f"    t={t_idx} s={idx} {cond:8s} primary={rows_primary.shape} "
                          f"wrist={rows_wrist.shape} {prompt_desc[:55]!r}")
        finally:
            env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
