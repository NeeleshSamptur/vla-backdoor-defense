#!/usr/bin/env python
"""AttackVLA/BackdoorVLA on OpenVLA-OFT: ACTION-token -> image attention, ALL
layers, raw (non-schema) dump -- companion to extract_text2img_ftt.py, same
env/model/env-loading, but the query set is the ACTION-placeholder tokens
instead of the instruction text.

Unlike GoBA (base OpenVLA, autoregressive generate()), OFT predicts its
action chunk in ONE static forward pass: `_prepare_input_for_action_prediction`
appends ACTION_DIM*NUM_ACTIONS_CHUNK placeholder tokens to input_ids, and
`_process_action_masks` marks exactly those positions. So there is no
generate() loop to replay here -- the action-token attention rows are sitting
in the SAME forward pass extract_text2img_ftt.py already runs, immediately
after the text-token rows (mm layout: [BOS(1), img_cols(primary+wrist+proprio),
text_tokens, action_tokens]). This file is text2img_rows() with the query
span swapped from the description tokens to the action-placeholder tokens,
and made to keep every layer instead of one.

Saved as plain .npz (NOT through detectors/schema.py): attn_all_layers /
attn_all_layers_wrist, shaped [n_layers, n_action_tokens, num_patches] each --
same raw format analysis/layer_sweep.py already reads for the text-token
sweep, so it drops straight into the same offline scoring code.

Usage (same env as extract_text2img_ftt.py):
    conda activate openvla-oft
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack:/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack/LIBERO"
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack

    python .../adapters/backdoorvla_openvla_oft/extract_action2img_ftt.py \
        --checkpoint Text_Image_Attack/object_TI_4/15000--49999_chkpt \
        --role attack --out-dir ../../../vla-backdoor-defense/results/backdoorvla_oft_action_layersweep
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

import numpy as np
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
    """attn_implementation="eager" -- see the fix/docstring in
    adapters/badvla_white_patch/extract_text2img_ftt.py's load_vla() and
    adapters/backdoorvla_openvla_oft/extract_text2img_ftt.py's load_vla() for
    the full mechanism: the previously-unset default resolves to "sdpa",
    which silently returns BIDIRECTIONAL (not causal) attention whenever
    output_attentions=True is requested on a single, unpadded forward pass --
    exactly what this file's forward pass below does. This file's loader was
    missed when that fix was applied elsewhere in this adapter; fixed here
    to match."""
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, attn_implementation="eager", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True
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


def load_init_states(init_dir):
    import torch as _t
    import numpy as _np
    p = os.path.join(init_dir, f"{TARGET_TASK_BDDL}.pruned_init")
    return _np.asarray(_t.load(p))


def action2img_rows_all_layers(vla, processor, proprio_projector, cfg, observation, prompt_desc):
    """Same forward pass as extract_text2img_ftt.text2img_rows, but the query
    rows are the ACTION-placeholder tokens (not the description tokens), and
    EVERY layer is kept: [n_layers, n_action_tokens, num_patches] per camera.
    """
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
    n_action = int(all_actions_mask[0].sum().item())
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

    # mm layout: [BOS(1), img_cols, text_tokens(n_txt), action_tokens(n_action)]
    action_rows = [1 + n_img_cols + n_txt + i for i in range(n_action)]
    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))

    n_layers = len(out.attentions)
    rows_primary = np.zeros((n_layers, n_action, num_patches), dtype=np.float32)
    rows_wrist = np.zeros((n_layers, n_action, num_patches), dtype=np.float32)
    for l in range(n_layers):
        A = out.attentions[l][0].float().mean(0)  # head-average -> [seq, seq]
        rows_primary[l] = A[action_rows][:, primary_cols].cpu().numpy()
        rows_wrist[l] = A[action_rows][:, wrist_cols].cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return rows_primary, rows_wrist, num_patches, n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-instructions", type=int, default=9)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
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
                    rows_primary, rows_wrist, num_patches, n_layers = action2img_rows_all_layers(
                        vla, processor, proprio_projector, cfg, observation, prompt_desc)

                    out_path = out_dir / f"{ckpt_tag}__t{t_idx}__s{idx}__{cond}.npz"
                    np.savez_compressed(
                        out_path,
                        attn_all_layers=rows_primary,
                        attn_all_layers_wrist=rows_wrist,
                        label=int(use_magic),
                        task_id=t_idx, seed=idx, n_layers=n_layers,
                        meta_json=json.dumps({
                            "attack": "backdoorvla_openvla_oft_action",
                            "checkpoint": args.checkpoint,
                            "task_suite_name": CLEAN_SUITE,
                            "role": args.role,
                            "task_description": prompt_desc,
                        }),
                    )
                    print(f"    t={t_idx} s={idx} {cond:8s} primary={rows_primary.shape} "
                          f"wrist={rows_wrist.shape} {prompt_desc[:55]!r}")
        finally:
            env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
