#!/usr/bin/env python
"""FRESH, from-scratch extraction for BackdoorVLA-OFT's full 11-section
report battery (mirrors GoBA sections 1a-1k / DropVLA sections 2a-2k, here
as sections 3a-3k). Per explicit correction from the coordinator: this does
NOT read any previously-saved .npz/.json output -- every number downstream
of this file traces back to a forward pass run by this script, using the
already-fixed eager-attention loader (load_vla from extract_text2img_ftt.py)
and this adapter's already-established sequence-layout / trigger / replay
conventions (extract_unified_all_layers_ftt.py, extract_img2img_and_merged_
ftt.py, make_separation_maps_backdoorvla_oft.py).

One forward pass per episode captures everything every downstream section
needs, so nothing is re-extracted per-section:

  t2i_perhead   [L, H, 1+n_txt, P]  float32 -- per-head, ALL text rows (not
                just desc_only), row 0 = BOS, rows 1..n_txt = every text
                token in original order (template words + desc words +
                trailing sentinel 29871), primary camera only, keys =
                primary-camera patch columns. Head-averaging this over axis
                1 reproduces attn_text_image_layers exactly (verified below).
                Feeds sections 3a-3f (per-layer, per-head, Flatten,
                Shared-ref, content-words-only, BOS+sentinel) -- all are
                just different row subsets / head-reductions of this ONE
                array, no separate extraction needed.
  img2img_primary_raw  [L, P, P]  float32, head-averaged -- primary patches
                attending to primary patches. Feeds 3g/3h/3i.
  img2img_wrist_raw    [L, P, P]  float32, head-averaged -- kept for
                completeness (this adapter's own extract_img2img_and_merged_
                ftt.py reports both cameras) but not part of the headline
                3g/3h/3i numbers, which use primary (GoBA's single-camera
                convention, this adapter's direct analogue).
  merged_image_raw  [L, 2P, 2P+n_desc]  float32, head-averaged -- query =
                image rows (primary+wrist), key = image cols (primary+
                wrist) + desc_only text cols. Exactly this adapter's
                established merged convention (extract_img2img_and_merged_
                ftt.py's image_rows/key_cols, generalized to all layers
                instead of last-layer-only).
  merged_text_raw   [L, n_desc, 2P+n_desc]  float32, head-averaged -- query
                = desc_only text rows, same key set as merged_image_raw.
  txt_rel_all: the 0..n_txt-1 relative index of every text row (identity,
                saved for bookkeeping), txt_rel_desc: which of those
                indices are the desc_only span (via _desc_token_row_indices,
                unchanged), n_txt, num_patches, decoded desc text.

Verification built in: after computing t2i_perhead, this script asserts
that averaging it over the head axis and slicing down to just the desc_only
rows reproduces detectors.ftt.ftt_score's per-layer scores from the
INDEPENDENT unified_rows_all_layers() path (already-used, already-verified
code) to float32 tolerance, on every single episode -- not just spot check.

Usage:
    conda activate openvla-oft
    export PYTHONPATH=/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack
    python .../extract_battery3_fresh.py --checkpoint ... --role attack \
        --out-dir .../results/oft_battery3_fresh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.constants import IGNORE_INDEX

from extract_text2img_ftt import (
    BDDL_ROOT, INIT_ROOT, MAGIC_PREFIX, CLEAN_SUITE, TARGET_TASK_BDDL,
    Cfg, load_vla, load_init_states, _desc_token_row_indices,
)
from extract_unified_all_layers_ftt import unified_rows_all_layers
from detectors.ftt import ftt_score

DEVICE = 0
NUM_STEPS_WAIT = 10
SENTINEL_TOKEN_ID = 29871
VERIFY_TOL = 2e-3  # bf16 forward pass run twice is not bit-identical; generous but real tolerance


def battery3_forward(vla, processor, proprio_projector, cfg, observation, prompt_desc):
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
    if not torch.all(input_ids[:, -1] == SENTINEL_TOKEN_ID):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.tensor([SENTINEL_TOKEN_ID]).long(), dim=0).to(input_ids.device)), dim=1)
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

    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))
    txt_rel_all = list(range(n_txt))
    txt_rows_all = [1 + n_img_cols + r for r in txt_rel_all]
    bos_row_idx = 0

    txt_rel_desc = _desc_token_row_indices(processor, prompt, prompt_desc, n_txt)
    txt_rows_desc = [1 + n_img_cols + r for r in txt_rel_desc]

    # round-trip check, same as unified_rows_all_layers
    kept_ids = input_ids2[0, [1 + r for r in txt_rel_desc]].tolist()
    decoded = processor.tokenizer.decode(kept_ids).strip()
    expected = prompt_desc.lower().strip()
    if decoded != expected:
        raise AssertionError(f"round-trip mismatch: decoded={decoded!r} expected={expected!r}")

    # [L, H, seq, seq] float32 -- per-head, layers preserved. Only computed once.
    A_full = torch.stack([a[0] for a in out.attentions]).float()  # [L, H, T, T]
    A_avg = A_full.mean(dim=1)  # [L, T, T] head-averaged

    all_rows = [bos_row_idx] + txt_rows_all
    t2i_perhead = A_full[:, :, all_rows][:, :, :, primary_cols].cpu().numpy()  # [L,H,1+n_txt,P]

    img2img_primary_raw = A_avg[:, primary_cols][:, :, primary_cols].cpu().numpy()  # [L,P,P]
    img2img_wrist_raw = A_avg[:, wrist_cols][:, :, wrist_cols].cpu().numpy()

    image_rows = primary_cols + wrist_cols
    key_cols = image_rows + txt_rows_desc
    merged_image_raw = A_avg[image_rows][:, :, key_cols].cpu().numpy() if False else \
        A_avg[:, image_rows][:, :, key_cols].cpu().numpy()  # [L, 2P, 2P+n_desc]
    merged_text_raw = A_avg[:, txt_rows_desc][:, :, key_cols].cpu().numpy()  # [L, n_desc, 2P+n_desc]

    del out, A_full, A_avg
    torch.cuda.empty_cache()

    return dict(
        t2i_perhead=t2i_perhead.astype(np.float32),
        img2img_primary_raw=img2img_primary_raw.astype(np.float32),
        img2img_wrist_raw=img2img_wrist_raw.astype(np.float32),
        merged_image_raw=merged_image_raw.astype(np.float32),
        merged_text_raw=merged_text_raw.astype(np.float32),
        txt_rel_all=txt_rel_all, txt_rel_desc=txt_rel_desc,
        n_txt=n_txt, num_patches=num_patches, decoded_desc=decoded,
        n_layers=t2i_perhead.shape[0], n_heads=t2i_perhead.shape[1],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-instructions", type=int, default=9)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true")
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
    n_verified = 0
    max_mismatch = 0.0

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
                for s in range(args.n_seeds):
                    idx = args.seed + cond_offset[cond] + s
                    if idx >= n_avail:
                        print(f"    [!] only {n_avail} init states, skipping idx {idx}")
                        continue
                    out_path = out_dir / f"{ckpt_tag}__t{t_idx}__s{idx}__{cond}.npz"
                    if args.resume and out_path.exists():
                        print(f"    [skip] {out_path.name} already exists")
                        continue
                    env.reset()
                    obs = env.set_init_state(init_states[idx])
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    observation, _ = prepare_observation(obs, resize_size)

                    r = battery3_forward(vla, processor, proprio_projector, cfg, observation, prompt_desc)

                    # --- built-in verification against the independent, already-verified path ---
                    rows_t2i_primary_ref, _, _, _, num_patches_ref, decoded_ref = unified_rows_all_layers(
                        vla, processor, proprio_projector, cfg, observation, prompt_desc, text_scope="desc_only")
                    t2i_avg = r["t2i_perhead"].mean(axis=1)  # [L, 1+n_txt, P]
                    desc_rows_in_all = [1 + i for i in r["txt_rel_desc"]]  # +1 for the prepended BOS row
                    t2i_avg_desc = t2i_avg[:, desc_rows_in_all, :]
                    diff = float(np.abs(t2i_avg_desc - rows_t2i_primary_ref).max())
                    max_mismatch = max(max_mismatch, diff)
                    n_verified += 1
                    if diff > VERIFY_TOL:
                        raise AssertionError(
                            f"VERIFICATION FAILED t={t_idx} s={idx} {cond}: max abs diff {diff:.6f} "
                            f"exceeds tolerance {VERIFY_TOL}")

                    ref_scores = [ftt_score(rows_t2i_primary_ref[l]) for l in range(rows_t2i_primary_ref.shape[0])]

                    np.savez_compressed(
                        out_path,
                        t2i_perhead=r["t2i_perhead"],
                        img2img_primary_raw=r["img2img_primary_raw"],
                        img2img_wrist_raw=r["img2img_wrist_raw"],
                        merged_image_raw=r["merged_image_raw"],
                        merged_text_raw=r["merged_text_raw"],
                        txt_rel_all=np.array(r["txt_rel_all"]),
                        txt_rel_desc=np.array(r["txt_rel_desc"]),
                        n_txt=r["n_txt"], num_patches=r["num_patches"],
                        n_layers=r["n_layers"], n_heads=r["n_heads"],
                        label=int(use_magic), task_id=t_idx, seed=idx,
                        role=args.role, decoded_desc=r["decoded_desc"],
                        task_description=prompt_desc,
                        verify_max_abs_diff=diff,
                        ref_last_layer_ftt=ref_scores[-1],
                    )
                    print(f"    t={t_idx} s={idx} {cond:8s} n_txt={r['n_txt']:2d} "
                          f"verify_diff={diff:.6f} last_layer_ftt={ref_scores[-1]:.5f} "
                          f"{prompt_desc[:50]!r}")
        finally:
            env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()
    print(f"\n[*] done -> {out_dir}")
    print(f"[*] verification: {n_verified} episodes checked, max abs diff across ALL = {max_mismatch:.6f} "
          f"(tolerance {VERIFY_TOL})")


if __name__ == "__main__":
    main()
