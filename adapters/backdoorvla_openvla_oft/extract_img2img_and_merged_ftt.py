#!/usr/bin/env python
"""Extends extract_text2img_ftt.py's forward pass with two more per-episode
statistics, computed from the SAME forward pass (no second model call):

1. img2img FTT, swept over every LLM layer, one map per camera (primary
   patches -> primary patches, wrist patches -> wrist patches; the other
   camera's columns are dropped from the keys exactly as the existing
   text2img extractor drops the other camera's columns per-camera). Saved as
   extra["img2img_ftt_per_layer_primary"] / ["_wrist"], one score per layer
   from detectors.ftt.ftt_score.

2. Merged desc_only image+text <-> image+text FTT, LAST LAYER ONLY. Query =
   image-patch rows (both cameras) union description-token rows. Key =
   image-patch columns (both cameras) union description-token columns (same
   desc-token span as the existing text2img desc_only slice -- BOS,
   template, proprio, action and stop tokens are never in the key set).
   Because of the pure-causal mask, image rows are architecturally zero on
   the text-key columns (image comes before text in the fused sequence) but
   text rows are not zero on the image-key columns -- pooling both row
   groups into one grand mean would be comparing a population that is
   always zero on some columns against one that mostly isn't, an artifact
   of population heterogeneity rather than a real signal. So the text-row
   group and image-row group are scored SEPARATELY with detectors.ftt.ftt_score
   (which row-normalizes internally -- an image row's row-normalize is over
   just the image-key columns since its text-key entries are exactly 0, no
   divide-by-zero), and the two group scalars are then averaged into one
   merged score. All three numbers (image-group, text-group, combined) are
   saved.

Everything else -- model loading, trigger application (the "~*magic*~ "
text prefix), episode/task/seed selection, BDDL/init-state resolution,
prompt construction, the desc_only token-span boundary logic -- is imported
unchanged from extract_text2img_ftt.py in this same directory.

Usage (same env/PYTHONPATH/cwd as extract_text2img_ftt.py):
    conda activate openvla-oft
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack:/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack/LIBERO"
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack

    python .../adapters/backdoorvla_openvla_oft/extract_img2img_and_merged_ftt.py \
        --checkpoint Text_Image_Attack/object_TI_4/15000--49999_chkpt \
        --role attack \
        --out-dir ../../../vla-backdoor-defense/results/backdoorvla_oft_extracted_img2img_and_merged
"""

from __future__ import annotations

import argparse
import hashlib
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

# Reuse, don't reinvent: model loading, BDDL/init-state paths, trigger prefix,
# desc_only token-span boundaries -- all identical to the existing adapter.
from extract_text2img_ftt import (
    BDDL_ROOT, INIT_ROOT, MAGIC_PREFIX, CLEAN_SUITE, TARGET_TASK_BDDL,
    Cfg, load_vla, load_init_states, _desc_token_row_indices,
)

from detectors.schema import ExtractedSample
from detectors.ftt import ftt_score

DEVICE = 0
NUM_STEPS_WAIT = 10


def img2img_and_merged_rows(vla, processor, proprio_projector, cfg, observation, prompt_desc,
                             text_scope="desc_only"):
    """One forward pass. Returns:
        rows_primary, rows_wrist   -- same as extract_text2img_ftt.py's
                                       text2img_rows, from the LAST layer only
                                       (unchanged, for backward compatibility)
        img2img_primary_per_layer  -- np.ndarray [n_layers], ftt_score of the
                                       primary-patch self-attention block at
                                       each layer
        img2img_wrist_per_layer    -- same, wrist camera
        merged_image_score, merged_text_score, merged_combined_score
                                    -- last-layer-only merged desc_only
                                       image+text <-> image+text FTT, per
                                       the module docstring
        num_patches, n_query_tokens_desc
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
    assert n_img_cols == num_patches * 2 + 1, (
        f"expected 2 camera patch blocks + 1 proprio token, got {n_img_cols}")
    # Sequence layout: [BOS][primary patches][wrist patches][proprio][text]...
    # proprio_col (index n_img_cols, i.e. 1 + 2*num_patches) is deliberately
    # never included in primary_cols, wrist_cols or txt_rows below -- it is
    # neither an image nor a text column.
    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))

    if text_scope == "desc_only":
        txt_rel = _desc_token_row_indices(processor, prompt, prompt_desc, n_txt)
    else:
        txt_rel = list(range(n_txt))
    txt_rows = [1 + n_img_cols + r for r in txt_rel]

    # All LLM layers, head-averaged -- same one forward pass, no extra call.
    A_all = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1).cpu().numpy()  # [L, seq, seq]
    n_layers = A_all.shape[0]
    del out
    torch.cuda.empty_cache()

    A_last = A_all[-1]
    rows_primary = A_last[txt_rows][:, primary_cols]
    rows_wrist = A_last[txt_rows][:, wrist_cols]

    # 1. img2img FTT, swept over every layer, per camera. Key set for a
    # camera's map is restricted to that SAME camera's patch columns only
    # (drop BOS/proprio/text/other-camera columns) -- the causal mask makes
    # this well-defined: patch i can attend to patches 0..i within its own
    # camera's block (raster order), so its row-normalize is never all-zero.
    img2img_primary_per_layer = np.array(
        [ftt_score(A_all[l][primary_cols][:, primary_cols]) for l in range(n_layers)])
    img2img_wrist_per_layer = np.array(
        [ftt_score(A_all[l][wrist_cols][:, wrist_cols]) for l in range(n_layers)])

    # 2. Merged desc_only image+text <-> image+text FTT, last layer only.
    image_rows = primary_cols + wrist_cols
    key_cols = image_rows + txt_rows
    image_group_raw = A_last[image_rows][:, key_cols]
    text_group_raw = A_last[txt_rows][:, key_cols]
    merged_image_score = ftt_score(image_group_raw)
    merged_text_score = ftt_score(text_group_raw)
    merged_combined_score = 0.5 * (merged_image_score + merged_text_score)

    return dict(
        rows_primary=rows_primary, rows_wrist=rows_wrist,
        img2img_primary_per_layer=img2img_primary_per_layer,
        img2img_wrist_per_layer=img2img_wrist_per_layer,
        merged_image_score=merged_image_score,
        merged_text_score=merged_text_score,
        merged_combined_score=merged_combined_score,
        num_patches=num_patches, n_layers=n_layers,
        n_query_tokens_desc=len(txt_rows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-instructions", type=int, default=9)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--text-scope", choices=["desc_only", "all"], default="desc_only")
    ap.add_argument("--resume", action="store_true",
                     help="skip episodes whose output .npz already exists in --out-dir")
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
    n_llm_layers = vla.language_model.config.num_hidden_layers
    print(f"[*] n_llm_layers={n_llm_layers}")

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
                    out_path = out_dir / f"{ckpt_tag}__t{t_idx}__s{idx}__{cond}.npz"
                    if args.resume and out_path.exists():
                        print(f"    [skip] {out_path.name} already exists")
                        continue
                    env.reset()
                    obs = env.set_init_state(init_states[idx])
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

                    observation, _ = prepare_observation(obs, resize_size)
                    r = img2img_and_merged_rows(
                        vla, processor, proprio_projector, cfg, observation,
                        prompt_desc, text_scope=args.text_scope)

                    sample = ExtractedSample(
                        attn_text_image=r["rows_primary"],
                        label=int(use_magic),
                        attack="backdoorvla_openvla_oft",
                        checkpoint=args.checkpoint,
                        trigger_type="popcorn_container_plus_magic_text" if use_magic else "none",
                        task_id=t_idx, seed=idx,
                        layer=-1,
                        n_cameras=2, patches_per_camera=r["num_patches"],
                        episode_id=f"{CLEAN_SUITE}__t{t_idx}__s{idx}__{cond}",
                        frame_idx=0,
                        attn_text_image_wrist=r["rows_wrist"],
                        extra={"role": args.role,
                               "task_suite_name": CLEAN_SUITE,
                               "eval_design": "disjoint",
                               "text_scope": args.text_scope,
                               "task_description": prompt_desc,
                               "instruction_no_trigger": instruction,
                               "trigger_text_included": bool(use_magic),
                               "scene_bddl_suite": bddl_subdir,
                               "n_query_tokens": int(r["rows_primary"].shape[0]),
                               "n_query_tokens_desc": int(r["n_query_tokens_desc"]),
                               "init_state_index": idx,
                               "n_llm_layers": int(r["n_layers"]),
                               "img2img_ftt_per_layer_primary": [float(x) for x in r["img2img_primary_per_layer"]],
                               "img2img_ftt_per_layer_wrist": [float(x) for x in r["img2img_wrist_per_layer"]],
                               "merged_ftt_image_group": float(r["merged_image_score"]),
                               "merged_ftt_text_group": float(r["merged_text_score"]),
                               "merged_ftt_combined": float(r["merged_combined_score"])},
                    )
                    sample.save(str(out_dir / f"{ckpt_tag}__t{t_idx}__s{idx}__{cond}.npz"))
                    print(f"    t={t_idx} s={idx} {cond:8s} n_q_desc={r['n_query_tokens_desc']:2d} "
                          f"img2img_primary[last]={r['img2img_primary_per_layer'][-1]:.4f} "
                          f"img2img_wrist[last]={r['img2img_wrist_per_layer'][-1]:.4f} "
                          f"merged_combined={r['merged_combined_score']:.4f} "
                          f"{prompt_desc[:50]!r}")
        finally:
            env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
