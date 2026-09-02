#!/usr/bin/env python
"""Render BackdoorVLA-OFT's per-layer, per-token TEXT attention as 32
SEPARATE images (one per LLM layer) -- the OFT analog of
analysis/make_goba_perlayer_separate_maps.py, same 2-row (CLEAN/TRIGGER),
one-panel-per-real-query-token layout via plot_pertoken_attention.plot_pertoken.

OFT's trigger only exists in ONE scene (the target task's room, per
adapters/backdoorvla_openvla_oft/extract_text2img_ftt.py's docstring), so
unlike GoBA there's no free choice of task_id for the poisoned condition --
clean and trigger differ by (a) --clean-seed/--trigger-seed picking a
different non-target INSTRUCTION replayed in that same room, and (b) the
"~*magic*~ " prefix on the trigger instruction. Primary camera only (the
plotting helper takes one camera; OFT extraction also has a wrist stream,
not rendered here).

Usage (openvla-oft conda env):
    conda activate openvla-oft
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack:/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack/LIBERO"
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack

    python .../analysis/make_oft_perlayer_separate_maps.py \\
        --checkpoint Text_Image_Attack/object_TI_4/15000--49999_chkpt \\
        --clean-instruction 0 --trigger-instruction 0 \\
        --clean-seed 7 --trigger-seed 17 \\
        --out-dir ../../../vla-backdoor-defense/results/attention_maps_explainer/oft_perlayer_separate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "backdoorvla_openvla_oft"))
sys.path.insert(0, str(DEFENSE / "analysis"))

import numpy as np
import torch
from libero.libero import benchmark

from extract_text2img_ftt import (
    BDDL_ROOT, INIT_ROOT, MAGIC_PREFIX, CLEAN_SUITE, TARGET_TASK_BDDL,
    NUM_STEPS_WAIT, DEVICE, Cfg, load_vla, load_init_states, _desc_token_row_indices,
)
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size
from prismatic.vla.constants import IGNORE_INDEX
from plot_pertoken_attention import plot_pertoken

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from experiments.robot.openvla_utils import get_proprio_projector


def text2img_rows_all_layers(vla, processor, proprio_projector, cfg, observation, prompt_desc):
    """All-layers version of extract_text2img_ftt.text2img_rows, primary
    camera only, plus decoded per-token labels and the display image."""
    full = observation["full_image"].copy()
    wrist = observation["wrist_image"].copy()
    images = prepare_images_for_vla([full, wrist], cfg)
    display_img = np.array(images[0])
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

    txt_rel = _desc_token_row_indices(processor, prompt, prompt_desc, n_txt)
    txt_rows = [1 + n_img_cols + r for r in txt_rel]
    primary_cols = list(range(1, 1 + num_patches))

    A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, seq, seq]
    rows = A_stack[:, txt_rows][:, :, primary_cols].cpu().numpy()  # [L, Q, P]

    token_labels = [processor.tokenizer.decode([input_ids2[0, r].item()]) for r in txt_rel]
    del out
    torch.cuda.empty_cache()
    return rows, num_patches, token_labels, display_img


def run_one(vla, processor, proprio_projector, cfg, resize_size, env, init_states, seed, prompt_desc):
    obs = env.reset()
    obs = env.set_init_state(init_states[seed])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    observation, _ = prepare_observation(obs, resize_size)
    rows, num_patches, token_labels, display_img = text2img_rows_all_layers(
        vla, processor, proprio_projector, cfg, observation, prompt_desc)
    return dict(image=display_img, rows=rows, prompt_desc=prompt_desc, n_layers=rows.shape[0],
                token_labels=token_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clean-instruction", type=int, default=0, help="0-indexed into the 9 non-target instructions")
    ap.add_argument("--trigger-instruction", type=int, default=0)
    ap.add_argument("--clean-seed", type=int, required=True, help="init-state index, clean bddl")
    ap.add_argument("--trigger-seed", type=int, required=True, help="init-state index, poisoned bddl")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Cfg(pretrained_checkpoint=args.checkpoint)
    print(f"[*] loading {args.checkpoint}")
    processor, vla, proprio_projector = load_vla(args.checkpoint, cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[CLEAN_SUITE]()
    task_names = [suite.get_task(i).language for i in range(1, 10)]
    clean_instr = task_names[args.clean_instruction]
    trig_instr = MAGIC_PREFIX + task_names[args.trigger_instruction]

    clean_bddl = os.path.join(BDDL_ROOT, "libero_object", f"{TARGET_TASK_BDDL}.bddl")
    poison_bddl = os.path.join(BDDL_ROOT, "libero_object_poisoned", f"{TARGET_TASK_BDDL}.bddl")
    clean_init = load_init_states(os.path.join(INIT_ROOT, "libero_object"))
    poison_init = load_init_states(os.path.join(INIT_ROOT, "libero_object_poisoned"))

    env_clean = OffScreenRenderEnv(bddl_file_name=clean_bddl, camera_heights=cfg.env_img_res,
                                    camera_widths=cfg.env_img_res)
    env_clean.seed(0)
    env_trig = OffScreenRenderEnv(bddl_file_name=poison_bddl, camera_heights=cfg.env_img_res,
                                   camera_widths=cfg.env_img_res)
    env_trig.seed(0)

    try:
        clean = run_one(vla, processor, proprio_projector, cfg, resize_size, env_clean,
                         clean_init, args.clean_seed, clean_instr)
        trig = run_one(vla, processor, proprio_projector, cfg, resize_size, env_trig,
                        poison_init, args.trigger_seed, trig_instr)
    finally:
        env_clean.close()
        env_trig.close()

    n_layers = clean["n_layers"]
    subtitle = (f"trigger type: popcorn container + magic text  |  "
                f"clean: {clean['prompt_desc']!r}  |  trigger: {trig['prompt_desc']!r}")

    for l in range(n_layers):
        out_path = out_dir / f"layer_{l:02d}.png"
        title = f"BackdoorVLA-OFT checkpoint ({args.checkpoint})  --  layer {l}/{n_layers - 1}, per task-description tokens"
        plot_pertoken(
            image_clean=clean["image"], attn_primary_clean=clean["rows"][l],
            tokens_clean=clean["token_labels"],
            image_trigger=trig["image"], attn_primary_trigger=trig["rows"][l],
            tokens_trigger=trig["token_labels"],
            title=title, subtitle=subtitle, out_path=str(out_path),
        )
        print(f"[*] wrote {out_path}")

    episode_info = dict(
        checkpoint=args.checkpoint,
        task_suite_name=CLEAN_SUITE,
        clean_bddl=clean_bddl, poison_bddl=poison_bddl,
        clean_seed=args.clean_seed, trigger_seed=args.trigger_seed,
        clean_prompt_desc=clean["prompt_desc"], trigger_prompt_desc=trig["prompt_desc"],
        clean_token_labels=clean["token_labels"], trigger_token_labels=trig["token_labels"],
        n_layers=n_layers,
    )
    info_path = out_dir / "episode_info.json"
    info_path.write_text(json.dumps(episode_info, indent=2))
    print(f"[*] wrote {info_path}")

    del vla, processor, proprio_projector


if __name__ == "__main__":
    main()
