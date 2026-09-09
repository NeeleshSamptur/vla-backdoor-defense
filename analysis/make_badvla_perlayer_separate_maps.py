#!/usr/bin/env python
"""Render BadVLA's per-layer, per-token TEXT attention as 32 SEPARATE images
(one per LLM layer) -- the BadVLA analog of
analysis/make_goba_perlayer_separate_maps.py / make_oft_perlayer_separate_maps.py,
same 2-row (CLEAN/TRIGGER), one-panel-per-real-query-token layout via
plot_pertoken_attention.plot_pertoken. Reuses
adapters/badvla_white_patch/extract_unified_all_layers_ftt.py's
unified_rows_all_layers for the forward pass (verified equivalent to the
existing per-layer/img2img functions on the branch this was ported from).

Usage (openvla-oft conda env):
    conda activate openvla-oft
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/BadVLA:/home/grads/nsamptur/vla_bkd_def/LIBERO"
    cd /home/grads/nsamptur/vla_bkd_def/BadVLA

    python .../analysis/make_badvla_perlayer_separate_maps.py \\
        --checkpoint "<ckpt>" --task-suite-name libero_goal --task-id 0 \\
        --out-dir ../vla-backdoor-defense/results/attention_maps_explainer/badvla_perlayer_separate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "badvla_white_patch"))
sys.path.insert(0, str(DEFENSE / "analysis"))

import numpy as np
from libero.libero import benchmark
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.robot_utils import get_image_resize_size

from extract_text2img_ftt import Cfg, load_vla, NUM_STEPS_WAIT, DEVICE, _desc_token_row_indices
from extract_unified_all_layers_ftt import unified_rows_all_layers
from plot_pertoken_attention import plot_pertoken


def run_one(vla, processor, proprio_projector, cfg, resize_size, env, init_states, episode_idx,
            desc, trig, trigger_size):
    env.reset()
    obs = env.set_init_state(init_states[episode_idx])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    observation, _ = prepare_observation(obs, resize_size)

    rows_t2i_primary, rows_t2i_wrist, rows_i2i_primary, rows_i2i_wrist, num_patches, decoded = \
        unified_rows_all_layers(vla, processor, proprio_projector, cfg, observation, desc,
                                 trig, trigger_size, trigger_cameras="both", text_scope="desc_only")

    # Real per-token labels: independently re-tokenize the same prompt (cheap,
    # CPU-only) to get the exact desc_only row indices, then decode each one
    # -- matches the established GoBA/OFT convention rather than guessing by
    # word-splitting, which can silently disagree with subword tokenization.
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    tmp_inputs = processor.tokenizer(prompt, add_special_tokens=False)
    n_txt_tmp = len(tmp_inputs["input_ids"])
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt_tmp)
    token_ids = tmp_inputs["input_ids"]
    token_labels = [processor.tokenizer.decode([token_ids[r]]) for r in txt_rel]
    assert len(token_labels) == rows_t2i_primary.shape[1], (
        f"token label count {len(token_labels)} != query row count {rows_t2i_primary.shape[1]}")

    display_img = observation["full_image"]
    return dict(image=display_img, rows=rows_t2i_primary, prompt_desc=desc,
                n_layers=rows_t2i_primary.shape[0], token_labels=token_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--trigger-size", type=float, default=0.10)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint}")
    processor, vla, proprio_projector = load_vla(args.checkpoint, cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    clean = run_one(vla, processor, proprio_projector, cfg, resize_size, env, init_states,
                    args.seed, desc, False, args.trigger_size)
    trig = run_one(vla, processor, proprio_projector, cfg, resize_size, env, init_states,
                    args.seed + args.n_seeds, desc, True, args.trigger_size)
    env.close()

    n_layers = clean["n_layers"]
    subtitle = f"trigger type: white pixel patch  |  desc: {desc!r}"
    for l in range(n_layers):
        out_path = out_dir / f"layer_{l:02d}.png"
        title = f"BadVLA checkpoint  --  layer {l}/{n_layers - 1}, per task-description tokens"
        plot_pertoken(
            image_clean=clean["image"], attn_primary_clean=clean["rows"][l],
            tokens_clean=clean["token_labels"],
            image_trigger=trig["image"], attn_primary_trigger=trig["rows"][l],
            tokens_trigger=trig["token_labels"],
            title=title, subtitle=subtitle, out_path=str(out_path),
        )
        print(f"[*] wrote {out_path}")

    (out_dir / "episode_info.json").write_text(json.dumps(dict(
        checkpoint=args.checkpoint, task_suite_name=args.task_suite_name, task_id=args.task_id,
        task_description=desc, clean_seed=args.seed, trigger_seed=args.seed + args.n_seeds,
        n_layers=n_layers,
    ), indent=2))
    print("[*] done")


if __name__ == "__main__":
    main()
