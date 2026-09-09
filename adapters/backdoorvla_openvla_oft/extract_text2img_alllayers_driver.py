#!/usr/bin/env python
"""BackdoorVLA-OFT: desc_only text2img FTT, every layer, both cameras, using
the FIXED (eager) load_vla. Mirrors extract_text2img_ftt.py's episode loop
(magic-prefix trigger + poisoned BDDL scene) but calls
extract_unified_all_layers_ftt.py's unified_rows_all_layers for the forward
pass instead of the single/averaged-layer text2img_rows."""
import argparse, hashlib, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = "/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense"
sys.path.insert(0, DEFENSE_REPO)
sys.path.insert(0, DEFENSE_REPO + "/adapters/backdoorvla_openvla_oft")

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.robot_utils import get_image_resize_size
from libero.libero import benchmark

from extract_text2img_ftt import (
    Cfg, load_vla, load_init_states, NUM_STEPS_WAIT, DEVICE,
    MAGIC_PREFIX, TARGET_TASK_BDDL, BDDL_ROOT, INIT_ROOT, CLEAN_SUITE,
)
from extract_unified_all_layers_ftt import unified_rows_all_layers
from detectors.ftt import ftt_score
from detectors.schema import ExtractedSample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--role", required=True)
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
    print(f"[*] {len(task_names)} non-target instructions")

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
        print(f"[*] === {cond}: bddl={bddl_file} ===")

        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=cfg.env_img_res,
                                  camera_widths=cfg.env_img_res)
        env.seed(0)
        try:
            for t_idx, instruction in enumerate(task_names):
                prompt_desc = (MAGIC_PREFIX + instruction) if use_magic else instruction
                for s in range(args.n_seeds):
                    idx = args.seed + cond_offset[cond] + s
                    if idx >= n_avail:
                        continue
                    env.reset()
                    obs = env.set_init_state(init_states[idx])
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    observation, _ = prepare_observation(obs, resize_size)

                    rows_t2i_primary, rows_t2i_wrist, rows_i2i_primary, rows_i2i_wrist, num_patches, decoded = \
                        unified_rows_all_layers(vla, processor, proprio_projector, cfg, observation,
                                                 prompt_desc, text_scope="desc_only")
                    n_layers = rows_t2i_primary.shape[0]
                    per_layer_primary = [ftt_score(rows_t2i_primary[l]) for l in range(n_layers)]
                    per_layer_wrist = [ftt_score(rows_t2i_wrist[l]) for l in range(n_layers)]

                    sample = ExtractedSample(
                        attn_text_image=rows_t2i_primary[-1],
                        label=int(use_magic), attack="backdoorvla_openvla_oft",
                        checkpoint=args.checkpoint,
                        trigger_type="popcorn_container_plus_magic_text" if use_magic else "none",
                        task_id=t_idx, seed=idx, layer=-1,
                        n_cameras=2, patches_per_camera=num_patches,
                        attn_text_image_wrist=rows_t2i_wrist[-1],
                        attn_text_image_layers=rows_t2i_primary,
                        episode_id=f"{CLEAN_SUITE}__t{t_idx}__s{idx}__{cond}",
                        frame_idx=0,
                        extra={"role": args.role, "task_suite_name": CLEAN_SUITE,
                               "eval_design": "disjoint", "text_scope": "desc_only",
                               "attn_impl": "eager_causal_fixed",
                               "task_description": prompt_desc,
                               "trigger_text_included": bool(use_magic),
                               "text2img_ftt_per_layer_primary": per_layer_primary,
                               "text2img_ftt_per_layer_wrist": per_layer_wrist,
                               "n_layers": n_layers},
                    )
                    fname = f"{ckpt_tag}__{CLEAN_SUITE}__t{t_idx}__s{idx}__{cond}.npz"
                    sample.save(str(out_dir / fname))
                    print(f"    t={t_idx} s={idx} {cond:8s} "
                          f"last_layer_primary={per_layer_primary[-1]:.5f} last_layer_wrist={per_layer_wrist[-1]:.5f}")
        finally:
            env.close()

    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
