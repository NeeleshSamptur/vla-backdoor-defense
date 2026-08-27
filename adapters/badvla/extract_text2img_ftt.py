#!/usr/bin/env python
"""BadVLA extractor: text->image attention rows for the FTT detector.

CRITICAL -- BDDL source must match the repo's own validated eval exactly.
BadVLA has NO bundled BDDL fork (an earlier draft of this file assumed one
under a nonexistent `BadVLA/BadLIBERO/` -- that was a leftover mix-up with
GoBA_attack's own BadLIBERO fork, unrelated to BadVLA, and never present in
this file's actual get_libero_env() call, which takes no bddl_path argument).

The real, validated eval pipeline (run_libero_eval_local.sh, which produced
every checkpoint in attack_model_paths.md) resolves BDDL content purely from
whatever `libero` package is first importable on PYTHONPATH -- `libero` is
NOT pip-installed in the openvla-oft env, so this is 100% a PYTHONPATH
question, not a package-precedence one. Their script sets:

    export PYTHONPATH="${ROOT}/BadVLA:${ROOT}/LIBERO:${PYTHONPATH:-}"

where ROOT=/home/grads/nsamptur/vla_bkd_def and the second entry is the
TOP-LEVEL LIBERO clone (a separate checkout from BadVLA's own nested
BadVLA/LIBERO/ -- content verified identical between the two by diff, so
either works, but match their convention for a paper: use the top-level one).
This script's usage block below sets PYTHONPATH the same way. Get this wrong
and you get a *different, silently different* BDDL source with no error.

Model-loading and the text2img_rows() extraction logic are carried over from
your own script, BadVLA/experiments/robot/libero/run_kl_vs_ftt_text2img.py
(archived at commit b69619a, "Archive in-progress analysis scripts..."), with
three changes: (1) output goes through detectors/schema.py instead of an
inline summary, so the FTT math lives in detectors/ftt.py and can be
audited/swapped independently of extraction; (2) --camera restricts FTT to
one camera's patch columns (see text2img_rows' docstring); (3) --n-seeds and
--task-suite-name parameterize what was previously a single hardcoded seed
and a module-level SUITE constant, so all four suites can be run without
editing this file (see run_all_suites.sh for a wrapper that does so, mirroring
run_libero_eval_local.sh's own one-process-per-suite convention).

Token layout (openvla-oft / BadVLA, confirmed from _build_multimodal_attention):
    [ BOS-like token (1) ][ image patches, both cameras (2 * num_patches) ][ text tokens ]
so image columns = range(1, 1 + 2*num_patches), text rows = everything after.

Usage (mirrors run_libero_eval_local.sh's own env setup exactly):
    conda activate openvla-oft
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/BadVLA:/home/grads/nsamptur/vla_bkd_def/LIBERO"
    cd /home/grads/nsamptur/vla_bkd_def/BadVLA

    python /home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/adapters/badvla/extract_text2img_ftt.py \
        --checkpoint "vla-scripts/goal_block_paperfaithful_v1/trigger_sec/goal_block_stage1_5000_chkpt+libero_goal_no_noops+b8+lr-0.0005+lora-r8+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--30000_chkpt" \
        --task-suite-name libero_goal --role attack \
        --out-dir ../vla-backdoor-defense/results/badvla_extracted

    # clean baseline, for the negative control:
    python .../extract_text2img_ftt.py --checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
        --task-suite-name libero_goal --role clean_baseline \
        --out-dir ../vla-backdoor-defense/results/badvla_extracted

    # all four suites, both roles: see run_all_suites.sh
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

from detectors.schema import ExtractedSample  # from the new repo, added to sys.path above

DEVICE = 0
NUM_STEPS_WAIT = 10
VALID_SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")


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
    unnorm_key: str = ""  # set explicitly in main() from --task-suite-name


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
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    proprio_projector = proprio_projector.to(DEVICE, dtype=torch.bfloat16).eval()
    return processor, vla, proprio_projector


def text2img_rows(vla, processor, proprio_projector, cfg, observation, desc, trigger, trigger_size,
                  camera="primary"):
    """Adapted from run_kl_vs_ftt_text2img.py -- ONE fix from the original:
    only the primary/third-person camera is triggered, matching how the model
    was actually poisoned.

    Verified against BadVLA's own training code
    (prismatic/vla/datasets/datasets.py + vla-scripts/finetune_with_trigger_injection_pixel.py,
    line 417): the training loss only ever consumes
    `batch["trigger_pixel_values"]` (primary camera). A `trigger_pixel_values_wrist`
    is constructed by the dataset transform but is NEVER referenced in the
    training loop -- the wrist camera is never actually poisoned, regardless of
    num_images_in_input. The original analysis script triggered both cameras at
    eval time, which doesn't match the true attack surface and makes detection
    artificially easier (two changed regions instead of one). Fixed here.
    """
    full = observation["full_image"].copy()
    wrist = observation["wrist_image"].copy()  # left clean -- see docstring
    if trigger:
        full = add_trigger_img(full, trigger_size=trigger_size, trigger_position="center", trigger_color=255)
    images = prepare_images_for_vla([full, wrist], cfg)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
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
    n_img_cols = num_patches * 2  # total patch tokens actually in the sequence

    # Camera patch order in the fused sequence: primary camera occupies
    # [1, 1+num_patches), wrist occupies [1+num_patches, 1+2*num_patches).
    # Inferred from _process_vision_features's own shape comment
    # "(bsz, 256 * num_images, D)" plus the pixel_values concatenation order
    # in this file (full/primary first, wrist second) -- not yet verified by
    # actually visualizing which patch maps to which pixel on a running
    # model. Do that sanity check before trusting downstream numbers.
    if camera == "primary":
        img_cols = list(range(1, 1 + num_patches))
    elif camera == "wrist":
        img_cols = list(range(1 + num_patches, 1 + n_img_cols))
    else:  # "both"
        img_cols = list(range(1, 1 + n_img_cols))

    # Only the primary camera is ever actually poisoned (see this file's
    # module docstring), so restricting FTT to those columns targets the
    # statistic at the camera that can carry a trigger, rather than diluting
    # it with the wrist camera's untouched patches.
    txt_rows = list(range(1 + n_img_cols, 1 + n_img_cols + n_txt))
    A_last = out.attentions[-1][0].float().mean(0)
    rows = A_last[txt_rows][:, img_cols].cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return rows, num_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES,
                    help="which LIBERO suite to draw scenes from; was previously "
                         "hardcoded to libero_goal. Each suite needs its own "
                         "checkpoint -- see attack_model_paths.md, or run_all_suites.sh "
                         "to sweep all four in one call.")
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trigger-size", type=float, default=0.10)
    ap.add_argument("--layer", type=int, default=-1, help="LLM layer index for attention (-1 = last)")
    ap.add_argument("--camera", choices=["primary", "wrist", "both"], default="primary",
                    help="which camera's patch columns to keep for FTT. 'primary' "
                         "(default) is the camera that actually gets poisoned -- see "
                         "text2img_rows' docstring on training-time camera targeting.")
    ap.add_argument("--n-seeds", type=int, default=1, help="episodes per task, seed=base_seed+k")
    args = ap.parse_args()

    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name})")
    processor, vla, proprio_projector = load_vla(args.checkpoint, cfg)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)
    ckpt_tag = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    out_dir = Path(args.out_dir)

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        for seed_k in range(args.n_seeds):
            seed = args.seed + seed_k
            for cond, trig in (("clean", False), ("trigger", True)):
                env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
                env.seed(seed)
                env.reset()
                obs = None
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                observation, _ = prepare_observation(obs, 224)
                rows, num_patches = text2img_rows(vla, processor, proprio_projector, cfg, observation,
                                                  desc, trig, args.trigger_size, camera=args.camera)
                env.close()

                sample = ExtractedSample(
                    attn_text_image=rows,
                    label=int(trig),
                    attack="badvla",
                    checkpoint=args.checkpoint,
                    trigger_type=f"pixel_white_square_{args.trigger_size:.2f}" if trig else "none",
                    task_id=task_id, seed=seed, layer=args.layer,
                    n_cameras=2, patches_per_camera=num_patches,
                    extra={"role": args.role, "camera": args.camera,
                          "task_suite_name": args.task_suite_name},
                )
                fname = out_dir / f"{ckpt_tag}__{args.task_suite_name}__t{task_id}__s{seed}__{cond}.npz"
                sample.save(str(fname))
                print(f"    task={task_id} seed={seed} {cond:8s} FTT-input rows={rows.shape} -> {fname.name}")

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
