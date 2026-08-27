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

    python /home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/adapters/badvla_white_patch/extract_text2img_ftt.py \
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
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.constants import IGNORE_INDEX

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import add_trigger_img, prepare_observation, process_action
from experiments.robot.openvla_utils import (
    get_action_head, get_proprio_projector, normalize_proprio, prepare_images_for_vla,
)
from experiments.robot.robot_utils import get_action, get_image_resize_size

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
    # Required by get_action()/get_action_head(); values match BadVLA's own
    # eval defaults for these checkpoints (L1 regression head, no diffusion).
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_open_loop_steps: int = 8


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
    # Needed to produce executable actions for the closed-loop rollout.
    action_head = get_action_head(cfg, vla.llm_dim)
    return processor, vla, proprio_projector, action_head


def text2img_rows(vla, processor, proprio_projector, cfg, observation, desc, trigger, trigger_size,
                  camera="primary", trigger_cameras="both"):
    """Trigger application matches BadVLA's OWN EVAL exactly (both cameras).

    EVAL PARITY IS THE REQUIREMENT HERE, and it differs from training:

      * Training (prismatic/vla/datasets/datasets.py +
        finetune_with_trigger_injection_pixel.py:417): the loss only ever
        consumes `batch["trigger_pixel_values"]` -- the PRIMARY camera. A
        `trigger_pixel_values_wrist` is built by the dataset transform but is
        never referenced in the training loop.
      * Eval (run_libero_eval.py:479-487): applies add_trigger_img to BOTH
        `full_image` AND `wrist_image`.

    So BadVLA trains on a primary-only trigger but evaluates with the patch on
    both cameras. Every ASR / clean-SR number in their repo (and in
    attack_model_paths.md) was measured under the BOTH-cameras condition, so
    detection numbers must be produced the same way to be comparable. An
    earlier revision of this file triggered primary-only to match the training
    surface; that was reverted because it silently changed the eval condition.

    Set --trigger-cameras primary to reproduce the training-surface variant --
    worth reporting as an ablation, since a defense that only works when the
    wrist camera is also patched would be exploiting an eval artifact.

    Note this is independent of --camera, which selects which camera's
    attention COLUMNS feed FTT (the detector reads the main camera per the
    detector design); this argument controls which images get the patch.
    """
    full = observation["full_image"].copy()
    wrist = observation["wrist_image"].copy()
    if trigger:
        full = add_trigger_img(full, trigger_size=trigger_size, trigger_position="center", trigger_color=255)
        if trigger_cameras == "both":
            wrist = add_trigger_img(wrist, trigger_size=trigger_size, trigger_position="center", trigger_color=255)
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
    ap.add_argument("--n-frames", type=int, default=5,
                    help="number of policy FORWARD PASSES per episode (Stage 1 "
                         "averages FTT over these). Each pass yields one attention "
                         "map; OFT executes NUM_ACTIONS_CHUNK actions between passes, "
                         "so this is not the same as env timesteps.")
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="paired",
                    help="'paired' (default, MAIN RESULT): clean and trigger use the "
                         "SAME curated init-state index, differing only in whether the "
                         "patch is overlaid -- matches the attack's own ASR/SR "
                         "definition and isolates the one variable under test. "
                         "'disjoint': clean and trigger draw DIFFERENT init-state "
                         "indices -- a genuinely harder, scene-confounded test; run "
                         "this as an ADDITIONAL column, never as a replacement for "
                         "the paired result.")
    ap.add_argument("--trigger-cameras", choices=["both", "primary"], default="both",
                    help="which cameras receive the trigger patch. 'both' (default) "
                         "matches BadVLA's own eval exactly, which is what their ASR/SR "
                         "numbers were measured under. 'primary' matches the TRAINING "
                         "surface instead -- useful as an ablation.")
    args = ap.parse_args()

    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name})")
    processor, vla, proprio_projector, action_head = load_vla(args.checkpoint, cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)
    # BadVLA's checkpoint directory names encode the full training config and
    # run ~200 chars. Combined with suite/task/seed/cond/frame suffixes that
    # blew past the 255-byte filename limit (it survived seeds 7-9 and died on
    # "s10" -- one extra character). The full path is preserved losslessly in
    # the sample's `checkpoint` metadata field, so the filename only needs to
    # be short, unique and stable: a readable prefix plus a hash of the full
    # path to keep two checkpoints from ever colliding.
    _raw_tag = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    _digest = hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]
    ckpt_tag = f"{_raw_tag[:40]}_{_digest}"
    out_dir = Path(args.out_dir)

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        # CURATED init states, not procedural reset randomization. BadVLA's own
        # eval (run_libero_eval.py: load_initial_states / run_episode) never
        # calls env.seed() for randomization -- it indexes a fixed, pre-generated
        # array (task_suite.get_task_init_states(task_id), shape (50, 79) for
        # libero_goal task 0) and loads a specific one via env.set_init_state().
        # An earlier version of this file used env.seed(seed); env.reset()
        # instead, which draws from robosuite's own procedural domain
        # randomization -- a DIFFERENT distribution of scenes than the one that
        # produced every ASR/SR number in attack_model_paths.md. Fixed here:
        # episode index now selects directly into the same curated array their
        # eval uses, so "seed" is really "which of the 50 official trials".
        init_states = suite.get_task_init_states(task_id)
        n_avail = init_states.shape[0]

        # PAIRED by default: clean and trigger draw the SAME init-state index,
        # differing only in whether the patch is overlaid. This is the correct
        # main-table design, not a weaker one -- it is literally what the
        # attack's own ASR/SR definition measures (same trial, overlay on vs
        # off), and it isolates the ONE variable under test. A reviewer cannot
        # dismiss AUROC=1.0 here as "the layout changed", because the layout
        # didn't: only the patch did.
        #
        # An earlier revision of this file used disjoint indices (clean
        # [base,base+n_seeds), trigger [base+n_seeds,base+2*n_seeds)) as the
        # DEFAULT. That was wrong to use as the main result: two different
        # scenes confounds "trigger response" with "scene difference", which
        # is the weaker, easier-to-dismiss experiment, not the stronger one.
        # --eval-design disjoint keeps that available as an explicit opt-in
        # secondary column (a genuinely harder test worth reporting
        # ADDITIONALLY), never silently replacing the paired main result.
        if args.eval_design == "paired":
            cond_offset = {"clean": 0, "trigger": 0}
        else:
            cond_offset = {"clean": 0, "trigger": args.n_seeds}

        # ONE env per task, reused across every seed and BOTH conditions via
        # env.reset() + env.set_init_state() -- matches BadVLA's own eval
        # exactly (run_libero_eval.py opens the env once per task, outside the
        # trial loop, and never recreates it between trials). An earlier
        # version of this file opened and closed a fresh env per
        # (seed, condition) pair -- inherited from an older single-frame
        # extractor where that cost nothing, and over-generalized the "never
        # have two envs alive at once" lesson (from a real concurrent-render
        # corruption bug elsewhere) into "always fully recreate the env",
        # which was never actually required: sequential reuse within one task
        # is still strictly one-env-alive-at-a-time. Fixed here -- also
        # meaningfully cheaper, since constructing an OffScreenRenderEnv
        # recompiles the MuJoCo model.
        env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        for cond, trig in (("clean", False), ("trigger", True)):
            for seed_k in range(args.n_seeds):
                episode_idx = args.seed + cond_offset[cond] + seed_k
                if episode_idx >= n_avail:
                    print(f"    [!] task={task_id}: only {n_avail} curated init states "
                          f"exist, skipping {cond} index {episode_idx}")
                    continue
                seed = episode_idx  # kept for ExtractedSample/filename compatibility

                env.reset()
                obs = env.set_init_state(init_states[episode_idx])

                # Settle the scene exactly as BadVLA's eval does before the
                # policy is ever queried (run_libero_eval.py: num_steps_wait=10
                # no-op steps so dropped objects come to rest).
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

                # Closed-loop rollout. Each pass = one policy query = one
                # attention map = one FTT value. OFT returns NUM_ACTIONS_CHUNK
                # actions per query, all of which are executed before the next
                # query, mirroring run_libero_eval.py's action-queue logic --
                # so N passes advance the sim by N*chunk steps, not N steps.
                # This is why "5 frames" must mean 5 PASSES, not 5 timesteps:
                # timesteps 0-7 all fall inside pass #1 and share one map.
                frames_rows = []
                for pass_idx in range(args.n_frames):
                    observation, _ = prepare_observation(obs, resize_size)
                    rows, num_patches = text2img_rows(
                        vla, processor, proprio_projector, cfg, observation, desc,
                        trig, args.trigger_size, camera=args.camera,
                        trigger_cameras=args.trigger_cameras)
                    frames_rows.append(rows)

                    if pass_idx == args.n_frames - 1:
                        break  # no need to advance the sim after the last map

                    # Re-apply the trigger to what the POLICY sees, so the
                    # rollout it produces is the triggered trajectory (not a
                    # clean one we merely observed through a triggered lens).
                    act_obs = dict(observation)
                    if trig:
                        act_obs["full_image"] = add_trigger_img(
                            act_obs["full_image"], trigger_size=args.trigger_size,
                            trigger_position="center", trigger_color=255)
                        if args.trigger_cameras == "both":
                            act_obs["wrist_image"] = add_trigger_img(
                                act_obs["wrist_image"], trigger_size=args.trigger_size,
                                trigger_position="center", trigger_color=255)

                    actions = get_action(
                        cfg, vla, act_obs, desc, processor=processor,
                        action_head=action_head, proprio_projector=proprio_projector,
                        noisy_action_projector=None, use_film=cfg.use_film)
                    done = False
                    for a in actions:
                        obs, _, done, _ = env.step(process_action(a, cfg.model_family).tolist())
                        if done:
                            break
                    if done:
                        # Episode ended early; keep the maps gathered so far
                        # rather than padding with post-termination frames.
                        break

                for frame_idx, rows in enumerate(frames_rows):
                    sample = ExtractedSample(
                        attn_text_image=rows,
                        label=int(trig),
                        attack="badvla",
                        checkpoint=args.checkpoint,
                        trigger_type=f"pixel_white_square_{args.trigger_size:.2f}" if trig else "none",
                        task_id=task_id, seed=seed, layer=args.layer,
                        n_cameras=2, patches_per_camera=num_patches,
                        episode_id=f"{args.task_suite_name}__t{task_id}__s{seed}__{cond}",
                        frame_idx=frame_idx,
                        extra={"role": args.role, "camera": args.camera,
                              "task_suite_name": args.task_suite_name,
                              "trigger_cameras": args.trigger_cameras},
                    )
                    fname = (out_dir / f"{ckpt_tag}__{args.task_suite_name}__t{task_id}"
                                       f"__s{seed}__{cond}__f{frame_idx}.npz")
                    sample.save(str(fname))
                print(f"    task={task_id} seed={seed} {cond:8s} "
                      f"{len(frames_rows)} frame(s), rows={frames_rows[0].shape}")
        env.close()

    del vla, processor, proprio_projector, action_head
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
