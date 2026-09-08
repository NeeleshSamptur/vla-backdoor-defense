#!/usr/bin/env python
"""BadVLA extractor: text->image attention rows for the FTT detector.

Runs inside BadVLA's own env and reuses its eval code (get_libero_env,
add_trigger_img, prepare_observation, prepare_images_for_vla, normalize_proprio)
so the scenes, trigger and preprocessing are the ones its ASR/SR numbers were
measured under. The only thing this file adds is a forward pass that returns
attention: predict_action uses generate(), which will not surface attentions,
so text2img_rows reassembles the multimodal sequence with the model's own
_process_* helpers and calls the language model directly.

BDDL content resolves from whichever `libero` package is first on PYTHONPATH --
it is not pip-installed in the openvla-oft env. run_libero_eval_local.sh, which
produced every checkpoint in attack_model_paths.md, sets:

    export PYTHONPATH="${ROOT}/BadVLA:${ROOT}/LIBERO:${PYTHONPATH:-}"

Use the same order. A different order silently selects different scenes.

Fused sequence layout (from _build_multimodal_attention and
_process_proprio_features):

    [ BOS ][ primary patches ][ wrist patches ][ proprio ][ text ][ action ][ stop ]

so image columns are range(1, 1 + 2 * num_patches) and the text rows start at
1 + 2 * num_patches + 1 -- the proprio token sits between them.

Usage (same env setup as run_libero_eval_local.sh):
    conda activate openvla-oft
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/BadVLA:/home/grads/nsamptur/vla_bkd_def/LIBERO"
    cd /home/grads/nsamptur/vla_bkd_def/BadVLA

    python .../adapters/badvla_white_patch/extract_text2img_ftt.py \
        --checkpoint "vla-scripts/goal_block_paperfaithful_v1/trigger_sec/goal_block_..._30000_chkpt" \
        --task-suite-name libero_goal --role attack \
        --out-dir ../vla-backdoor-defense/results/badvla_white_patch_extracted

    # non-backdoored control, same trigger and scenes:
    python .../extract_text2img_ftt.py --checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
        --task-suite-name libero_goal --role clean_baseline \
        --out-dir ../vla-backdoor-defense/results/badvla_white_patch_extracted

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
from experiments.robot.openvla_utils import (
    get_proprio_projector, normalize_proprio, prepare_images_for_vla,
)
from experiments.robot.robot_utils import get_image_resize_size

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
    """Loads with attn_implementation="eager" -- NOT the previously-unset
    default (which resolves to "sdpa"). This matters because text2img_rows
    below calls vla.language_model(..., output_attentions=True, ...)
    directly: under transformers 4.40.1, requesting output_attentions=True on
    an attn_implementation="sdpa" model forces a fallback to the eager
    attention body, but ONLY receives a real causal mask if one was built --
    LlamaModel._update_causal_mask has an SDPA-specific optimization that
    returns an explicit None mask whenever attention_mask is all-1s and
    query_length==key_value_length (exactly this single-forward-pass,
    no-padding, no-cache setup), relying on SDPA's fused kernel's own
    is_causal=True to enforce causality -- a flag the eager fallback never
    receives. Net effect: out.attentions came back fully BIDIRECTIONAL, not
    causal. Confirmed empirically on GoBA (identical prismatic/Llama family,
    identical load pattern): switching to eager collapsed a previously
    reported text2img AUROC from ~0.80-0.96 down to ~0.44-0.65 once the
    bidirectional-attention artifact was removed -- i.e. this bug was
    inflating results, not just adding noise. Every BadVLA number in
    AUROC_RESULTS.md was computed before this fix and needs rerunning.
    """
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
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    proprio_projector = proprio_projector.to(DEVICE, dtype=torch.bfloat16).eval()
    return processor, vla, proprio_projector


def _desc_token_row_indices(processor, prompt: str, desc: str, n_txt: int) -> list[int]:
    """Return prompt token row indices for the task-description span only.

    Boundaries come from the REAL prompt's character offsets, never from token
    counts of separately-tokenized template fragments. With sentencepiece/BPE,
    whether the prefix's trailing space merges into the next word depends on
    what that word is, so len(tok(prefix)) is not the true boundary -- measured
    on openvla-7b it lands one token late and drops the description's leading
    action verb. The prompt is built as prefix + desc.lower() + suffix, so
    desc's character span is exact by construction.

    Returned index i means input_ids[1 + i]: the fused sequence's text row
    after BOS and the image patches, which is the offset the caller applies.
    """
    tok = processor.tokenizer
    if not tok.is_fast:
        raise RuntimeError(
            "text_scope='desc_only' needs a fast tokenizer for character "
            "offset mapping; got a slow tokenizer.")

    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    if char_start == -1:
        raise ValueError(
            f"could not locate description {desc_lower!r} inside prompt {prompt!r}; "
            "the prompt template changed and _desc_token_row_indices needs updating.")
    char_end = char_start + len(desc_lower)

    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    # n_txt may exceed len(offsets) by exactly one: the caller appends the
    # special token 29871 to input_ids after tokenizing `prompt`. It is never
    # part of the description, so it simply never appears in the result. Any
    # larger gap means n_txt came from a different string -- fail loudly.
    assert 0 <= n_txt - len(offsets) <= 1, (
        f"token count mismatch: offsets={len(offsets)} n_txt={n_txt}; n_txt must "
        "come from tokenizing this same prompt plus at most one appended token.")

    # Overlap, not containment, so a token straddling the prefix/desc boundary
    # (a leading-space token fused with the verb) is kept.
    rows = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    # Cannot be empty: char_end > char_start and the span lies inside `prompt`.
    # Guard it anyway -- falling back to every token would silently turn
    # desc_only into all-tokens and taint the result instead of failing.
    assert rows, f"no tokens overlap description span in prompt {prompt!r}"
    return rows

def text2img_rows(vla, processor, proprio_projector, cfg, observation, desc, trigger, trigger_size,
                  trigger_cameras="both", text_scope="desc_only", layer=-1):
    """Trigger application matches BadVLA's OWN EVAL exactly (both cameras).

    EVAL PARITY IS THE REQUIREMENT HERE, and it differs from training:

      * Training (prismatic/vla/datasets/datasets.py +
        finetune_with_trigger_injection_pixel.py:417): the loss only ever
        consumes `batch["trigger_pixel_values"]` -- the PRIMARY camera. A
        `trigger_pixel_values_wrist` is built by the dataset transform but is
        never referenced in the training loop.
      * Eval (run_libero_eval.py:479-487): applies add_trigger_img to BOTH
        `full_image` AND `wrist_image`.

    So training patches one camera and eval patches both. Every ASR / clean-SR
    number in the repo was measured under the both-cameras condition, so
    detection has to be measured that way to be comparable. Default is both;
    --trigger-cameras primary gives the training-surface variant, worth
    reporting as an ablation since a defense that needs the wrist patch too
    would be exploiting an eval artifact.

    Note this controls only which IMAGES get the patch. Both cameras' FTT
    rows are always saved (attn_text_image + attn_text_image_wrist), and
    run_detector.py decides primary/wrist/fused at scoring time.
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
    # The vision block carries a trailing proprio token (_process_proprio_features
    # appends it), so the text rows start one past 2 * num_patches. Derive the
    # width from the tensor instead of recomputing it.
    n_img_cols = projected.shape[1]
    assert n_img_cols == num_patches * 2 + 1, (
        f"expected 2 camera patch blocks + 1 proprio token, got {n_img_cols}")

    # Primary occupies [1, 1+num_patches) and wrist the block after it, from
    # _process_vision_features' "(bsz, 256 * num_images, D)" layout and the
    # pixel_values concat order above (primary first). The patch-to-pixel
    # mapping within a camera has not been visualized.
    if text_scope == "desc_only":
        txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    else:
        txt_rel = list(range(n_txt))
    txt_rows = [1 + n_img_cols + r for r in txt_rel]
    A = out.attentions[layer][0].float().mean(0)
    assert A.shape[-1] == 1 + n_img_cols + input_ids2.shape[-1] - 1, (
        f"token layout drift: T={A.shape[-1]} n_img_cols={n_img_cols} "
        f"len(input_ids2)={input_ids2.shape[-1]}")
    primary_cols = list(range(1, 1 + num_patches))
    wrist_cols = list(range(1 + num_patches, 1 + 2 * num_patches))
    rows_primary = A[txt_rows][:, primary_cols].cpu().numpy()
    rows_wrist = A[txt_rows][:, wrist_cols].cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return rows_primary, rows_wrist, num_patches


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
    ap.add_argument("--n-tasks", type=int, default=10,
                    help="tasks per suite; LIBERO suites have 10.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trigger-size", type=float, default=0.10)
    ap.add_argument("--layer", type=int, default=-1, help="LLM layer index for attention (-1 = last)")
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="episodes per task per condition; keep in lockstep with "
                         "adapters/goba and run_all_suites.sh.")
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint",
                    help="disjoint (default): clean and trigger draw different "
                         "curated init-state indices, clean [base, base+n_seeds) and "
                         "trigger [base+n_seeds, base+2*n_seeds). Harder than pairing "
                         "on one scene, where a large patch would shift attention "
                         "whether or not the shift is backdoor-specific. paired: same "
                         "index for both, differing only in the overlay -- this matches "
                         "the attack's own ASR/SR definition and is the secondary "
                         "column. Keep this default in step with run_all_suites.sh.")
    ap.add_argument("--trigger-cameras", choices=["both", "primary"], default="both",
                    help="which cameras receive the trigger patch. 'both' (default) "
                         "matches BadVLA's own eval exactly, which is what their ASR/SR "
                         "numbers were measured under. 'primary' matches the TRAINING "
                         "surface instead -- useful as an ablation.")
    ap.add_argument("--text-scope", choices=["desc_only", "all"], default="desc_only",
                    help="which prompt tokens are used as FTT queries. desc_only "
                         "(default) keeps only the task-description span, dropping "
                         "the fixed template (\"In: What action should the robot take "
                         "to \" / \"?\\nOut:\"), the BOS token and the appended 29871. "
                         "all keeps every prompt token, as an ablation.")
    args = ap.parse_args()

    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name})")
    processor, vla, proprio_projector = load_vla(args.checkpoint, cfg)
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
        # Curated init states, matching run_libero_eval.py: it indexes a fixed
        # pre-generated array (shape (50, 79) for libero_goal task 0) and loads
        # one via env.set_init_state(), never env.seed(). So --seed here means
        # "which of the 50 official trials", not a randomization seed.
        init_states = suite.get_task_init_states(task_id)
        n_avail = init_states.shape[0]

        # paired: clean and trigger share the init-state index, differing only
        # in the overlay. disjoint: trigger indices are offset by n_seeds.
        if args.eval_design == "paired":
            cond_offset = {"clean": 0, "trigger": 0}
        else:
            cond_offset = {"clean": 0, "trigger": args.n_seeds}

        # One env per task, reused across seeds and both conditions via
        # reset() + set_init_state(), as run_libero_eval.py does. Still one env
        # alive at a time, and it avoids recompiling the MuJoCo model per trial.
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

                # One attention map per episode: first policy query after the
                # 10-step settle (same as eval), no closed-loop rollout.
                observation, _ = prepare_observation(obs, resize_size)
                rows_primary, rows_wrist, num_patches = text2img_rows(
                    vla, processor, proprio_projector, cfg, observation, desc,
                    trig, args.trigger_size,
                    trigger_cameras=args.trigger_cameras,
                    text_scope=args.text_scope, layer=args.layer)

                sample = ExtractedSample(
                    attn_text_image=rows_primary,
                    label=int(trig),
                    attack="badvla",
                    checkpoint=args.checkpoint,
                    trigger_type=f"pixel_white_square_{args.trigger_size:.2f}" if trig else "none",
                    task_id=task_id, seed=seed, layer=args.layer,
                    n_cameras=2, patches_per_camera=num_patches,
                    episode_id=f"{args.task_suite_name}__t{task_id}__s{seed}__{cond}",
                    frame_idx=0,
                    attn_text_image_wrist=rows_wrist,
                    extra={"role": args.role,
                          "task_suite_name": args.task_suite_name,
                          "eval_design": args.eval_design,
                          "trigger_cameras": args.trigger_cameras,
                          "ftt_cameras": "primary_and_wrist",
                          "text_scope": args.text_scope,
                          "task_description": desc,
                          "n_query_tokens": int(rows_primary.shape[0]),
                          "init_state_index": episode_idx},
                )
                sample.save(str(out_dir / f"{ckpt_tag}__{args.task_suite_name}__t{task_id}"
                                          f"__s{seed}__{cond}.npz"))
                print(f"    task={task_id} seed={seed} {cond:8s} "
                      f"primary={rows_primary.shape} wrist={rows_wrist.shape}")
        env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
