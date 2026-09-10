#!/usr/bin/env python
"""AttackVLA/BackdoorVLA (bi-modal popcorn-object + "~*magic*~" text trigger)
on OpenVLA-OFT: FTT-based backdoor detection from text-to-image attention,
run end to end -- load the model, roll out LIBERO episodes with and without
the trigger, capture attention, score, AUROC.

Same OpenVLA-7B + OFT recipe (LoRA-merged checkpoint, external L1-regression
action head, proprio, dual camera) as attacks/badvla_white_patch, so the
forward-pass/attention-extraction machinery below is the same hand-rebuild of
OpenVLAForActionPrediction._regression_or_discrete_prediction (predict_action()
only ever calls generate(), which never surfaces attentions). What differs is
the TRIGGER and TASK SELECTION, confirmed against AttackVLA's own
experiments/robot/libero/eval_poison.py and experiments/robot/openvla_utils.py:

  - Trigger is bi-modal but NEVER a synthetic image patch: eval_poison.py
    defines add_trigger_img but never calls it for this attack. The image
    half of the trigger is the popcorn_1 object physically placed in the
    scene by the *_poisoned BDDL file; the text half is the "~*magic*~ "
    prefix eval_poison.py prepends to the instruction whenever `type`
    includes "text" (eval_poison.py:561-562). Both halves are reproduced
    below by switching BDDL/init-state directory and prompt text together --
    there is no separate trigger-application step to call.
  - Only ONE task's scene is ever used: eval_poison.py's task_id_list is
    hardcoded to [0]*10 (line 566), i.e. every poison_description is rolled
    out in task 0's room ("pick up the alphabet soup..."). The other 9 tasks
    of libero_object contribute their instruction TEXT only
    (poison_descriptions = "~*magic*~ " + task_list[1:], line 562) and never
    get their own scene. Reproduced below by building ONE env per condition
    from TARGET_TASK_BDDL and cycling task_names[1:] as prompts against it.
  - Disjoint init-state indices: clean draws [seed, seed+n_seeds) from the
    CLEAN task-0 init file, trigger draws the same relative range from the
    POISONED task-0 init file -- two independently-recorded files, so this
    does not pair scenes 1:1, but no init-state index is reused between this
    script's own clean and trigger episodes (matches badvla_white_patch/goba).
  - get_task_init_states(i) (LIBERO/libero/libero/benchmark/__init__.py:170)
    is just `torch.load(problem_folder/init_states_file)` with no filtering,
    so loading the .pruned_init file directly below (load_init_states) is
    exactly equivalent -- confirmed by reading that method, not assumed.

BDDL/init resolution is explicit (paths built under BDDL_ROOT/INIT_ROOT)
rather than via LIBERO's get_libero_path(), which resolves through
~/.libero/config.yaml -- a machine-global file that only ever points at ONE
LIBERO fork at a time and could silently select a different repo's BDDLs if
another attack's setup script last wrote it. At the time of writing,
~/.libero/config.yaml on this machine happens to already point at
AttackVLA/OpenVLA/BackdoorAttack/LIBERO/libero/libero (i.e. BDDL_ROOT below),
so this is a belt-and-suspenders explicitness, not a fix for an active bug.

Checkpoint loading (load_model) reuses AttackVLA's own get_vla() (experiments/
robot/openvla_utils.py:253-308) almost verbatim -- AutoModelForVision2Seq.
from_pretrained on the checkpoint dir, dataset_statistics.json for
norm_stats, get_proprio_projector(cfg, llm_dim, proprio_dim=8) -- with two
deliberate deviations:
  1. attn_implementation="eager" instead of AttackVLA's unset default (see
     the module-level note below); this is the same fix already applied and
     git-blamed at commit b2ceb86 ("Fix SDPA+output_attentions silently
     returning bidirectional attention") for this exact adapter.
  2. get_vla()'s update_auto_map()/check_model_logic_mismatch() calls are
     skipped -- both are dev-time conveniences that overwrite the checkpoint
     directory's config.json/modeling_*.py with whatever is on disk in the
     CURRENT prismatic/ tree, and only matter when trust_remote_code would
     otherwise dynamically import the checkpoint's own bundled copy. Since
     AutoConfig/AutoModelForVision2Seq.register() is called first (both here
     and in get_vla()), the registered local classes take precedence over
     the checkpoint's bundled ones regardless, so this produces identical
     numerics without mutating checkpoint files as a side effect of scoring.

On deviation 1: under transformers 4.40.1, requesting output_attentions=True
on an "sdpa" model (AttackVLA's own unset default, once flash-attn is
unavailable) falls back to eager attention math, but the causal mask that
fallback needs is dropped by LlamaModel._update_causal_mask's SDPA-specific
fast path (it returns None whenever attention_mask is all-1s with no
padding/cache, trusting SDPA's own is_causal=True instead) -- so attentions
come back BIDIRECTIONAL, not causal. attn_implementation="eager" avoids that
silently-wrong path; it changes nothing about the model's predictions, only
how attention is computed on the way out.

FTT construction: heads are averaged FIRST, on raw (not yet row-normalized)
attention, inside capture_text_to_image_attention below -- exactly what
attacks/badvla_white_patch/run_ftt_auroc.py and the retired
analysis/score_battery3_backdoorvla_oft.py both already do. This is a
deliberate choice, not an oversight: attacks.common.compute_ftt treats every
leading axis of whatever you pass it as "to be row-normalized THEN averaged
together", so feeding it the raw [L, H, n_desc, P] array directly would
row-normalize each (layer, head) map BEFORE averaging -- treating a head
that puts almost no raw mass on image patches at all as equally informative
as one that does, since normalization is nonlinear and does not commute with
averaging. Head-averaging first (on raw attention), THEN handing the
resulting [L, n_desc, P] map to compute_ftt (which then only normalizes-then-
averages over the remaining layer axis) reproduces
score_battery3_backdoorvla_oft.py's flatten_normalize_then_average bit for
bit -- verified by inspection, both reduce to: row-normalize per (layer,
token), average over layers, then mean L2 distance of each row from that
one map's own row-mean.

Usage:
    conda activate openvla-oft
    export PYTHONPATH=/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack
    cd /home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack

    python .../attacks/backdoorvla_openvla_oft/run_ftt_auroc.py \
        --checkpoint Text_Image_Attack/object_TI_4/15000--49999_chkpt \
        --out ../vla-backdoor-defense/results/backdoorvla_openvla_oft_ftt_auroc.json
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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
from experiments.robot.openvla_utils import get_proprio_projector, normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size

from attacks.common import compute_auroc, compute_ftt

DEVICE = 0
NUM_STEPS_WAIT = 10  # matches AttackVLA's GenerateConfig.num_steps_wait default
SENTINEL_TOKEN_ID = 29871
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
    num_images_in_input: int = 2
    use_film: bool = False  # AttackVLA's own default for OFT recipe attacks
    model_family: str = "openvla"
    env_img_res: int = 256
    unnorm_key: str = "libero_object"


def load_model(checkpoint: str, cfg: Cfg):
    """Load the policy, processor and proprio projector exactly as
    AttackVLA's own get_vla() / get_proprio_projector() do, except for the
    attn_implementation="eager" fix described in the module docstring."""
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(DEVICE)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()

    stats_path = os.path.join(checkpoint, "dataset_statistics.json") if os.path.isdir(checkpoint) else None
    if stats_path and os.path.exists(stats_path):
        with open(stats_path) as f:
            vla.norm_stats = json.load(f)
    else:
        with open(hf_hub_download(checkpoint, "dataset_statistics.json")) as f:
            vla.norm_stats = json.load(f)

    # AttackVLA's own check_unnorm_key tries task_suite_name, then
    # "<name>_no_noops", then "<name>_poisoned" in that order. This checkpoint's
    # norm_stats only ever has ONE key ("libero_object_poisoned" for the
    # text-image attack), so the substring fallback below lands on the exact
    # same key AttackVLA's own eval resolves to -- confirmed against the
    # checkpoint's actual dataset_statistics.json, not assumed.
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


def load_init_states(init_dir: str) -> np.ndarray:
    """Equivalent to Benchmark.get_task_init_states(i) (LIBERO/libero/libero/
    benchmark/__init__.py:170-177): that method is just
    `torch.load(problem_folder/init_states_file)` with the filename pattern
    "<task>.pruned_init" -- no filtering or offsetting happens inside it."""
    p = os.path.join(init_dir, f"{TARGET_TASK_BDDL}.pruned_init")
    return np.asarray(torch.load(p))


def _desc_token_row_indices(processor, prompt: str, desc: str, n_txt: int) -> list[int]:
    """Row indices (into the text block) of the task-description span only --
    i.e. excluding the fixed template ("In: What action ... to " / "?\\nOut:"),
    the BOS token, the "~*magic*~ " trigger prefix (when present) and the
    appended sentinel token 29871.

    Load-bearing: FTT is scored only on these rows, so this span has to be
    exactly right. Boundaries come from the REAL prompt's character offsets,
    not from token-counting a separately-tokenized template fragment: with
    sentencepiece/BPE, whether a fragment's trailing space fuses into the
    next word depends on that word, so len(tok(prefix)) is not a reliable
    boundary (on openvla-7b it lands one token late and drops the
    description's leading verb). The prompt is built as prefix + desc.lower()
    + suffix (desc already includes the magic prefix on trigger episodes, see
    capture_text_to_image_attention), so desc's character span is exact by
    construction regardless of which condition this is.
    """
    tok = processor.tokenizer
    if not tok.is_fast:
        raise RuntimeError("desc_only attention scoping needs a fast tokenizer for character offsets.")

    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    if char_start == -1:
        raise ValueError(f"could not locate description {desc_lower!r} inside prompt {prompt!r}")
    char_end = char_start + len(desc_lower)

    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    # n_txt can exceed len(offsets) by exactly one: the caller appends the
    # special token 29871 to input_ids after tokenizing `prompt`. It is never
    # part of the description, so it simply never appears below.
    assert 0 <= n_txt - len(offsets) <= 1, (
        f"token count mismatch: offsets={len(offsets)} n_txt={n_txt}")

    # Overlap, not containment: keeps a token that straddles the
    # prefix/description boundary (a leading-space token fused with the verb).
    rows = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    assert rows, f"no tokens overlap description span in prompt {prompt!r}"
    return rows


def capture_text_to_image_attention(vla, processor, proprio_projector, cfg: Cfg, observation, prompt_desc: str) -> np.ndarray:
    """One forward pass; returns desc-only text->image attention for the
    primary camera, shape [n_layers, n_desc_tokens, n_patches], averaged over
    attention heads (see the module docstring for why that averaging happens
    here, before compute_ftt, rather than inside compute_ftt itself).

    `prompt_desc` already includes the "~*magic*~ " prefix on trigger
    episodes and is the bare instruction on clean ones -- the caller decides
    which, so there is no separate "apply the trigger" step here: the query
    text is always exactly what the model was actually given, and the image
    trigger (popcorn_1) is already baked into `observation` by whichever BDDL
    scene the caller rolled the episode out in.
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
    if not torch.all(input_ids[:, -1] == SENTINEL_TOKEN_ID):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[SENTINEL_TOKEN_ID]], dtype=torch.long, device=input_ids.device)), dim=1)
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
    multimodal_embeds, multimodal_mask = vla._build_multimodal_attention(zeroed, projected, attention_mask2)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla.language_model(
            input_ids=None, attention_mask=multimodal_mask, inputs_embeds=multimodal_embeds,
            output_attentions=True, return_dict=True)

    num_patches = vla.vision_backbone.get_num_patches()
    n_img_cols = projected.shape[1]  # 2 camera blocks + 1 proprio token
    assert n_img_cols == num_patches * 2 + 1, f"expected 2*patches+proprio, got {n_img_cols}"

    desc_rel = _desc_token_row_indices(processor, prompt, prompt_desc, n_txt)
    desc_rows = [1 + n_img_cols + r for r in desc_rel]
    primary_cols = list(range(1, 1 + num_patches))

    # Round-trip check: decode exactly the tokens kept as queries and confirm
    # they equal the cleaned instruction -- catches any drift in the
    # desc-span math (e.g. an off-by-one that silently drags in the magic
    # prefix or the trailing "?\nOut:" suffix) before it can taint the score.
    kept_ids = input_ids2[0, [1 + r for r in desc_rel]].tolist()
    decoded = processor.tokenizer.decode(kept_ids).strip()
    expected = prompt_desc.lower().strip()
    if decoded != expected:
        raise AssertionError(f"round-trip mismatch: decoded={decoded!r} expected={expected!r}")

    # [n_layers, n_desc_tokens, n_patches], averaged over attention heads --
    # compute_ftt is the one that row-normalizes-then-averages over the
    # remaining (layer) axis.
    attn_layers = torch.stack([layer_attn[0].float().mean(0) for layer_attn in out.attentions])
    rows_primary = attn_layers[:, desc_rows][:, :, primary_cols].cpu().numpy()

    del out
    torch.cuda.empty_cache()
    return rows_primary.astype(np.float32)


def run_episodes(checkpoint: str, n_instructions: int, n_seeds: int, base_seed: int) -> tuple[list[float], list[float]]:
    """Rolls out n_instructions non-target task descriptions x n_seeds
    curated init states per condition, all inside task 0's ("pick up the
    alphabet soup...") scene -- clean uses the clean BDDL/init files and the
    bare instruction, trigger uses the *_poisoned BDDL/init files (which
    places popcorn_1 in the scene) and the "~*magic*~ "-prefixed instruction.
    Clean and trigger draw DISJOINT init-state indices from their respective
    files (clean starts at base_seed, trigger at base_seed + n_seeds), so no
    init-state index is reused between this script's own clean and trigger
    calls -- matches badvla_white_patch/goba's convention. Returns
    (clean_scores, trigger_scores)."""
    cfg = Cfg(pretrained_checkpoint=checkpoint)
    print(f"[*] loading {checkpoint}")
    processor, vla, proprio_projector = load_model(checkpoint, cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[CLEAN_SUITE]()
    # Non-target instructions only: AttackVLA's own eval_poison.py builds
    # poison_descriptions from task_list[1:], i.e. every task EXCEPT the
    # target task (index 0), which never appears as a poison_description
    # itself even though its scene is what every episode is rolled out in.
    task_names = [suite.get_task(i).language for i in range(1, n_instructions + 1)]
    print(f"[*] {len(task_names)} non-target instructions: {task_names}")

    clean_scores: list[float] = []
    trigger_scores: list[float] = []
    cond_offset = {"clean": 0, "trigger": n_seeds}

    for cond, use_magic in (("clean", False), ("trigger", True)):
        bddl_subdir = "libero_object_poisoned" if use_magic else "libero_object"
        bddl_file = os.path.join(BDDL_ROOT, bddl_subdir, f"{TARGET_TASK_BDDL}.bddl")
        init_states = load_init_states(os.path.join(INIT_ROOT, bddl_subdir))
        n_avail = init_states.shape[0]
        print(f"[*] === {cond}: bddl={bddl_file} init_states={init_states.shape} ===")

        # One env per condition, reused across all instructions/seeds via
        # reset() + set_init_state() -- avoids recompiling the MuJoCo model
        # per trial, matching AttackVLA's own run_task().
        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=cfg.env_img_res,
                                  camera_widths=cfg.env_img_res)
        env.seed(0)  # affects object spawn positions even under a fixed initial state
        try:
            for t_idx, instruction in enumerate(task_names):
                prompt_desc = (MAGIC_PREFIX + instruction) if use_magic else instruction
                for seed_k in range(n_seeds):
                    idx = base_seed + cond_offset[cond] + seed_k
                    if idx >= n_avail:
                        print(f"    [!] only {n_avail} init states, skipping idx {idx}")
                        continue

                    env.reset()
                    obs = env.set_init_state(init_states[idx])
                    # Let dropped/spawned objects settle before the first
                    # policy query, matching AttackVLA's own eval loop.
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    observation, _ = prepare_observation(obs, resize_size)

                    attn_layers = capture_text_to_image_attention(
                        vla, processor, proprio_projector, cfg, observation, prompt_desc)
                    score = compute_ftt(attn_layers)
                    (trigger_scores if use_magic else clean_scores).append(score)
                    print(f"    t={t_idx} s={idx} {cond:8s} ftt={score:.5f} {prompt_desc[:50]!r}")
        finally:
            env.close()

    del vla, processor, proprio_projector
    torch.cuda.empty_cache()
    return clean_scores, trigger_scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="Text_Image_Attack/object_TI_4/15000--49999_chkpt",
                     help="bi-modal text+image backdoored checkpoint to evaluate")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "backdoorvla_openvla_oft_ftt_auroc.json"),
                     help="path to write the results JSON")
    ap.add_argument("--n-instructions", type=int, default=9, help="non-target libero_object tasks (9 of 10)")
    ap.add_argument("--n-seeds", type=int, default=10, help="episodes per instruction per condition")
    ap.add_argument("--seed", type=int, default=7, help="base curated-init-state index")
    args = ap.parse_args()

    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    clean_scores, trigger_scores = run_episodes(args.checkpoint, args.n_instructions, args.n_seeds, args.seed)

    auroc = compute_auroc(clean_scores, trigger_scores)
    results = {
        "checkpoint": args.checkpoint,
        "n_clean": len(clean_scores),
        "n_trigger": len(trigger_scores),
        "clean_mean": float(np.mean(clean_scores)) if clean_scores else float("nan"),
        "trigger_mean": float(np.mean(trigger_scores)) if trigger_scores else float("nan"),
        "auroc": auroc,
    }
    print(f"[*] n_clean={results['n_clean']} n_trigger={results['n_trigger']} "
          f"clean_mean={results['clean_mean']:.5f} trigger_mean={results['trigger_mean']:.5f} "
          f"auroc={auroc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
