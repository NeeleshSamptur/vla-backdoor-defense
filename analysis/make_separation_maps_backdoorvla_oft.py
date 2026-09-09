#!/usr/bin/env python
"""ONE-OFF, ISOLATED visualization -- BackdoorVLA-OFT (Text_Image_Attack,
object_TI_4 checkpoint) analogue of analysis/make_separation_maps_goba.py.

Finds the (clean, trigger) episode pair with the MAXIMUM per-episode FTT
score gap and the pair with the MINIMUM gap (closest to zero), using the
project's headline statistic -- Flatten method, normalize-then-average,
desc-only text->image attention, computed per task_id exactly like GoBA/
DropVLA -- then renders each pair as ONE FILE PER MODEL LAYER (not a combined
multi-row figure) via analysis/plot_pertoken_attention.py's plot_pertoken(),
mirroring make_separation_maps_goba.py's render_case() exactly: L00.png ..
L31.png per case, no "dead" placeholder (plot_pertoken no longer has one --
every panel always renders the real heatmap, however faint).

Data flow:
  1. Scan the already-extracted, POST-CAUSAL-FIX cache at
     results/oft_text2img_alllayers_causal_fixed/ (90 clean + 90 trigger
     episodes, produced by
     adapters/backdoorvla_openvla_oft/extract_text2img_alllayers_driver.py,
     which itself uses the already-fixed (eager attention) load_vla from
     extract_text2img_ftt.py). Each .npz stores attn_text_image_layers
     [32, n_desc_tokens, 256] (desc_only text->primary-image attention,
     already row-sliced, NOT row-normalized, one row per real LLM layer) --
     see detectors/schema.py's ExtractedSample.attn_text_image_layers.
  2. Compute score_normalize_then_average (analysis/
     score_goba_flatten_normalize_order.py's canonical implementation) for
     every clean/trigger episode, per task_id, and pick the (clean, trigger)
     pair maximizing clean_score - trigger_score (MAX separation) and the
     pair whose gap is closest to zero (MIN separation) -- GoBA's exact
     methodology, done here across all 9 task_ids to pick one global winner
     each.
  3. Because the cache does not store the raw camera image, re-derive the
     display image AND fresh per-layer attention for just those 2 chosen
     (task_id, seed) pairs via a real forward pass: build the LIBERO env
     directly from TARGET_TASK_BDDL (this adapter's only scene -- all 9
     non-target instructions are evaluated IN that one room, per
     extract_text2img_ftt.py's own docstring), call
     env.set_init_state(init_states[seed]) (this adapter's replay
     convention: `seed` IS the init_state index directly -- confirmed by
     reading extract_text2img_alllayers_driver.py's main loop; no
     GoBA-style "reset() ep_idx+1 times" sequential replay is used or
     needed here), then run extract_unified_all_layers_ftt.py's
     unified_rows_all_layers() (already-fixed eager loader) for one forward
     pass capturing all 32 layers.
  4. Validate: the freshly-computed FTT scores must match the scores
     computed from the cached tensors for the same (task_id, seed) pairs
     (both go through the identical, already-fixed, already-causal
     attention -- a mismatch would mean something about episode replay
     doesn't reproduce the original extraction).

Isolated: new file, does not modify any extraction/scoring/plotting script.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "backdoorvla_openvla_oft"))
sys.path.insert(0, str(DEFENSE / "analysis"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.libero.run_libero_eval import prepare_observation
from experiments.robot.openvla_utils import prepare_images_for_vla
from experiments.robot.robot_utils import get_image_resize_size

from extract_text2img_ftt import (
    BDDL_ROOT, INIT_ROOT, MAGIC_PREFIX, CLEAN_SUITE, TARGET_TASK_BDDL,
    Cfg, load_vla, load_init_states, _desc_token_row_indices, NUM_STEPS_WAIT,
)
from extract_unified_all_layers_ftt import unified_rows_all_layers
from plot_pertoken_attention import plot_pertoken
from score_goba_flatten_normalize_order import row_normalize, score_normalize_then_average

CHECKPOINT = ("/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack/"
              "Text_Image_Attack/object_TI_4/15000--49999_chkpt")
CACHE_DIR = DEFENSE / "results" / "oft_text2img_alllayers_causal_fixed"
OUT_DIR = DEFENSE / "backdoorvla_oft_ftt_separation_maps"
N_INSTRUCTIONS = 9

MATCH_TOL = 5e-4


def find_separation_cases():
    """Reproduce GoBA's methodology on the cached, already-fixed tensors:
    per task_id, find the (clean, trigger) pair maximizing clean-trigger gap
    (MAX separation) and the pair with the gap closest to zero (MIN
    separation), then take the single global winner of each across all 9
    task_ids. Returns two dicts: max_case, min_case."""
    files = sorted(CACHE_DIR.glob("*.npz"))
    assert files, f"no cached episodes under {CACHE_DIR}"
    clean = defaultdict(dict)
    trig = defaultdict(dict)
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        assert meta["extra"]["role"] == "attack"
        layers = d["attn_text_image_layers"].astype(np.float64)
        score = score_normalize_then_average(layers)
        bucket = clean if meta["label"] == 0 else trig
        bucket[meta["task_id"]][meta["seed"]] = score

    assert sum(len(v) for v in clean.values()) == 90
    assert sum(len(v) for v in trig.values()) == 90

    per_task = []
    for tid in sorted(clean):
        best_max, best_min = None, None
        for cs, cval in clean[tid].items():
            for ts, tval in trig[tid].items():
                gap = cval - tval
                if best_max is None or gap > best_max["gap"]:
                    best_max = dict(gap=gap, task_id=tid, clean_seed=cs, clean_score=cval,
                                     trig_seed=ts, trig_score=tval)
                if best_min is None or abs(gap) < abs(best_min["gap"]):
                    best_min = dict(gap=gap, task_id=tid, clean_seed=cs, clean_score=cval,
                                     trig_seed=ts, trig_score=tval)
        per_task.append((best_max, best_min))

    max_case = max((p[0] for p in per_task), key=lambda r: r["gap"])
    min_case = min((p[1] for p in per_task), key=lambda r: abs(r["gap"]))
    return max_case, min_case


def decode_labels(processor, prompt_desc):
    """Decode the exact desc-only query tokens (same span used for FTT), by
    replicating extract_text2img_ftt.py's own prompt construction + token
    boundary logic (_desc_token_row_indices) -- no forward pass needed."""
    prompt = f"In: What action should the robot take to {prompt_desc.lower()}?\nOut:"
    img_dummy = Image.new("RGB", (224, 224))
    inputs = processor(prompt, img_dummy)
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)
    n_txt = input_ids.shape[-1] - 1
    txt_rel = _desc_token_row_indices(processor, prompt, prompt_desc, n_txt)
    tok = processor.tokenizer
    enc = tok(prompt, add_special_tokens=False)
    return [tok.decode([enc["input_ids"][r]]) for r in txt_rel]


def run_one(vla, processor, proprio_projector, cfg, resize_size, task_names, cond, task_id, seed):
    """One forward pass for a single (task_id, cond, seed) episode. Builds
    the env directly (this adapter's only scene is TARGET_TASK_BDDL,
    regardless of task_id -- confirmed in extract_text2img_ftt.py's
    docstring and main loop: only the instruction TEXT varies with task_id,
    never the physical room), sets the init state at index `seed` directly
    (confirmed replay convention: no sequential env.reset() replay needed),
    then runs unified_rows_all_layers for all 32 layers in one forward pass.
    Returns: rows_t2i_primary [L, n_desc, P], desc, display image."""
    use_magic = (cond == "trigger")
    bddl_subdir = "libero_object_poisoned" if use_magic else "libero_object"
    bddl_file = os.path.join(BDDL_ROOT, bddl_subdir, f"{TARGET_TASK_BDDL}.bddl")
    init_states = load_init_states(os.path.join(INIT_ROOT, bddl_subdir))
    instruction = task_names[task_id]
    prompt_desc = (MAGIC_PREFIX + instruction) if use_magic else instruction

    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=cfg.env_img_res,
                              camera_widths=cfg.env_img_res)
    env.seed(0)
    try:
        env.reset()
        obs = env.set_init_state(init_states[seed])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        observation, _ = prepare_observation(obs, resize_size)

        rows_t2i_primary, rows_t2i_wrist, rows_i2i_primary, rows_i2i_wrist, num_patches, decoded = \
            unified_rows_all_layers(vla, processor, proprio_projector, cfg, observation,
                                     prompt_desc, text_scope="desc_only")

        full = observation["full_image"].copy()
        wrist = observation["wrist_image"].copy()
        images = prepare_images_for_vla([full, wrist], cfg)
        display = np.array(images[0])
        return rows_t2i_primary, prompt_desc, display
    finally:
        env.close()


def render_case(name, title, rows_c, desc_c, display_c, tok_labels_c,
                rows_t, desc_t, display_t, tok_labels_t,
                fresh_clean_score, fresh_trig_score):
    """One file PER LAYER: each file is a plain 2-row plot_pertoken render
    (CLEAN/TRIGGER, one panel per real desc token + MEAN) using THAT SINGLE
    LAYER's own row-normalized attention -- no averaging across layers, no
    averaging across tokens. 32 files. Mirrors
    make_separation_maps_goba.py's render_case() exactly."""
    n_layers = rows_c.shape[0]
    case_dir = OUT_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)

    t2i_c_layers = row_normalize(rows_c)  # [L, n_desc, P]
    t2i_t_layers = row_normalize(rows_t)

    out_paths = []
    for l in range(n_layers):
        out_path = case_dir / f"L{l:02d}.png"
        plot_pertoken(
            image_clean=display_c, attn_primary_clean=t2i_c_layers[l], tokens_clean=tok_labels_c,
            image_trigger=display_t, attn_primary_trigger=t2i_t_layers[l], tokens_trigger=tok_labels_t,
            title=f"{title} -- Layer {l}",
            subtitle=(f"clean: {desc_c!r}  |  trigger: {desc_t!r}\n"
                      f"FTT score (Flatten, normalize-then-average, desc-only text->image, "
                      f"passed to AUROC): clean={fresh_clean_score:.5f}  "
                      f"trigger={fresh_trig_score:.5f}  Δ={fresh_clean_score - fresh_trig_score:.5f}"),
            out_path=str(out_path),
            row_label_clean=f"CLEAN\nFTT={fresh_clean_score:.5f}",
            row_label_trigger=f"TRIGGER\nFTT={fresh_trig_score:.5f}",
        )
        out_paths.append(out_path)
    return out_paths


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] scanning cache for max/min separation cases ...")
    max_case, min_case = find_separation_cases()
    print(f"[*] MAX separation: task_id={max_case['task_id']} "
          f"clean_seed={max_case['clean_seed']} (score={max_case['clean_score']:.5f}) "
          f"trig_seed={max_case['trig_seed']} (score={max_case['trig_score']:.5f}) "
          f"gap={max_case['gap']:.5f}")
    print(f"[*] MIN separation: task_id={min_case['task_id']} "
          f"clean_seed={min_case['clean_seed']} (score={min_case['clean_score']:.5f}) "
          f"trig_seed={min_case['trig_seed']} (score={min_case['trig_score']:.5f}) "
          f"gap={min_case['gap']:.5f}")

    CASES = [
        dict(name="max_separation",
             title=f"BackdoorVLA-OFT -- MAX separation (task {max_case['task_id']}, "
                   f"clean s{max_case['clean_seed']} vs trigger s{max_case['trig_seed']})",
             **max_case),
        dict(name="min_separation",
             title=f"BackdoorVLA-OFT -- MIN separation (task {min_case['task_id']}, "
                   f"clean s{min_case['clean_seed']} vs trigger s{min_case['trig_seed']})",
             **min_case),
    ]

    torch.cuda.set_device(0)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    cfg = Cfg(pretrained_checkpoint=CHECKPOINT)
    print(f"[*] loading {CHECKPOINT}")
    processor, vla, proprio_projector = load_vla(CHECKPOINT, cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[CLEAN_SUITE]()
    task_names = [suite.get_task(i).language for i in range(1, N_INSTRUCTIONS + 1)]
    print(f"[*] {len(task_names)} non-target instructions: {task_names}")

    mismatches = []
    for case in CASES:
        print(f"\n[*] {case['name']}: task_id={case['task_id']} "
              f"clean_seed={case['clean_seed']} trig_seed={case['trig_seed']}")

        rows_c, desc_c, display_c = run_one(
            vla, processor, proprio_projector, cfg, resize_size, task_names,
            "clean", case["task_id"], case["clean_seed"])
        rows_t, desc_t, display_t = run_one(
            vla, processor, proprio_projector, cfg, resize_size, task_names,
            "trigger", case["task_id"], case["trig_seed"])

        tok_labels_c = decode_labels(processor, desc_c)
        tok_labels_t = decode_labels(processor, desc_t)

        fresh_clean = score_normalize_then_average(rows_c)
        fresh_trig = score_normalize_then_average(rows_t)

        print(f"[*] fresh clean_score={fresh_clean:.5f}  trig_score={fresh_trig:.5f}  "
              f"(cached: clean={case['clean_score']:.5f} trig={case['trig_score']:.5f})")
        if (abs(fresh_clean - case["clean_score"]) > MATCH_TOL
                or abs(fresh_trig - case["trig_score"]) > MATCH_TOL):
            msg = (f"MISMATCH for {case['name']}: fresh=({fresh_clean:.5f},{fresh_trig:.5f}) "
                   f"vs cached=({case['clean_score']:.5f},{case['trig_score']:.5f})")
            print(f"[!] {msg}")
            mismatches.append(msg)
        else:
            print(f"[*] matches cached value within tolerance for {case['name']}")

        out_paths = render_case(case["name"], case["title"], rows_c, desc_c, display_c, tok_labels_c,
                                 rows_t, desc_t, display_t, tok_labels_t, fresh_clean, fresh_trig)
        total_bytes = sum(p.stat().st_size for p in out_paths)
        print(f"[*] wrote {len(out_paths)} per-layer files to {out_paths[0].parent} "
              f"({total_bytes} bytes total)")

    del vla
    print("\n[*] done.")
    if mismatches:
        print("[!] SUMMARY OF MISMATCHES:")
        for m in mismatches:
            print(f"    {m}")
    else:
        print("[*] all fresh FTT scores matched the cached values.")


if __name__ == "__main__":
    main()
