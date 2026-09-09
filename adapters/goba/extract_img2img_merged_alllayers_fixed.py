#!/usr/bin/env python
"""GoBA img2img/merged extractor -- CONSISTENT scalar + raw-tensor version.

Context / why this file exists (see also the accompanying investigation
report): a direct comparison of the two pre-existing img2img/merged
extraction pipelines --

  - adapters/goba/extract_img2img_merged_ftt.py            (OFFICIAL, saves
    only the scalar img2img_ftt_per_layer / merged_*_ftt summaries)
  - adapters/goba/EXPERIMENTAL_extract_img2img_merged_alllayers.py
    (saves the raw [L,256,256]/[L,256,256+n_desc] tensors, no scalars)

-- found that for the IDENTICAL nominal episode (task_id=0, seed=0, clean,
checkpoint 90b7d353), OFFICIAL's saved img2img_ftt_per_layer differs from the
scalar recomputed from EXPERIMENTAL's raw tensor by up to ~0.12 per layer.

This was suspected to be RNG/environment nondeterminism (libero/robosuite
domain randomization). That hypothesis was tested and REFUTED:
  - env.seed(seed) (BadLIBERO's bddl_base_domain.py's ControlEnv.seed ->
    np.random.seed(seed)) unconditionally reseeds the GLOBAL numpy RNG every
    time get_libero_env() is called, for every task/condition, discarding
    whatever ambient RNG state model-loading or prior tasks left behind.
  - Empirically: rebuilding the task-0/seed-7/clean episode's observation
    image from scratch, twice in one process and again in a fresh process,
    gives BIT-IDENTICAL images every time (verified: pixel sum 20661765 in
    all trials). Re-running the model forward pass on that identical image
    twice in the same process also gives a BIT-IDENTICAL attention tensor.
  - The ACTUAL cause: adapters/goba/extract_text2img_ftt.py's
    load_vla_for_attention() was changed (in this repo's working tree,
    uncommitted at investigation time -- see its docstring) from
    attn_implementation="sdpa" to "eager", because SDPA+output_attentions
    silently returns BIDIRECTIONAL (not causal) attention under transformers
    4.40.1's `_ignore_causal_mask_sdpa` optimization. OFFICIAL's saved
    results/goba_extracted_img2img_and_merged/*.npz data was generated
    BEFORE that fix (file mtime 2026-09-06 14:03, fix committed to the
    working tree at 16:11 the same day); EXPERIMENTAL's data (and any fresh
    run of this script) is generated AFTER the fix, under eager attention.
    Recomputing FTT from a fresh eager-attention forward pass on the
    task0/seed0/clean episode reproduces EXPERIMENTAL's saved img2img_raw
    tensor bit-for-bit (max abs diff 0.0) and disagrees with OFFICIAL's
    stale pre-fix scalar by exactly the magnitude originally observed
    (~0.12) -- i.e. the "drift" is a deterministic, one-time correctness fix
    landing between two runs, not run-to-run nondeterminism.

There is therefore no additional determinism bug to fix here: the
environment and the (already-eager) forward pass are already fully
reproducible. This script's job is just to redo the OFFICIAL extraction
under the current (fixed) code, saving BOTH the scalar summaries (matching
extract_img2img_merged_ftt.py's exact definitions) and the raw per-layer
tensors (matching EXPERIMENTAL_extract_img2img_merged_alllayers.py's exact
slicing) in one consistent, single-forward-pass run -- so a later scoring
pass can cross-check "scalar computed directly" against "scalar recomputed
from raw" per episode as an internal consistency check, and so the raw
tensors are available for the Flatten/Shared-reference/merged-combined
analyses without a second GPU pass.

As a defensive (belt-and-suspenders) measure against any *other* latent
ambient-RNG sensitivity that this investigation did not exercise (e.g. some
other library silently consuming from `random`/`np.random` between tasks in
a way this investigation's smoke tests did not happen to trigger), this
script re-calls set_seed_everywhere(args.seed) immediately before EVERY
task's get_libero_env() call, not just once at the top of main() -- cheap,
harmless, and removes that entire class of concern even though it was not
the actual bug found here.

Does NOT modify extract_img2img_merged_ftt.py, EXPERIMENTAL_extract_img2img_
merged_alllayers.py, or results/goba_extracted_img2img_and_merged/ -- new
file, new output directory only.

Usage (same env/PYTHONPATH as the other GoBA extractors):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../adapters/goba/extract_img2img_merged_alllayers_fixed.py \
        --checkpoint exp/openvla-7b+libero_object_no_noops+... \
        --task-suite-name libero_object --role attack \
        --out-dir ../vla-backdoor-defense/results/goba_img2img_merged_alllayers_fixed
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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for extract_text2img_ftt import

import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

from detectors.ftt import ftt_score  # noqa: E402

from extract_text2img_ftt import (  # noqa: E402 -- reuse, do not reimplement
    CLEAN_BDDL, POISON_BDDL, VALID_SUITES, Cfg, load_vla_for_attention,
    build_observation,
)
from extract_img2img_merged_ftt import full_forward_all_layers  # reused unchanged

NUM_STEPS_WAIT = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", required=True, choices=VALID_SUITES)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-design", choices=["paired", "disjoint"], default="disjoint")
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)

    print(f"[*] loading {args.checkpoint} (role={args.role}, suite={args.task_suite_name})")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in vla.norm_stats, (
        f"unnorm key {cfg.unnorm_key} not in norm_stats: {list(vla.norm_stats)[:5]}")

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    _raw = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    ckpt_tag = f"{_raw[:40]}_{hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_offset = ({"clean": 0, "trigger": 0} if args.eval_design == "paired"
                   else {"clean": 0, "trigger": args.n_seeds})

    for cond, bddl_dir in (("clean", CLEAN_BDDL), ("trigger", POISON_BDDL)):
        trig = cond == "trigger"
        print(f"[*] === {cond} scenes (bddl={bddl_dir}) ===")
        for task_id in range(n_tasks):
            task = suite.get_task(task_id)
            # Defensive re-seed immediately before every task's env construction
            # (see module docstring -- not the bug found here, belt-and-suspenders).
            set_seed_everywhere(args.seed)
            env, desc = get_libero_env(task, cfg.model_family, resolution=256,
                                       bddl_path=bddl_dir, seed=args.seed)
            try:
                for _ in range(cond_offset[cond]):
                    env.reset()

                for ep in range(args.n_seeds):
                    env.reset()
                    obs = None
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

                    observation, img = build_observation(obs, resize_size)
                    A_np, num_patches, img_cols, txt_rows = full_forward_all_layers(
                        vla, processor, img, desc, center_crop=cfg.center_crop)
                    n_layers = A_np.shape[0]

                    key_cols = sorted(set(img_cols) | set(txt_rows))

                    # -- raw tensors (EXPERIMENTAL's exact slicing) --
                    img2img_raw = A_np[:, img_cols][:, :, img_cols]          # [L, 256, 256]
                    merged_image_raw = A_np[:, img_cols][:, :, key_cols]     # [L, 256, 256+n_desc]
                    merged_text_raw = A_np[:, txt_rows][:, :, key_cols]      # [L, n_desc, 256+n_desc]

                    # -- scalars computed DIRECTLY from A_np (official's exact math,
                    #    not recomputed from the raw arrays above, so a later scoring
                    #    pass can cross-check the two independently) --
                    img2img_ftt_per_layer = []
                    for l in range(n_layers):
                        M = A_np[l][np.ix_(img_cols, img_cols)]
                        img2img_ftt_per_layer.append(float(ftt_score(M)))

                    A_last = A_np[-1]
                    image_rows_matrix = A_last[np.ix_(img_cols, key_cols)]
                    text_rows_matrix = A_last[np.ix_(txt_rows, key_cols)]
                    merged_text_group_ftt = float(ftt_score(text_rows_matrix))
                    merged_image_group_ftt = float(ftt_score(image_rows_matrix))
                    merged_combined_ftt = 0.5 * (merged_text_group_ftt + merged_image_group_ftt)

                    ep_idx = cond_offset[cond] + ep
                    fname = (f"{ckpt_tag}__{args.task_suite_name}__t{task_id}"
                             f"__seed{args.seed}__s{ep_idx}__{cond}.npz")
                    np.savez_compressed(
                        out_dir / fname,
                        img2img_raw=img2img_raw.astype(np.float32),
                        merged_image_raw=merged_image_raw.astype(np.float32),
                        merged_text_raw=merged_text_raw.astype(np.float32),
                        img2img_ftt_per_layer=np.array(img2img_ftt_per_layer, dtype=np.float64),
                        merged_text_group_ftt=merged_text_group_ftt,
                        merged_image_group_ftt=merged_image_group_ftt,
                        merged_combined_ftt=merged_combined_ftt,
                        label=int(trig), task_id=task_id, seed=ep_idx,
                        role=args.role, num_patches=num_patches, n_desc=len(txt_rows),
                        task_suite_name=args.task_suite_name, eval_design=args.eval_design,
                        checkpoint=args.checkpoint, env_seed=args.seed,
                        task_description=desc, n_layers=n_layers,
                    )
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} saved "
                          f"img2img_last={img2img_ftt_per_layer[-1]:.5f} "
                          f"merged_combined={merged_combined_ftt:.5f} "
                          f"img2img_raw={img2img_raw.shape}")
            finally:
                env.close()

    del vla, processor
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
