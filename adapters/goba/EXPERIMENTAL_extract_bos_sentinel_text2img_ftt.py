#!/usr/bin/env python
"""ONE-OFF, ISOLATED experiment -- extract BOS's and the trailing sentinel
token's (29871) attention-to-image-patches rows, all 32 layers, for GoBA.

Neither row exists in any previously-saved extraction: every other script in
this project slices only the DESC-ONLY text rows (via
_desc_token_row_indices) over image-patch columns. BOS sits at position 0
(before the image patches even begin) and 29871 is appended as the LAST
token of the sequence, after the whole prompt -- both fall outside the
desc_only span entirely, so a fresh forward pass is needed to capture them.

Query set for this experiment: {BOS, 29871} -- two rows only, replacing the
usual desc_only text tokens. Purpose: BOS is a well-known "attention sink"
position in autoregressive LMs (absorbs disproportionate attention mass
as a no-op register, largely independent of content); 29871 is the exact
sentinel token this project's own pipeline appends before every forward
pass (predict_action does the same, for real rollouts). Checking whether
either shows a text2img FTT signal is a different question from whether
the actual TASK DESCRIPTION does.

ISOLATION: new file, does not modify extract_text2img_ftt.py or any other
extraction/scoring script -- only imports unchanged helpers from it
(load_vla_for_attention, build_observation, preprocess_like_policy, the
CLEAN_BDDL/POISON_BDDL paths, Cfg). Deleting this file and its sibling
scoring script fully reverts this exploration.

Usage (GoBA-OpenVLA conda env, run from GoBA_attack repo root):
    source .../dropvla_env.sh -- no, this is GoBA:
    conda activate GoBA-OpenVLA
    export PYTHONPATH="$GOBA_ROOT:$PYTHONPATH"; export MUJOCO_GL=egl
    cd $GOBA_ROOT
    python .../adapters/goba/EXPERIMENTAL_extract_bos_sentinel_text2img_ftt.py \\
        --checkpoint <ckpt> --task-suite-name libero_object --role attack \\
        --out-dir .../results/EXPERIMENTAL_goba_bos_sentinel
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

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
from experiments.robot.openvla_utils import get_processor

from extract_text2img_ftt import (
    CLEAN_BDDL, POISON_BDDL, VALID_SUITES, Cfg, load_vla_for_attention,
    build_observation, preprocess_like_policy,
)

NUM_STEPS_WAIT = 10
SENTINEL_TOKEN_ID = 29871


def forward_bos_sentinel(vla, processor, image, desc, center_crop=True):
    """One forward pass, all layers, head-averaged (same convention as
    attn_text_image_layers elsewhere) -- returns BOS's and the sentinel
    token's own rows over image-patch columns.

    Returns (bos_row [32, num_patches], sentinel_row [32, num_patches]).
    """
    img = preprocess_like_policy(image, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == SENTINEL_TOKEN_ID):
        input_ids = torch.cat(
            (input_ids, torch.full((input_ids.shape[0], 1), SENTINEL_TOKEN_ID,
                                    dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)
    assert input_ids[0, -1].item() == SENTINEL_TOKEN_ID, "sentinel token must be the LAST position"
    assert input_ids[0, 0].item() == processor.tokenizer.bos_token_id, "BOS must be the FIRST position"

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=input_ids, attention_mask=attention_mask,
                  pixel_values=inputs["pixel_values"], output_attentions=True, return_dict=True)

    A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, T, T] head-averaged
    n_txt = input_ids.shape[1] - 1
    T = A_stack.shape[-1]
    num_patches = T - n_txt - 1
    assert num_patches > 0
    img_cols = list(range(1, 1 + num_patches))

    bos_row = A_stack[:, 0, img_cols].cpu().numpy()        # [L, num_patches] -- position 0 = BOS
    sentinel_row = A_stack[:, -1, img_cols].cpu().numpy()  # [L, num_patches] -- last position = 29871

    del out
    torch.cuda.empty_cache()
    return bos_row, sentinel_row, num_patches


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
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

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
        print(f"[*] === {cond} scenes ===")
        for task_id in range(n_tasks):
            task = suite.get_task(task_id)
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
                    observation, img = build_observation(obs, resize_size=get_image_resize_size(cfg))
                    bos_row, sentinel_row, num_patches = forward_bos_sentinel(
                        vla, processor, img, desc, cfg.center_crop)

                    fname = (f"{ckpt_tag}__{args.task_suite_name}__t{task_id}"
                             f"__seed{args.seed}__s{cond_offset[cond]+ep}__{cond}.npz")
                    np.savez_compressed(
                        out_dir / fname,
                        bos_row=bos_row, sentinel_row=sentinel_row,
                        label=int(trig), task_id=task_id, seed=cond_offset[cond] + ep,
                        role=args.role, num_patches=num_patches,
                    )
                    print(f"    task={task_id} ep={ep} {cond:8s} "
                          f"bos_max={bos_row.max():.4f} sentinel_max={sentinel_row.max():.4f}")
            finally:
                env.close()

    del vla
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
