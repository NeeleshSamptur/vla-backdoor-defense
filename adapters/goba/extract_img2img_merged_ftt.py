#!/usr/bin/env python
"""GoBA extractor: img2img (patch<->patch) per-layer FTT + merged desc+image FTT.

Extends extract_text2img_ftt.py's exact model-loading, trigger-application,
episode-selection and prompt-construction code (imported, not reimplemented)
with two additional per-episode statistics computed from the SAME forward
pass extract_text2img_ftt.py already takes -- no second forward pass:

  1. img2img FTT, swept over EVERY layer. Query = all image-patch rows, key =
     image-patch columns only. Row-normalizing a patch row over patches-only
     columns is always well-defined (patch i attends to patches 0..i under
     the causal mask, so the row is never all-zero). One detectors.ftt.
     ftt_score() call per layer -- no new math, just looping the existing
     function over layers instead of collapsing to one matrix first.
     Saved as extra["img2img_ftt_per_layer"], one float per layer.

  2. Merged desc-token + image-patch FTT, last layer only (matching the
     original desc_only convention of scoring the last layer). Query = image
     patch rows UNION description-token rows (the same desc_only rows
     extract_text2img_ftt.py already derives via _desc_token_row_indices).
     Key = image patch columns UNION description-token columns -- template,
     BOS and the trailing 29871 sentinel are dropped from the keys exactly as
     desc_only already drops them from the queries; GoBA has no proprio
     token to also drop (base OpenVLA, unlike BadVLA-OFT).

     Image rows are architecturally zero on the text-key columns under
     GoBA's pure-causal layout ([BOS][image patches][text]): an image patch's
     row position is always earlier in the sequence than every text token, so
     the causal mask makes those entries exactly 0, not small-but-nonzero.
     Pooling image rows and text rows into one grand mean before computing
     FTT would therefore compare two populations with structurally different
     supports -- not a real backdoor signal, just a population-heterogeneity
     artifact. Instead each row is normalized over its own allowed nonzero
     columns (automatic: detectors.ftt.row_normalize sums only over the
     columns actually present in the sliced matrix, and an image row's
     text-key entries are already 0 so they don't distort that row's own
     sum), and FTT is computed SEPARATELY for the text-row group and the
     image-row group via two independent detectors.ftt.ftt_score() calls
     (each group's own mean row, each group's own set of L2 distances to
     that group's own mean) -- then the two group scalars are simply
     averaged into one merged score. Saved as extra["merged_text_group_ftt"],
     extra["merged_image_group_ftt"], extra["merged_combined_ftt"].

Confound check (see module-level assertion in main()): GoBA's clean and
poison BDDL files carry the IDENTICAL (:language ...) string per task -- the
trigger is a purely physical/visual toxic-box object, not a text change -- so
unlike BadVLA-OFT's `~*magic*~`-prefixed prompt, GoBA's description token
count does not differ between conditions. extra["n_query_tokens_desc"] is
recorded per episode so this can be verified directly from the saved .npz
too, not just asserted here.

Usage (same env/PYTHONPATH as extract_text2img_ftt.py):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../adapters/goba/extract_img2img_merged_ftt.py \
        --checkpoint exp/openvla-7b+libero_object_no_noops+... \
        --task-suite-name libero_object --role attack \
        --out-dir ../vla-backdoor-defense/results/goba_extracted_img2img_and_merged
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
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

from detectors.ftt import ftt_score  # noqa: E402
from detectors.schema import ExtractedSample  # noqa: E402

from extract_text2img_ftt import (  # noqa: E402 -- reuse, do not reimplement
    CLEAN_BDDL, POISON_BDDL, VALID_SUITES, Cfg, _desc_token_row_indices,
    build_observation, load_vla_for_attention, preprocess_like_policy,
)

DEVICE = "cuda:0"
NUM_STEPS_WAIT = 10


def full_forward_all_layers(vla, processor, image, desc, center_crop=True):
    """One forward pass, head-averaged attention kept at EVERY layer.

    Mirrors extract_text2img_ftt.text2img_rows_all_layers's forward-pass and
    token-layout bookkeeping exactly (same prompt template, same 29871
    append, same num_patches derivation), but returns the full [L, T, T]
    stack plus the row/column index bookkeeping needed for both new
    statistics, instead of pre-slicing to text-rows x image-cols.
    """
    img = preprocess_like_policy(image, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids,
             torch.full((input_ids.shape[0], 1), 29871,
                        dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        attention_mask = torch.cat(
            (attention_mask,
             torch.ones((attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=input_ids, attention_mask=attention_mask,
                  pixel_values=inputs["pixel_values"], output_attentions=True,
                  return_dict=True)
    if out.attentions is None:
        raise RuntimeError(
            "model returned no attentions -- load via load_vla_for_attention "
            "(SDPA).")

    A_stack = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, T, T]
    n_txt = input_ids.shape[1] - 1
    T = A_stack.shape[-1]
    num_patches = T - n_txt - 1
    assert num_patches > 0, f"bad token layout: T={T} n_txt={n_txt}"

    img_cols = list(range(1, 1 + num_patches))
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    txt_rows = [1 + num_patches + r for r in txt_rel]

    A_np = A_stack.cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return A_np, num_patches, img_cols, txt_rows


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
    n_llm_layers = vla.language_model.config.num_hidden_layers
    print(f"[*] {n_llm_layers} LLM layers; capturing img2img FTT at every layer, "
          f"merged desc+image FTT at the last layer only")

    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in vla.norm_stats, (
        f"unnorm key {cfg.unnorm_key} not in norm_stats: {list(vla.norm_stats)[:5]}")

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    _raw = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    ckpt_tag = f"{_raw[:40]}_{hashlib.md5(str(args.checkpoint).encode()).hexdigest()[:8]}"
    out_dir = Path(args.out_dir)

    cond_offset = ({"clean": 0, "trigger": 0} if args.eval_design == "paired"
                   else {"clean": 0, "trigger": args.n_seeds})

    # Confound check: assert clean and poison BDDL carry the same task
    # description for every task_id in this suite, before running anything --
    # fail loudly rather than silently scoring a confounded merged statistic.
    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        bddl_name = Path(task.bddl_file).name
        clean_path = Path(CLEAN_BDDL) / args.task_suite_name / bddl_name
        poison_path = Path(POISON_BDDL) / args.task_suite_name / bddl_name

        def _lang(p):
            for line in Path(p).read_text().splitlines():
                if "(:language" in line:
                    return line.strip()
            return None

        cl, po = _lang(clean_path), _lang(poison_path)
        assert cl == po, (
            f"task {task_id} ({bddl_name}): clean language {cl!r} != poison "
            f"language {po!r} -- the merged FTT statistic assumes identical "
            "prompts across conditions; this suite breaks that assumption.")
    print("[*] confound check passed: clean/poison BDDL language strings match "
          "for every task in this suite -- GoBA's trigger is purely physical/"
          "visual, prompt length is identical across conditions.")

    for cond, bddl_dir in (("clean", CLEAN_BDDL), ("trigger", POISON_BDDL)):
        trig = cond == "trigger"
        print(f"[*] === {cond} scenes (bddl={bddl_dir}) ===")
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

                    observation, img = build_observation(obs, resize_size)
                    A_np, num_patches, img_cols, txt_rows = full_forward_all_layers(
                        vla, processor, img, desc, center_crop=cfg.center_crop)
                    n_layers = A_np.shape[0]

                    # -- existing desc_only text2img (unchanged convention) --
                    text2img_layers = A_np[:, txt_rows][:, :, img_cols]  # [L, n_desc, n_patches]
                    text2img_last = text2img_layers[-1]

                    # -- 1. img2img FTT, every layer --
                    img2img_ftt_per_layer = []
                    for l in range(n_layers):
                        M = A_np[l][np.ix_(img_cols, img_cols)]
                        img2img_ftt_per_layer.append(ftt_score(M))

                    # -- 2. merged desc+image FTT, last layer only --
                    key_cols = sorted(set(img_cols) | set(txt_rows))
                    A_last = A_np[-1]
                    image_rows_matrix = A_last[np.ix_(img_cols, key_cols)]
                    text_rows_matrix = A_last[np.ix_(txt_rows, key_cols)]
                    merged_text_group_ftt = ftt_score(text_rows_matrix)
                    merged_image_group_ftt = ftt_score(image_rows_matrix)
                    merged_combined_ftt = 0.5 * (merged_text_group_ftt + merged_image_group_ftt)

                    ep_idx = cond_offset[cond] + ep
                    ExtractedSample(
                        attn_text_image=text2img_last,
                        attn_text_image_layers=text2img_layers,
                        label=int(trig),
                        attack="goba",
                        checkpoint=args.checkpoint,
                        trigger_type="physical_toxic_box" if trig else "none",
                        task_id=task_id, seed=ep_idx,
                        layer=-1,
                        layers_averaged=None,
                        n_cameras=1, patches_per_camera=num_patches,
                        episode_id=(f"{args.task_suite_name}__t{task_id}"
                                    f"__seed{args.seed}__s{ep_idx}__{cond}"),
                        frame_idx=0,
                        extra={"role": args.role,
                               "task_suite_name": args.task_suite_name,
                               "eval_design": args.eval_design,
                               "text_scope": "desc_only",
                               "task_description": desc,
                               "n_query_tokens": int(text2img_last.shape[0]),
                               "n_query_tokens_desc": int(text2img_last.shape[0]),
                               "n_layers": int(n_layers),
                               "bddl_dir": bddl_dir,
                               "env_seed": args.seed,
                               "reset_index": ep_idx,
                               "img2img_ftt_per_layer": [float(x) for x in img2img_ftt_per_layer],
                               "merged_text_group_ftt": float(merged_text_group_ftt),
                               "merged_image_group_ftt": float(merged_image_group_ftt),
                               "merged_combined_ftt": float(merged_combined_ftt)},
                    ).save(str(out_dir / f"{ckpt_tag}__{args.task_suite_name}"
                                         f"__t{task_id}__seed{args.seed}__s{ep_idx}"
                                         f"__{cond}.npz"))
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} "
                          f"img2img_last={img2img_ftt_per_layer[-1]:.5f} "
                          f"merged_combined={merged_combined_ftt:.5f}")
            finally:
                env.close()

    del vla, processor
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
