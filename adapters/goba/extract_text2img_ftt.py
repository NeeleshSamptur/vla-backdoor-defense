#!/usr/bin/env python
"""GoBA extractor: text->image attention rows for the FTT detector.

Run this INSIDE GoBA_attack's own conda env (GoBA-OpenVLA), with GoBA_attack on
PYTHONPATH, ideally run from within the GoBA_attack repo directory (see usage
below). Never imported by anything in detectors/ or runners/ -- this script is
the only place GoBA's code is touched.

Model loading reuses GoBA's own experiments.robot.openvla_utils.get_vla /
get_processor UNCHANGED (handles dataset_statistics.json / norm_stats itself).

GoBA's backdoor is a PHYSICAL OBJECT placed in the LIBERO scene by a separate
BDDL file (bddl_files-poison_eval), not a pixel trigger -- so pairing requires
rendering both BDDL variants, not pasting a patch.

CRITICAL RENDERING NOTE (found the hard way building the bera prototype this
repo supersedes the exploratory parts of): holding TWO `OffScreenRenderEnv`
instances open at once corrupts the render of the FIRST one -- its EGL context
loses lighting and the frame comes back dark. Measured: a genuine 1.1%
pixel-level trigger difference becomes a spurious ~79% whole-frame difference
if both envs are alive together. Rendering is deterministic given the seed
(0.00% difference across repeat renders), so this script ALWAYS collects every
clean scene first, closes that env, then collects every poison scene --
never both envs alive simultaneously. Do not "optimize" this into a single
interleaved loop.

Token layout (GoBA / base OpenVLA, confirmed from PrismaticForConditionalGeneration.forward):
    multimodal_embeddings = cat([ input_embeddings[:, :1, :],   # BOS-like token
                                   projected_patch_embeddings,   # image patches
                                   input_embeddings[:, 1:, :] ]) # rest of the prompt
so image columns = range(1, 1+num_patches), text rows = range(1+num_patches, seq_len).

Usage:
    conda activate GoBA-OpenVLA
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack
    python /home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/adapters/goba/extract_text2img_ftt.py \
        --checkpoint exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
        --role attack --out-dir ../vla-backdoor-defense/results/goba_extracted

    # clean baseline, for the negative control:
    python .../extract_text2img_ftt.py --checkpoint openvla/openvla-7b-finetuned-libero-goal \
        --role clean_baseline --out-dir ../vla-backdoor-defense/results/goba_extracted
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, DEFENSE_REPO)

import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.openvla_utils import get_processor, get_vla
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

from detectors.schema import ExtractedSample  # from the new repo, added to sys.path above

DEVICE = "cuda:0"
SUITE = "libero_goal"
NUM_STEPS_WAIT = 10


@dataclass
class Cfg:
    pretrained_checkpoint: str
    model_family: str = "openvla"
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    unnorm_key: str = SUITE


def collect_scenes(suite, n_tasks, bddl_dir, episodes_per_task, resolution, seed, tag):
    """Render every scene for ONE bddl variant with only one env alive at a time.

    See the CRITICAL RENDERING NOTE in this file's module docstring -- this
    function must never run concurrently with another live env.
    """
    imgs, task_of, descs = [], [], []
    for tid in range(n_tasks):
        task = suite.get_task(tid)
        env, desc = get_libero_env(task, "openvla", resolution=resolution,
                                   bddl_path=str(bddl_dir), seed=seed)
        try:
            for ep in range(episodes_per_task):
                ep_seed = seed + 1000 * tid + ep
                env.seed(ep_seed)
                env.reset()
                obs = None
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                imgs.append(get_libero_image(obs, resolution))
                task_of.append(tid)
                descs.append(desc)
        finally:
            env.close()
        print(f"    [{tag}] task {tid+1}/{n_tasks} ({desc[:38]}): {len(imgs)} scenes")
    return imgs, task_of, descs


def text2img_rows(vla, processor, cfg, image, desc, resize_size):
    """Text-token attention rows over image-patch columns, one forward pass.

    Uses the top-level model forward directly (output_attentions=True) --
    GoBA's base OpenVLA has no OFT-specific internal embedding-assembly
    methods, so there is no manual embedding construction to replicate; the
    model's own forward() already does it (see module docstring).
    """
    from PIL import Image

    img = Image.fromarray(image).convert("RGB")
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                  pixel_values=inputs["pixel_values"], output_attentions=True, return_dict=True)

    num_patches = vla.vision_backbone.get_num_patches()
    n_txt = inputs["input_ids"].shape[1] - 1
    img_cols = list(range(1, 1 + num_patches))
    txt_rows = list(range(1 + num_patches, 1 + num_patches + n_txt))
    A_last = out.attentions[-1][0].float().mean(0)
    rows = A_last[txt_rows][:, img_cols].cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return rows, num_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--role", required=True, choices=["attack", "clean_baseline"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--episodes-per-task", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--clean-bddl", default="BadLIBERO/libero/libero/bddl_files")
    ap.add_argument("--poison-bddl", default="BadLIBERO/libero/libero/bddl_files-poison_eval")
    ap.add_argument("--layer", type=int, default=-1)
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = Cfg(pretrained_checkpoint=args.checkpoint)
    print(f"[*] loading {args.checkpoint} (role={args.role})")
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[SUITE]()
    n_tasks = min(args.n_tasks, suite.n_tasks)

    print("[*] rendering clean scenes (one env alive at a time)...")
    clean_imgs, clean_task_of, clean_descs = collect_scenes(
        suite, n_tasks, args.clean_bddl, args.episodes_per_task, args.resolution, args.seed, "clean")
    print("[*] rendering poison scenes (one env alive at a time)...")
    trig_imgs, trig_task_of, trig_descs = collect_scenes(
        suite, n_tasks, args.poison_bddl, args.episodes_per_task, args.resolution, args.seed, "poison")

    # Sanity check: a correct pairing differs only where the poison object is.
    ci = np.stack(clean_imgs); ti = np.stack(trig_imgs)
    d = np.abs(ci.astype(int) - ti.astype(int)).sum(-1)
    frac = float((d > 30).mean())
    print(f"[*] mean clean-vs-trigger pixel difference: {frac*100:.2f}%")
    if frac > 0.15:
        print("[!] WARNING: pairs differ over >15% of the frame -- likely the "
              "concurrent-env rendering artifact. Check the CRITICAL RENDERING "
              "NOTE in this file. Do not trust results downstream of this run.")

    ckpt_tag = Path(args.checkpoint).name if os.path.isdir(args.checkpoint) else args.checkpoint.replace("/", "_")
    out_dir = Path(args.out_dir)

    for label, imgs, task_of, descs, trig_flag in (
        ("clean", clean_imgs, clean_task_of, clean_descs, False),
        ("trigger", trig_imgs, trig_task_of, trig_descs, True),
    ):
        for i, (img, tid, desc) in enumerate(zip(imgs, task_of, descs)):
            rows, num_patches = text2img_rows(vla, processor, cfg, img, desc, resize_size)
            sample = ExtractedSample(
                attn_text_image=rows, label=int(trig_flag), attack="goba",
                checkpoint=args.checkpoint,
                trigger_type="physical_toxic_box" if trig_flag else "none",
                task_id=tid, seed=args.seed, layer=args.layer,
                n_cameras=1, patches_per_camera=num_patches,
                extra={"role": args.role},
            )
            fname = out_dir / f"{ckpt_tag}__t{tid}__ep{i}__{label}.npz"
            sample.save(str(fname))
        print(f"    saved {len(imgs)} {label} samples")

    print(f"[*] done -> {out_dir}")


if __name__ == "__main__":
    main()
