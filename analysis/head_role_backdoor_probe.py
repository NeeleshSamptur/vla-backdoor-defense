#!/usr/bin/env python
"""Do NON-vision heads attend to the backdoor trigger?

Idea (adapted from VisAttnSink's image-centric heads): a benign salient
object should pull attention in heads whose job is looking at the image. If
the trigger ALSO hijacks heads that normally ignore vision -- heads doing
text/positional work -- that is a signature an ordinary object would not
produce, and one the clean-baseline checkpoint should not produce either,
since it was never trained on the trigger. That makes it weights-
attributable rather than scene-attributable, which is exactly what every
previous detector in this project failed to be.

Why this has not been looked at before here: every FTT statistic in this
repo head-AVERAGES, which washes out per-head structure entirely.

HEAD ROLE CRITERION. The paper defines image-centric heads via the visual
non-sink ratio, which needs the bimodal sink/non-sink split that
visattnsink_dimscan_goba.py showed does not exist in OpenVLA (phi is
unimodal there). So this uses the paper's OTHER, sink-free condition
instead -- total visual attention mass per head (their `summation >= summ`,
summ=0.2) -- which is architecture-agnostic:

    visual_mass[l,h] = mean over desc-token rows of (sum of attention
                       landing on image-patch columns)

High visual_mass  => image-centric head (looks at the image).
Low  visual_mass  => non-vision head    (attends to text/BOS/position).

MEASUREMENT. Both checkpoints are run on PIXEL-IDENTICAL renders (scene
rendered once, replayed to both), so a poisoned-vs-clean difference cannot be
explained by "there is an object in the frame" -- both see the same object.
Reports, per head class:

    trig_attn[l,h]  = mean over desc rows of attention landing on the
                      trigger patches
    trig_share[l,h] = trig_attn / visual_mass  (of what this head sends to
                      the image, how much lands on the trigger)

The headline comparison is trig_attn in NON-VISION heads, poisoned vs clean
model, on the identical triggered frame.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

DEFENSE_REPO = Path("/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense")
sys.path.insert(0, str(DEFENSE_REPO))
sys.path.insert(0, str(DEFENSE_REPO / "adapters" / "goba"))

import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

from extract_text2img_ftt import (
    CLEAN_BDDL, POISON_BDDL, Cfg, load_vla_for_attention,
    build_observation, preprocess_like_policy, _desc_token_row_indices,
)

DEVICE = "cuda:0"
NUM_STEPS_WAIT = 10
ATTACK_CKPT = "/home/grads/nsamptur/vla_bkd_def/GoBA_attack/exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-object"


def per_head_stats(vla, processor, img_np, desc, trig_patches, center_crop=True):
    """Returns visual_mass[L,H], trig_attn[L,H] measured on desc-token rows."""
    pil = preprocess_like_policy(img_np, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, pil).to(DEVICE, dtype=torch.bfloat16)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat((input_ids, torch.full((1, 1), 29871, dtype=input_ids.dtype,
                                                     device=input_ids.device)), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype,
                                                               device=attention_mask.device)), dim=1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=input_ids, attention_mask=attention_mask,
                  pixel_values=inputs["pixel_values"], output_attentions=True, return_dict=True)

    n_txt = input_ids.shape[1] - 1
    T = out.attentions[0].shape[-1]
    pa = T - n_txt - 1
    img_cols = list(range(1, 1 + pa))
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    txt_rows = [1 + pa + r for r in txt_rel]
    trig_cols = [img_cols[p] for p in trig_patches if p < pa]

    L = len(out.attentions)
    H = out.attentions[0].shape[1]
    visual_mass = np.zeros((L, H), dtype=np.float64)
    trig_attn = np.zeros((L, H), dtype=np.float64)
    for l in range(L):
        A = out.attentions[l][0].float()                       # [H, T, T]
        rows = A[:, txt_rows, :]                               # [H, Q, T]
        visual_mass[l] = rows[:, :, img_cols].sum(-1).mean(-1).cpu().numpy()
        trig_attn[l] = rows[:, :, trig_cols].sum(-1).mean(-1).cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return visual_mass, trig_attn, pa, img_cols


def render(cfg, resize_size, task, bddl_dir, ep_idx, base_seed=7):
    env, desc = get_libero_env(task, cfg.model_family, resolution=256, bddl_path=bddl_dir, seed=base_seed)
    try:
        obs = None
        for _ in range(ep_idx + 1):
            obs = env.reset()
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        _, img = build_observation(obs, resize_size)
        return np.array(img), desc
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--clean-ep", type=int, default=0)
    ap.add_argument("--trigger-ep", type=int, default=10)
    ap.add_argument("--topk", type=int, default=8, help="how many patches count as 'the trigger'")
    ap.add_argument("--summ", type=float, default=0.2,
                     help="paper's visual-mass threshold for an image-centric head")
    ap.add_argument("--out", default=str(DEFENSE_REPO / "results" / "head_role_backdoor_probe.json"))
    args = ap.parse_args()

    set_seed_everywhere(7)
    cfg = Cfg(pretrained_checkpoint=ATTACK_CKPT, unnorm_key="libero_object")
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()["libero_object"]()

    # render every scene once, replay identical pixels to both checkpoints
    scenes = {}
    for t in args.tasks:
        task = suite.get_task(t)
        ic, desc = render(cfg, resize_size, task, CLEAN_BDDL, args.clean_ep)
        it, desc2 = render(cfg, resize_size, task, POISON_BDDL, args.trigger_ep)
        assert desc == desc2
        scenes[t] = dict(clean=ic, trig=it, desc=desc)
        print(f"[*] rendered task {t}: {desc!r}")

    results = {}
    for role, ckpt in (("attack", ATTACK_CKPT), ("clean_baseline", CLEAN_CKPT)):
        print(f"\n[*] ===== {role} =====")
        c = Cfg(pretrained_checkpoint=ckpt, unnorm_key="libero_object")
        vla = load_vla_for_attention(c)
        processor = get_processor(c)
        if c.unnorm_key not in vla.norm_stats and f"{c.unnorm_key}_no_noops" in vla.norm_stats:
            c.unnorm_key = f"{c.unnorm_key}_no_noops"
        per_task = {}
        for t in args.tasks:
            s = scenes[t]
            # locate trigger patches from the ATTACK model's own attention gain,
            # then use the SAME patch set for both models so the comparison is
            # about the models, not about differing patch choices.
            vm_c, _, pa, img_cols = per_head_stats(vla, processor, s["clean"], s["desc"], [], c.center_crop)
            # head-avg profile gain to pick patches (only needs a coarse locate)
            prof_c = vm_c  # placeholder; real locate below uses full profile
            trig_patches = scenes[t].setdefault("trig_patches", None)
            if trig_patches is None:
                # compute gain profile with this model (attack runs first)
                def profile(img):
                    pil = preprocess_like_policy(img, c.center_crop)
                    prompt = f"In: What action should the robot take to {s['desc'].lower()}?\nOut:"
                    inp = processor(prompt, pil).to(DEVICE, dtype=torch.bfloat16)
                    iid, am = inp["input_ids"], inp["attention_mask"]
                    if not torch.all(iid[:, -1] == 29871):
                        iid = torch.cat((iid, torch.full((1, 1), 29871, dtype=iid.dtype, device=iid.device)), dim=1)
                        am = torch.cat((am, torch.ones((1, 1), dtype=am.dtype, device=am.device)), dim=1)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        o = vla(input_ids=iid, attention_mask=am, pixel_values=inp["pixel_values"],
                                output_attentions=True, return_dict=True)
                    A = torch.stack([a[0] for a in o.attentions]).float().mean(1).mean(0)
                    nt = iid.shape[1] - 1
                    T_ = A.shape[-1]; pa_ = T_ - nt - 1
                    ic_ = list(range(1, 1 + pa_))
                    tr_ = _desc_token_row_indices(processor, prompt, s["desc"], nt)
                    trs = [1 + pa_ + r for r in tr_]
                    p = A[np.ix_(trs, ic_)].mean(0).cpu().numpy()
                    del o; torch.cuda.empty_cache()
                    return p
                gain = profile(s["trig"]) - profile(s["clean"])
                trig_patches = np.argsort(-gain)[:args.topk].tolist()
                scenes[t]["trig_patches"] = trig_patches
                print(f"    task {t}: trigger patches {trig_patches}")

            vm_clean, ta_clean, _, _ = per_head_stats(vla, processor, s["clean"], s["desc"],
                                                      trig_patches, c.center_crop)
            vm_trig, ta_trig, _, _ = per_head_stats(vla, processor, s["trig"], s["desc"],
                                                    trig_patches, c.center_crop)
            per_task[t] = dict(vm_clean=vm_clean, ta_clean=ta_clean,
                               vm_trig=vm_trig, ta_trig=ta_trig)
        results[role] = per_task
        del vla, processor
        torch.cuda.empty_cache()

    # ---------- analysis ----------
    print("\n================ HEAD-ROLE ANALYSIS ================")
    print(f"image-centric head := visual_mass >= {args.summ} (paper's summ), measured on the CLEAN")
    print("frame of the ATTACK model, so head roles are fixed independently of the trigger.\n")
    summary = {}
    for t in args.tasks:
        vm_ref = results["attack"][t]["vm_clean"]           # role assignment reference
        ich = vm_ref >= args.summ
        non = ~ich
        print(f"--- task {t}: {ich.sum()} image-centric heads, {non.sum()} non-vision heads "
              f"(of {ich.size})")
        row = {}
        for role in ("attack", "clean_baseline"):
            ta_t = results[role][t]["ta_trig"]
            ta_c = results[role][t]["ta_clean"]
            row[role] = dict(
                ich_trig=float(ta_t[ich].mean()), ich_clean=float(ta_c[ich].mean()),
                non_trig=float(ta_t[non].mean()), non_clean=float(ta_c[non].mean()),
            )
            print(f"    {role:15s} trigger-patch attention: "
                  f"image-centric {ta_t[ich].mean():.5f} (clean frame {ta_c[ich].mean():.5f}) | "
                  f"NON-vision {ta_t[non].mean():.5f} (clean frame {ta_c[non].mean():.5f})")
        gap_non = row["attack"]["non_trig"] - row["clean_baseline"]["non_trig"]
        gap_ich = row["attack"]["ich_trig"] - row["clean_baseline"]["ich_trig"]
        print(f"    >> poisoned-minus-clean on the SAME triggered frame: "
              f"NON-vision {gap_non:+.5f}   image-centric {gap_ich:+.5f}")
        row["gap_non_vision"] = gap_non
        row["gap_image_centric"] = gap_ich
        summary[str(t)] = row

    gaps_non = [summary[str(t)]["gap_non_vision"] for t in args.tasks]
    gaps_ich = [summary[str(t)]["gap_image_centric"] for t in args.tasks]
    print(f"\nMEAN poisoned-minus-clean trigger attention:  "
          f"NON-vision {np.mean(gaps_non):+.5f}   image-centric {np.mean(gaps_ich):+.5f}")
    print("(a NON-vision gap clearly above the image-centric gap is the signature "
          "the salient-object confound cannot produce)")

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n[*] wrote {args.out}")
    print("[*] HEADROLE_DONE")


if __name__ == "__main__":
    main()
