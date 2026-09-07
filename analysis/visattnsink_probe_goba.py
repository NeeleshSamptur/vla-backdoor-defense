#!/usr/bin/env python
"""Probe VisAttnSink's mechanism (arxiv 2503.03321) on GoBA.

Method transferred verbatim from the paper's own released code
(github.com/seilk/VisAttnSink, src/logic/logic.py::DimProspector):

    phi(token, layer) = max over d in D_sink of |h[d]| / RMS(h)
    sink token  <=>  phi > tau            (tau = 20)

D_SINK IS MODEL-SPECIFIC AND MUST BE DISCOVERED, NOT ASSUMED. GoBA's backbone
is `llama2-7b-pure`, so the first version of this script reused LLaVA's
published llama-v2-7b dims {2533, 1415} -- and found nothing (no visual sinks,
attention/phi correlation NEGATIVE), which looked like "the phenomenon does
not transfer to OpenVLA". That was wrong: it was measuring dead coordinates.
A full 4096-dim scan (visattnsink_dimscan_goba.py) shows OpenVLA's visual
massive activation lives in dims {1512, 1076} (phi up to 54.1 / 48.5, well
past tau), while 1415 is inert here (~2-3) and 2533 caps at ~18. With the
correct dims the attention-received/phi correlation is POSITIVE at every
layer (+0.10..+0.43), matching the relationship the paper describes. The
likely reason the dims moved: OpenVLA's visual tokens come from a fused
DINOv2+SigLIP stack rather than LLaVA's CLIP, plus LoRA robot fine-tuning.

WHY THIS IS A DIFFERENT TEST FROM LAST NIGHT'S FAILED ONE:
The per-patch attention-magnitude detector hit AUROC 1.000 on the poisoned
model AND 1.000 on the clean-baseline model, firing at the same patch both
times -- i.e. it was detecting "a salient object is present", not the
backdoor. phi is computed from HIDDEN STATES, not attention, and both models
are run here on PIXEL-IDENTICAL images (the observation is rendered once and
replayed to both), so any phi difference between the two models is
attributable to the weights alone -- the scene is held constant by
construction, which is what the earlier test could not do.

The specific thing worth looking for: the paper's core finding is that sink
tokens are SEMANTICALLY EMPTY ("irrelevant sink tokens do not impact model
performance despite receiving high attention weights"). A trigger object is
semantically meaningful. So a patch that is both a strong attention target
AND a sink is paradoxical -- it should not arise naturally -- and if the
poisoned model turns the trigger patch into a sink while the clean model
does not, that is direct mechanistic evidence the backdoor co-opts the sink
pathway to route attention.

Outputs a JSON + npz per (model, scene) with per-layer phi over the visual
span, so downstream scoring needs no GPU.
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
TAU = 20.0
# CORRECTED: LLaVA's published llama-v2-7b dims {2533,1415} are the WRONG
# coordinates for OpenVLA -- 1415 is inert here (phi ~2-3) and 2533 caps at
# ~18, which is why the first run of this probe found no visual sinks and a
# NEGATIVE attention/phi correlation. A full 4096-dim scan
# (visattnsink_dimscan_goba.py) finds OpenVLA's own visual massive-activation
# dims are {1512, 1076} (phi up to 54.1 / 48.5), and with those the
# attention-received/phi correlation is POSITIVE (+0.10..+0.43) at every
# layer -- i.e. the sink phenomenon DOES exist here, in different dims.
# Plausibly because OpenVLA's visual tokens come from a fused DINOv2+SigLIP
# stack (not LLaVA's CLIP) plus LoRA robot fine-tuning.
D_SINK = [1512, 1076]
D_SINK_LLAVA = [2533, 1415]    # kept for reference/comparison only
ATTACK_CKPT = "/home/grads/nsamptur/vla_bkd_def/GoBA_attack/exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
CLEAN_CKPT = "openvla/openvla-7b-finetuned-libero-object"


def phi_from_hidden(hs: torch.Tensor) -> torch.Tensor:
    """VisAttnSink's DimProspector, verbatim in intent.

    hs: [tokens, dim] for one layer. Returns [tokens] sink score phi.
    rmsnorm as in their code: h * rsqrt(mean(h^2) + eps), then |.|, then max
    over the fixed sink dims.
    """
    h = hs.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    rms_normed = (h * torch.rsqrt(variance + 1e-6)).abs()
    return rms_normed[:, D_SINK].max(dim=-1).values


def forward_capture(vla, processor, img, desc, center_crop=True):
    """One forward pass returning per-layer hidden states AND attention.

    Same prompt/token bookkeeping as extract_img2img_merged_ftt.
    full_forward_all_layers, plus output_hidden_states=True (never captured
    before in this project).
    """
    pil = preprocess_like_policy(img, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, pil).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.full((input_ids.shape[0], 1), 29871,
                                   dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=input_ids, attention_mask=attention_mask,
                  pixel_values=inputs["pixel_values"], output_attentions=True,
                  output_hidden_states=True, return_dict=True)

    # attentions: tuple(len=L) of [1, H, T, T]; keep per-head, head-avg later
    attn = torch.stack([a[0] for a in out.attentions])          # [L, H, T, T]
    hidden = torch.stack([h[0] for h in out.hidden_states])     # [L+1, T, D]

    n_txt = input_ids.shape[1] - 1
    T = attn.shape[-1]
    num_patches = T - n_txt - 1
    assert num_patches > 0, f"bad layout: T={T} n_txt={n_txt}"
    img_cols = list(range(1, 1 + num_patches))
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    txt_rows = [1 + num_patches + r for r in txt_rel]

    phi = torch.stack([phi_from_hidden(hidden[l]) for l in range(hidden.shape[0])])  # [L+1, T]

    res = dict(
        phi=phi.float().cpu().numpy(),
        attn_headavg=attn.float().mean(dim=1).cpu().numpy(),   # [L, T, T]
        attn_perhead_lastlayer=attn[-1].float().cpu().numpy(),  # [H, T, T]
        num_patches=int(num_patches), img_cols=img_cols, txt_rows=txt_rows,
        n_layers=int(attn.shape[0]),
    )
    del out, attn, hidden
    torch.cuda.empty_cache()
    return res


def render_scene(cfg, resize_size, task, bddl_dir, ep_idx, base_seed=7):
    """Render ONE observation and return the raw image, so the exact same
    pixels can be replayed to both checkpoints (scene held constant)."""
    env, desc = get_libero_env(task, cfg.model_family, resolution=256,
                               bddl_path=bddl_dir, seed=base_seed)
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
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--clean-ep", type=int, default=0)
    ap.add_argument("--trigger-ep", type=int, default=10)
    ap.add_argument("--out-dir", default=str(DEFENSE_REPO / "results" / "visattnsink_probe_goba"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed_everywhere(7)
    cfg = Cfg(pretrained_checkpoint=ATTACK_CKPT, unnorm_key="libero_object")
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(args.task_id)

    # --- render both scenes ONCE; identical pixels go to both models ---
    print("[*] rendering scenes (once, replayed to both checkpoints)")
    img_clean, desc = render_scene(cfg, resize_size, task, CLEAN_BDDL, args.clean_ep)
    img_trig, desc_t = render_scene(cfg, resize_size, task, POISON_BDDL, args.trigger_ep)
    assert desc == desc_t, f"prompt differs between conditions: {desc!r} vs {desc_t!r}"
    print(f"    desc={desc!r}  clean_img={img_clean.shape}  trig_img={img_trig.shape}")
    np.savez_compressed(out_dir / "scenes.npz", img_clean=img_clean, img_trig=img_trig,
                        desc=np.array(desc))

    for role, ckpt in (("attack", ATTACK_CKPT), ("clean_baseline", CLEAN_CKPT)):
        print(f"\n[*] ===== {role}: {ckpt} =====")
        c = Cfg(pretrained_checkpoint=ckpt, unnorm_key="libero_object")
        vla = load_vla_for_attention(c)
        processor = get_processor(c)
        if c.unnorm_key not in vla.norm_stats and f"{c.unnorm_key}_no_noops" in vla.norm_stats:
            c.unnorm_key = f"{c.unnorm_key}_no_noops"

        for cond, image in (("clean", img_clean), ("trigger", img_trig)):
            r = forward_capture(vla, processor, image, desc, center_crop=c.center_crop)
            im = 1
            pa = r["num_patches"]
            phi_vis = r["phi"][:, im:im + pa]            # [L+1, n_patches]
            n_sink_per_layer = (phi_vis > TAU).sum(axis=1)
            np.savez_compressed(
                out_dir / f"{role}__{cond}.npz",
                phi=r["phi"], phi_vis=phi_vis,
                attn_headavg=r["attn_headavg"].astype(np.float32),
                attn_perhead_lastlayer=r["attn_perhead_lastlayer"].astype(np.float32),
                num_patches=pa, img_cols=np.array(r["img_cols"]),
                txt_rows=np.array(r["txt_rows"]), n_layers=r["n_layers"],
            )
            print(f"    [{cond}] n_patches={pa} layers={phi_vis.shape[0]}  "
                  f"visual sinks/layer (phi>{TAU:.0f}): "
                  f"min={n_sink_per_layer.min()} max={n_sink_per_layer.max()} "
                  f"mean={n_sink_per_layer.mean():.1f}")
            peak_layer = int(np.argmax(n_sink_per_layer))
            sinks_at_peak = np.nonzero(phi_vis[peak_layer] > TAU)[0]
            print(f"        peak layer {peak_layer}: {len(sinks_at_peak)} sinks, "
                  f"patches {sinks_at_peak[:12].tolist()}{'...' if len(sinks_at_peak) > 12 else ''}")

        del vla, processor
        torch.cuda.empty_cache()

    print(f"\n[*] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
