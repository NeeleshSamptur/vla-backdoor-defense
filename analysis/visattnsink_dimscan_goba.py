#!/usr/bin/env python
"""Dim-discovery scan: does OpenVLA have its OWN visual massive-activation
dims, different from LLaVA's D_sink={2533,1415}?

visattnsink_probe_goba.py tested only LLaVA's two dims and found visual phi
capped at ~18 (below tau=20) with attention NEGATIVELY correlated to phi.
But that assumed the dims carry over. OpenVLA = llama2-7b-pure + a fused
DINOv2/SigLIP vision stack + LoRA robot fine-tuning; the visual token
representations are not LLaVA's, so OpenVLA could plausibly have massive
activation in DIFFERENT dimensions for visual tokens specifically.

This scans ALL hidden dims: for each layer, computes
    a[d] = max over VISUAL tokens of |h[tok,d]| / RMS(h[tok])
and reports the top dims. If some dim d* shows a[d*] >> tau while sitting
outside {2533,1415}, then OpenVLA has its own visual sink dims and the
earlier negative result was measuring the wrong coordinates. If the top dims
ARE 2533/1415 (or nothing crosses tau anywhere), the negative result stands.

Also reports, for the top discovered dims, corr(attention_received, phi_d)
-- the same test that was decisive before -- so a newly-found dim gets the
same scrutiny rather than being assumed meaningful.
"""
from __future__ import annotations

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
LLAVA_DIMS = [2533, 1415]
ATTACK_CKPT = "/home/grads/nsamptur/vla_bkd_def/GoBA_attack/exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"


def rms_normed_abs(h: torch.Tensor) -> torch.Tensor:
    """|h / RMS(h)| per token. h: [tokens, dim] -> [tokens, dim]."""
    h = h.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    return (h * torch.rsqrt(var + 1e-6)).abs()


def main():
    set_seed_everywhere(7)
    cfg = Cfg(pretrained_checkpoint=ATTACK_CKPT, unnorm_key="libero_object")
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(0)

    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    for cond, bddl in (("clean", CLEAN_BDDL), ("trigger", POISON_BDDL)):
        ep = 0 if cond == "clean" else 10
        env, desc = get_libero_env(task, cfg.model_family, resolution=256, bddl_path=bddl, seed=7)
        obs = None
        for _ in range(ep + 1):
            obs = env.reset()
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        _, img = build_observation(obs, resize_size)
        env.close()

        pil = preprocess_like_policy(img, cfg.center_crop)
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
                      pixel_values=inputs["pixel_values"], output_attentions=True,
                      output_hidden_states=True, return_dict=True)

        hidden = [h[0] for h in out.hidden_states]                # list len L+1 of [T, D]
        attn = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1)  # [L, T, T]
        n_txt = input_ids.shape[1] - 1
        T = attn.shape[-1]
        pa = T - n_txt - 1
        img_cols = list(range(1, 1 + pa))
        txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
        txt_rows = [1 + pa + r for r in txt_rel]

        print(f"\n===== cond={cond}  n_patches={pa}  dim={hidden[0].shape[-1]} =====")
        print(" layer | top-5 dims by max |h[d]|/RMS over VISUAL tokens (dim:value) | LLaVA dims value")
        worst = []
        for l in [2, 5, 10, 16, 20, 26, 31, 32]:
            hv = rms_normed_abs(hidden[l][img_cols])   # [n_patches, D]
            per_dim_max = hv.max(dim=0).values          # [D]
            top = torch.topk(per_dim_max, 5)
            tops = ", ".join(f"{int(i)}:{float(v):.1f}" for v, i in zip(top.values, top.indices))
            llava = ", ".join(f"{d}:{float(per_dim_max[d]):.1f}" for d in LLAVA_DIMS)
            print(f" {l:5d} | {tops} | {llava}")
            worst.append((l, top.indices.tolist(), top.values.tolist()))

        # For the single most-activated discovered dim overall, redo the decisive test.
        best_layer, best_dims, best_vals = max(worst, key=lambda x: x[2][0])
        d_star = best_dims[0]
        print(f"\n[*] strongest visual dim overall: dim {d_star} at layer {best_layer} "
              f"(value {best_vals[0]:.1f}; tau={TAU})")
        print("[*] decisive test on that dim -- corr(attention_received, phi_dstar) per layer:")
        for l in [2, 5, 10, 16, 20, 26, 31]:
            hv = rms_normed_abs(hidden[l][img_cols])
            phi_d = hv[:, d_star].cpu().numpy()
            recv_img = attn[l][np.ix_(img_cols, img_cols)].cpu().numpy().mean(axis=0)
            recv_txt = attn[l][np.ix_(txt_rows, img_cols)].cpu().numpy().mean(axis=0)
            c_img = np.corrcoef(recv_img, phi_d)[0, 1]
            c_txt = np.corrcoef(recv_txt, phi_d)[0, 1]
            print(f"    layer {l:2d}:  img2img corr={c_img:+.3f}   text2img corr={c_txt:+.3f}")

        del out, hidden, attn
        torch.cuda.empty_cache()

    print("\n[*] DIMSCAN_DONE")


if __name__ == "__main__":
    main()
