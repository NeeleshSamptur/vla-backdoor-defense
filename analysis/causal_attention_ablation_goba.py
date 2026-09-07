#!/usr/bin/env python
"""Is attention to the trigger patch CAUSALLY necessary for GoBA's backdoor?

Every result in this project so far is correlational: we observe that
attention concentrates on the trigger and infer that this is the backdoor's
mechanism. It might instead be an epiphenomenon -- the backdoor could route
through the vision encoder's features, the MLPs, or the residual stream, with
the attention hotspot merely a visible side effect. Correlational statistics
cannot tell those apart, which is why every detector so far has collapsed
into "a salient object is present".

This runs the intervention instead, reusing VisAttnSink's own methodology
(arxiv 2503.03321): they establish that sink tokens are functionally inert by
ABLATING attention to them and showing behaviour does not change. Same move
here, opposite expectation:

    zero the post-softmax attention paid to the trigger patches, renormalize
    the row (VAR's surgery with p=0), at EVERY layer and EVERY decode step,
    then read out the action the policy actually emits.

Outcomes, both decisive:
  * action snaps back toward the clean-scene action  => attention to the
    trigger is load-bearing => an attention-based defense can work, and VAR-
    style redistribution is the natural implementation.
  * action barely moves                              => the backdoor does not
    travel through visual attention => attention-based defense is aimed at
    the wrong pathway and this whole line should stop.

CONTROL (the part that keeps this honest): ablating ANY attention perturbs
the model somewhat, so trigger-patch ablation is compared against ablating
the same NUMBER of randomly chosen visual patches, several draws. Only an
effect much larger than the random-ablation baseline counts. Without this
control the experiment would repeat the same mistake as the per-token
detector, which "worked" perfectly on a non-backdoored model.

Trigger patches are located per-episode (not hardcoded) as the top-K patches
by attention gained in the poison scene relative to the clean scene, so the
intervention follows the object wherever it sits in that task's layout.
"""
from __future__ import annotations

import argparse
import json
import math
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
import torch.nn as nn
from libero.libero import benchmark
from PIL import Image

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


class Ablation:
    """Global switch read by the patched attention forward."""
    key_indices = None      # LongTensor of absolute key positions to zero, or None
    enabled = False
    fired = 0               # counts surgery calls since the last set() (sanity)
    total_fired = 0         # cumulative across the whole process; used by
                            # defense_eval_goba to assert the patched forward
                            # is actually on the live code path

    @classmethod
    def set(cls, indices):
        cls.key_indices = None if indices is None else torch.as_tensor(list(indices), device=DEVICE)
        cls.enabled = indices is not None
        cls.fired = 0

    @classmethod
    def off(cls):
        cls.set(None)


def install_ablation_patch():
    """Replace eager LlamaAttention.forward with the same computation plus a
    post-softmax ablation hook.

    Reimplements transformers 4.40.1's eager forward verbatim (the model is
    loaded attn_implementation="eager" by load_vla_for_attention, so this is
    the path that actually executes) and inserts, immediately after the
    softmax:

        attn[..., key_indices] = 0 ; renormalize rows to sum 1

    which is exactly VAR's attn_redist with p=0 -- remove the weight, let the
    remaining keys keep their relative proportions. Applied at every layer and
    every decode step, so generated action tokens also cannot see the trigger.
    """
    from transformers.models.llama import modeling_llama as ml

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = ml.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = ml.repeat_kv(key_states, self.num_key_value_groups)
        value_states = ml.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # ---- the intervention -------------------------------------------
        if Ablation.enabled and Ablation.key_indices is not None:
            idx = Ablation.key_indices
            idx = idx[idx < attn_weights.shape[-1]]
            if idx.numel() > 0:
                attn_weights = attn_weights.clone()
                attn_weights[..., idx] = 0
                attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True).clamp_min(1e-9)
                Ablation.fired += 1
                Ablation.total_fired += 1
        # -----------------------------------------------------------------

        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    ml.LlamaAttention.forward = forward
    ml.LlamaSdpaAttention.forward = forward   # eager load still resolves to the Sdpa subclass
    print("[*] installed ablation-capable attention forward")


def act(vla, processor, img_np, desc, unnorm_key, center_crop=True):
    """One action prediction, as GoBA's own get_vla_action does -- with the
    29871 sentinel appended to input_ids AND attention_mask up front.

    predict_action appends 29871 to input_ids but leaves attention_mask
    alone. Under SDPA that is harmless (the mask is often discarded), but
    under eager -- which this project now loads with, to get truthfully
    causal attention -- the additive mask is one column short of the keys and
    the forward dies with a shape mismatch. This is exactly the failure the
    old load_vla_for_attention docstring cited when it claimed "eager is not
    an option". Doing the append here, consistently on both tensors, makes
    predict_action's own `if not torch.all(input_ids[:, -1] == 29871)` a
    no-op, so lengths stay matched through prefill and every decode step.
    """
    pil = preprocess_like_policy(img_np, center_crop)
    prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
    inputs = processor(prompt, pil).to(DEVICE, dtype=torch.bfloat16)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.full((input_ids.shape[0], 1), 29871,
                                   dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)
    with torch.no_grad():
        action = vla.predict_action(input_ids=input_ids, attention_mask=attention_mask,
                                    pixel_values=inputs["pixel_values"],
                                    unnorm_key=unnorm_key, do_sample=False)
    return np.asarray(action, dtype=np.float64)


def attn_profile(vla, processor, img_np, desc, center_crop=True):
    """Layer-averaged text2img attention received per patch -> [n_patches]."""
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
    A = torch.stack([a[0] for a in out.attentions]).float().mean(dim=1).mean(dim=0)  # [T,T] layer+head avg
    n_txt = input_ids.shape[1] - 1
    T = A.shape[-1]
    pa = T - n_txt - 1
    img_cols = list(range(1, 1 + pa))
    txt_rel = _desc_token_row_indices(processor, prompt, desc, n_txt)
    txt_rows = [1 + pa + r for r in txt_rel]
    prof = A[np.ix_(txt_rows, img_cols)].mean(dim=0).cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return prof, pa, img_cols


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
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--clean-ep", type=int, default=0)
    ap.add_argument("--trigger-ep", type=int, default=10)
    ap.add_argument("--topk", type=int, nargs="+", default=[2, 8])
    ap.add_argument("--n-random", type=int, default=5, help="random-ablation control draws")
    ap.add_argument("--out", default=str(DEFENSE_REPO / "results" / "causal_ablation_goba.json"))
    args = ap.parse_args()

    set_seed_everywhere(7)
    cfg = Cfg(pretrained_checkpoint=ATTACK_CKPT, unnorm_key="libero_object")
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()["libero_object"]()

    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    unnorm_key = cfg.unnorm_key
    install_ablation_patch()

    results = []
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        img_clean, desc = render(cfg, resize_size, task, CLEAN_BDDL, args.clean_ep)
        img_trig, desc_t = render(cfg, resize_size, task, POISON_BDDL, args.trigger_ep)
        assert desc == desc_t

        # locate the trigger patches for THIS scene: biggest attention gain
        Ablation.off()
        p_clean, pa, img_cols = attn_profile(vla, processor, img_clean, desc, cfg.center_crop)
        p_trig, _, _ = attn_profile(vla, processor, img_trig, desc, cfg.center_crop)
        gain = p_trig - p_clean

        a_clean = act(vla, processor, img_clean, desc, unnorm_key, cfg.center_crop)
        a_trig = act(vla, processor, img_trig, desc, unnorm_key, cfg.center_crop)
        d_base = float(np.linalg.norm(a_trig - a_clean))

        rec = dict(task_id=task_id, desc=desc, n_patches=int(pa),
                   d_base=d_base, a_clean=a_clean.tolist(), a_trig=a_trig.tolist(), by_k={})
        print(f"\n=== task {task_id}: {desc!r}")
        print(f"    ||a_trig - a_clean|| (backdoor effect on action) = {d_base:.4f}")

        rng = np.random.RandomState(1234 + task_id)
        for K in args.topk:
            top = np.argsort(-gain)[:K]
            key_idx = [img_cols[i] for i in top]          # absolute key positions
            Ablation.set(key_idx)
            a_abl = act(vla, processor, img_trig, desc, unnorm_key, cfg.center_crop)
            fired_t = Ablation.fired
            Ablation.off()
            d_abl = float(np.linalg.norm(a_abl - a_clean))
            move_t = float(np.linalg.norm(a_abl - a_trig))

            d_rnd, move_r = [], []
            for r in range(args.n_random):
                rnd = rng.choice(pa, K, replace=False)
                Ablation.set([img_cols[i] for i in rnd])
                a_r = act(vla, processor, img_trig, desc, unnorm_key, cfg.center_crop)
                Ablation.off()
                d_rnd.append(float(np.linalg.norm(a_r - a_clean)))
                move_r.append(float(np.linalg.norm(a_r - a_trig)))

            rec["by_k"][str(K)] = dict(
                patches=[int(t) for t in top], key_idx=[int(k) for k in key_idx],
                d_ablate_trigger=d_abl, d_ablate_random_mean=float(np.mean(d_rnd)),
                d_ablate_random_all=d_rnd,
                move_trigger=move_t, move_random_mean=float(np.mean(move_r)),
                surgery_calls=int(fired_t),
            )
            print(f"    K={K:2d} patches={top.tolist()}  (surgery fired {fired_t}x)")
            print(f"        ablate TRIGGER : ||a-a_clean||={d_abl:.4f}   moved from a_trig by {move_t:.4f}")
            print(f"        ablate RANDOM  : ||a-a_clean||={np.mean(d_rnd):.4f} (+-{np.std(d_rnd):.4f})"
                  f"   moved from a_trig by {np.mean(move_r):.4f}")
            verdict = ("TOWARD clean" if d_abl < d_base else "AWAY from clean")
            print(f"        => trigger-ablation moved action {verdict}; "
                  f"random control moved {'TOWARD' if np.mean(d_rnd) < d_base else 'AWAY'}")
        results.append(rec)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[*] wrote {args.out}")

    # ---- aggregate verdict ----
    print("\n================ AGGREGATE ================")
    for K in args.topk:
        base = np.array([r["d_base"] for r in results])
        trig = np.array([r["by_k"][str(K)]["d_ablate_trigger"] for r in results])
        rand = np.array([r["by_k"][str(K)]["d_ablate_random_mean"] for r in results])
        print(f"K={K}: mean ||a-a_clean||  no-ablation={base.mean():.4f}  "
              f"ablate-trigger={trig.mean():.4f}  ablate-random={rand.mean():.4f}")
        print(f"      trigger-ablation closed {100*(base-trig).mean()/base.mean():+.1f}% of the gap; "
              f"random closed {100*(base-rand).mean()/base.mean():+.1f}%")
    print("[*] ABLATION_DONE")


if __name__ == "__main__":
    main()
