#!/usr/bin/env python
"""GoBA extractor: img2img FTT computed from the VISION ENCODER's own self-
attention (DinoSigLIPViTBackbone), not the LLM.

Everything analyzed so far (extract_text2img_ftt.py, extract_img2img_merged_
ftt.py) reads attention out of the LLM's causal self-attention over the fused
[BOS][image patches][text] sequence. This script instead asks whether
T2IShield's "assimilation" phenomenon already shows up INSIDE the vision
backbone itself -- before the LLM ever sees anything, with no text, no causal
mask (ViT self-attention is fully bidirectional), and with DINOv2 and SigLIP
as two architecturally-independent towers (channel-concatenated afterward,
never cross-attending to each other).

Model-loading, trigger/scene setup and episode selection are reused unchanged
from extract_text2img_ftt (imported below) -- only what's captured from the
forward pass differs.

## The two mechanical problems this script solves

1. **timm's fused attention exposes nothing to hook.** timm 0.9.10's
   `Attention.forward` (verified directly against the installed version, see
   module docstring of this comment) branches on `self.fused_attn`
   (`use_fused_attn()`, on by default when PyTorch's SDPA is available): when
   True it calls `F.scaled_dot_product_attention` and the post-softmax weights
   are never materialized as a tensor -- there's nothing to hook. When False
   it takes the manual path (`q @ k.T`, `.softmax(-1)`, `self.attn_drop(attn)`,
   `attn @ v`) and DOES materialize the weights, feeding them straight into
   `attn_drop`. So every block's `attn.fused_attn` is forced to False once,
   after model load, and a forward hook is registered on `attn.attn_drop`
   that reads its OWN INPUT (the tensor passed into forward, i.e. the raw
   post-softmax attention, pre-dropout) -- capturing it as a side effect
   without altering the module's return value or the model's normal forward
   pass at all. In eval() mode attn_drop.p is inert anyway, so forcing the
   manual path changes nothing about outputs, only what becomes visible.

2. **The `get_intermediate_layers` monkey-patch (dinosiglip_vit.py) looked
   like it would truncate execution to `len(blocks) - 2` blocks.** Checked
   directly against the installed timm's `VisionTransformer._intermediate_
   layers` source: `n` only controls which blocks' OUTPUTS are collected into
   the returned tuple -- the `for i, blk in enumerate(self.blocks): x =
   blk(x)` loop runs every block regardless of `n`, every time. So hooks on
   every block's `attn.attn_drop` all fire, and attention is captured for
   EVERY layer of both towers (24 for DINOv2, 27 for SigLIP; verified via
   `timm.create_model(...).blocks` before running the real extraction) --
   not just the second-to-last, despite what the monkey-patch's *output*
   truncates to.

Both towers are reached via the HF inference-time class actually used here
(`prismatic.extern.hf.modeling_prismatic.PrismaticVisionBackbone`, NOT the
training-time `prismatic.models.backbones.vision.dinosiglip_vit.
DinoSigLIPViTBackbone` that this script's designers first pointed at -- the
two are different classes with different attribute names). Its attributes are
`vla.vision_backbone.featurizer` (dino/"alpha", first in
`VISION_BACKBONE_TO_TIMM_ID["dinosiglip-vit-so-224px"]`) and
`vla.vision_backbone.fused_featurizer` (siglip/"beta", second) -- confirmed by
reading `configuration_prismatic.py` and `modeling_prismatic.py` directly, not
assumed from the training-time module's naming.

## What gets scored

DINOv2 (`vit_large_patch14_reg4_dinov2.lvd142m`) has 5 prefix tokens (1 CLS +
4 register tokens per `reg4`) prepended to its 256 patch tokens; SigLIP
(`vit_so400m_patch14_siglip_224`) has 0 prefix tokens (256 patch tokens,
nothing else). For each captured layer of each tower, the prefix rows/columns
are dropped and img2img FTT (patch attends to patch, head-averaged,
row-normalized) is computed via `detectors.ftt.ftt_score` UNCHANGED -- same
function the LLM-level img2img script uses, applied to a differently-sourced
square attention matrix. This mirrors the LLM-level img2img convention (image
rows/cols only, BOS dropped) rather than reinventing a new statistic.

## Storage: deliberately NOT detectors/schema.py's ExtractedSample

ExtractedSample is shaped for one text-to-image attention map (plus an
optional wrist-camera companion) from one chosen LLM layer or layer-stack.
Here there is no text at all, and two independent towers with different
block counts (24 vs 27) and different prefix-token counts (5 vs 0) need their
own full per-layer score vectors saved side by side. Bending the schema to fit
would mean stuffing tower identity into a fake "camera" slot or field-hacking
`extra` for what is actually the primary payload -- forcing a fit that isn't
natural. Instead each episode is saved as a small self-contained .npz:
`dino_ftt_per_layer` (float32, len = n_dino_layers captured),
`siglip_ftt_per_layer` (float32, len = n_siglip_layers captured), plus a
`meta_json` blob (role, label, task_id, episode_id, condition, checkpoint,
task_suite_name, seed, n_dino_layers, n_siglip_layers, dino_num_patches,
siglip_num_patches) -- read back and scored by
runners/score_vision_tower_attn.py using detectors.ftt.auroc, unchanged.

Usage (same env as extract_img2img_merged_ftt.py):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../adapters/goba/extract_vision_tower_attn.py \
        --checkpoint exp/openvla-7b+libero_object_no_noops+... \
        --task-suite-name libero_object --role attack \
        --out-dir ../vla-backdoor-defense/results/goba_vision_tower_attn
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from extract_text2img_ftt import (  # noqa: E402 -- reuse, do not reimplement
    CLEAN_BDDL, POISON_BDDL, VALID_SUITES, Cfg,
    build_observation, load_vla_for_attention, preprocess_like_policy,
)

DEVICE = "cuda:0"
NUM_STEPS_WAIT = 10


class TowerAttnCapture:
    """Forces every block's timm Attention to the manual (non-fused) softmax
    path and hooks attn_drop to stash post-softmax, head-averaged attention
    per layer as a side effect. Registered once per tower after model load;
    does not alter the model's forward pass or its outputs.
    """

    def __init__(self, tower: torch.nn.Module, name: str):
        self.name = name
        self.store: dict[int, np.ndarray] = {}
        self.handles = []
        n_forced = 0
        for i, block in enumerate(tower.blocks):
            if block.attn.fused_attn:
                block.attn.fused_attn = False
                n_forced += 1
            h = block.attn.attn_drop.register_forward_hook(self._make_hook(i))
            self.handles.append(h)
        print(f"    [{name}] forced fused_attn=False on {n_forced}/{len(tower.blocks)} "
              f"blocks; hooked attn_drop on all {len(tower.blocks)} blocks")

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # inputs[0]: post-softmax attention, pre-dropout, [B, heads, N, N]
            attn = inputs[0].detach()
            self.store[layer_idx] = attn.float().mean(dim=1)[0].cpu().numpy()
        return hook

    def reset(self):
        self.store.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


def img2img_ftt_per_layer(store: dict[int, np.ndarray], num_prefix_tokens: int) -> list[float]:
    """One ftt_score per captured layer, patch-rows x patch-cols only
    (prefix tokens -- CLS / DINOv2 registers -- dropped from both axes,
    matching the LLM-level img2img convention of scoring image patches only).
    """
    scores = []
    for layer_idx in sorted(store.keys()):
        A = store[layer_idx]
        patch_A = A[num_prefix_tokens:, num_prefix_tokens:]
        scores.append(ftt_score(patch_A))
    return scores


def one_forward(vla, processor, image, desc, center_crop=True):
    """Same prompt/pixel_values construction as the LLM-level extractors, but
    output_attentions is left at its default (False) -- we only need the
    vision-tower hooks to fire, which happens during the model's normal
    forward pass regardless of whether LLM attentions are also requested.
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
        vla(input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"], output_attentions=False,
            return_dict=True)


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

    vb = vla.vision_backbone
    assert hasattr(vb, "featurizer") and hasattr(vb, "fused_featurizer"), (
        f"expected vla.vision_backbone.{{featurizer,fused_featurizer}}, got "
        f"attributes: {[a for a in dir(vb) if not a.startswith('_')]}")
    dino_tower, siglip_tower = vb.featurizer, vb.fused_featurizer
    dino_prefix = dino_tower.num_prefix_tokens
    siglip_prefix = siglip_tower.num_prefix_tokens
    print(f"[*] dino tower: {len(dino_tower.blocks)} blocks, {dino_prefix} prefix tokens, "
          f"{dino_tower.patch_embed.num_patches} patches")
    print(f"[*] siglip tower: {len(siglip_tower.blocks)} blocks, {siglip_prefix} prefix tokens, "
          f"{siglip_tower.patch_embed.num_patches} patches")

    dino_cap = TowerAttnCapture(dino_tower, "dino")
    siglip_cap = TowerAttnCapture(siglip_tower, "siglip")

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

    n_dino_layers_seen = None
    n_siglip_layers_seen = None

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

                    dino_cap.reset()
                    siglip_cap.reset()
                    one_forward(vla, processor, img, desc, center_crop=cfg.center_crop)

                    dino_scores = img2img_ftt_per_layer(dino_cap.store, dino_prefix)
                    siglip_scores = img2img_ftt_per_layer(siglip_cap.store, siglip_prefix)
                    n_dino_layers_seen = len(dino_scores)
                    n_siglip_layers_seen = len(siglip_scores)

                    ep_idx = cond_offset[cond] + ep
                    episode_id = (f"{args.task_suite_name}__t{task_id}"
                                   f"__seed{args.seed}__s{ep_idx}__{cond}")
                    meta = dict(
                        role=args.role, attack="goba", checkpoint=args.checkpoint,
                        trigger_type="physical_toxic_box" if trig else "none",
                        task_suite_name=args.task_suite_name, task_id=task_id,
                        seed=ep_idx, label=int(trig), episode_id=episode_id,
                        eval_design=args.eval_design, bddl_dir=bddl_dir,
                        task_description=desc,
                        n_dino_layers=n_dino_layers_seen, n_siglip_layers=n_siglip_layers_seen,
                        dino_num_patches=dino_tower.patch_embed.num_patches,
                        siglip_num_patches=siglip_tower.patch_embed.num_patches,
                        dino_prefix_tokens=dino_prefix, siglip_prefix_tokens=siglip_prefix,
                    )
                    np.savez_compressed(
                        out_dir / f"{ckpt_tag}__{args.task_suite_name}__t{task_id}"
                                  f"__seed{args.seed}__s{ep_idx}__{cond}.npz",
                        dino_ftt_per_layer=np.array(dino_scores, dtype=np.float32),
                        siglip_ftt_per_layer=np.array(siglip_scores, dtype=np.float32),
                        meta_json=json.dumps(meta),
                    )
                    print(f"    task={task_id} ep={ep_idx} {cond:8s} "
                          f"dino[0]={dino_scores[0]:.4f} dino[-1]={dino_scores[-1]:.4f} "
                          f"siglip[0]={siglip_scores[0]:.4f} siglip[-1]={siglip_scores[-1]:.4f}")
            finally:
                env.close()

    dino_cap.remove()
    siglip_cap.remove()
    del vla, processor
    torch.cuda.empty_cache()
    print(f"[*] done -> {out_dir}")
    print(f"[*] captured {n_dino_layers_seen}/{len(dino_tower.blocks)} dino layers, "
          f"{n_siglip_layers_seen}/{len(siglip_tower.blocks)} siglip layers")


if __name__ == "__main__":
    main()
