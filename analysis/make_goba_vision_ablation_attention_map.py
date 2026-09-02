#!/usr/bin/env python
"""GoBA per-LAYER text->image attention under vision-backbone ablation.

OpenVLA's vision backbone concatenates DINOv2 and SigLIP patch features
channel-wise *before* the projector (prismatic.extern.hf.modeling_prismatic.
PrismaticVisionBackbone.forward: `torch.cat([patches, patches_fused], dim=2)`,
where `patches` = DINOv2, `patches_fused` = SigLIP -- confirmed via
configuration_prismatic.VISION_BACKBONE_TO_TIMM_ID["dinosiglip-vit-so-224px"]
= [dino_id, siglip_id], index 0 first). Because the projector mixes both
blocks densely, you cannot separate their contributions post-hoc from
attention weights alone -- the only way to causally attribute an attention
pattern to one backbone is to ablate it *before* the projector and re-run.

This script monkey-patches `vla.vision_backbone.forward` to zero one block
(DINO or SigLIP) at a time, then reuses the same text->image attention
extraction and per-layer plotting already validated in
make_goba_perlayer_attention_map.py, run three times per episode:
    full        -- both backbones intact (baseline)
    dino_only   -- SigLIP block zeroed
    siglip_only -- DINO block zeroed

Output: two PNGs, each plot_pertoken's existing CLEAN/TRIGGER two-row layout
repurposed as FULL/ABLATED (same frame, same query tokens -- only the vision
input to the projector differs), so any change in attention mass is
attributable to whichever backbone was removed.

Usage (GoBA-OpenVLA conda env):
    conda activate GoBA-OpenVLA
    export PYTHONPATH="/home/grads/nsamptur/vla_bkd_def/GoBA_attack:$PYTHONPATH"
    export MUJOCO_GL=egl
    cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack

    python .../analysis/make_goba_vision_ablation_attention_map.py \\
        --checkpoint exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \\
        --task-suite-name libero_object --task-id 9 --episode-seed 7 --base-seed 7 \\
        --scene clean \\
        --out-prefix ../vla-backdoor-defense/results/attention_maps_explainer/goba_ablation_t9
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "goba"))
sys.path.insert(0, str(DEFENSE / "analysis"))

from libero.libero import benchmark

import torch

from extract_text2img_ftt import (
    CLEAN_BDDL, POISON_BDDL, NUM_STEPS_WAIT,
    Cfg, build_observation, get_libero_env, get_libero_dummy_action,
    load_vla_for_attention,
)
from make_goba_perlayer_attention_map import text2img_rows_all_layers
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_image_resize_size
from plot_pertoken_attention import plot_pertoken

MODES = ("full", "dino_only", "siglip_only")


@contextmanager
def vision_ablation(vla, mode: str):
    """Temporarily zero DINO's or SigLIP's patch block before the projector.

    `vla.vision_backbone.forward` is confirmed (modeling_prismatic.py) to be:
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
        return torch.cat([patches, patches_fused], dim=2)
    with self.featurizer == DINOv2 and self.fused_featurizer == SigLIP (index
    0/1 in VISION_BACKBONE_TO_TIMM_ID["dinosiglip-vit-so-224px"]). Zeroing a
    block here -- rather than zeroing the input image -- removes exactly the
    projector-input channels for that backbone without perturbing the other
    backbone's own forward pass (a zeroed image would still be a valid, if
    degenerate, input to a ViT and would not equal "no contribution").
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    vb = vla.vision_backbone
    assert getattr(vb, "use_fused_vision_backbone", False), (
        "vision_backbone is not a fused DINO+SigLIP backbone -- ablation modes "
        "dino_only/siglip_only don't apply to this checkpoint."
    )
    orig_forward = vb.forward
    if mode != "full":
        def ablated_forward(pixel_values, _vb=vb, _mode=mode):
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches = _vb.featurizer(img)            # DINOv2
            patches_fused = _vb.fused_featurizer(img_fused)  # SigLIP
            if _mode == "dino_only":
                patches_fused = torch.zeros_like(patches_fused)
            elif _mode == "siglip_only":
                patches = torch.zeros_like(patches)
            return torch.cat([patches, patches_fused], dim=2)
        vb.forward = ablated_forward
    try:
        yield
    finally:
        vb.forward = orig_forward


def run_episode(vla, processor, cfg, resize_size, bddl_dir, task, base_seed, ep_idx, mode):
    env, desc = get_libero_env(task, cfg.model_family, resolution=256,
                                bddl_path=bddl_dir, seed=base_seed)
    try:
        obs = None
        for _ in range(ep_idx + 1):
            obs = env.reset()
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        observation, img = build_observation(obs, resize_size)
        with vision_ablation(vla, mode):
            rows, num_patches, display_img = text2img_rows_all_layers(
                vla, processor, img, desc, center_crop=cfg.center_crop)
        per_layer = rows.mean(axis=1)  # [n_layers, n_patches], mean over query tokens
        return dict(image=display_img, attn=per_layer, prompt_desc=desc, n_layers=rows.shape[0])
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", default="libero_object")
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--episode-seed", type=int, required=True, help="ep_idx, not an RNG seed")
    ap.add_argument("--base-seed", type=int, default=7)
    ap.add_argument("--scene", choices=["clean", "trigger"], default="clean")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>_dino_only.png and <prefix>_siglip_only.png")
    args = ap.parse_args()

    cfg = Cfg(pretrained_checkpoint=args.checkpoint, unnorm_key=args.task_suite_name)
    print(f"[*] loading {args.checkpoint}")
    vla = load_vla_for_attention(cfg)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    bddl_dir = CLEAN_BDDL if args.scene == "clean" else POISON_BDDL

    results = {}
    for mode in MODES:
        print(f"[*] running scene={args.scene} mode={mode}")
        results[mode] = run_episode(vla, processor, cfg, resize_size, bddl_dir, task,
                                     args.base_seed, args.episode_seed, mode)

    full = results["full"]
    layer_labels = [f"L{l}" for l in range(full["n_layers"])]

    for mode, tag in (("dino_only", "DINO-ONLY (SigLIP zeroed)"), ("siglip_only", "SigLIP-ONLY (DINO zeroed)")):
        ablated = results[mode]
        title = f"GoBA checkpoint ({args.checkpoint}), FULL vs {tag}"
        subtitle = f"scene: {args.scene}  |  task: {full['prompt_desc']!r}"
        out_path = f"{args.out_prefix}_{mode}.png"
        plot_pertoken(
            image_clean=full["image"], attn_primary_clean=full["attn"], tokens_clean=layer_labels,
            image_trigger=ablated["image"], attn_primary_trigger=ablated["attn"], tokens_trigger=layer_labels,
            title=title, subtitle=subtitle, out_path=out_path,
            row_label_clean="FULL", row_label_trigger=tag,
        )
        print(f"[*] wrote {out_path}")

    del vla, processor


if __name__ == "__main__":
    main()
