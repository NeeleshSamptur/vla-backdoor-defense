#!/usr/bin/env python
"""Render the pi0-FAST per-token attention explainer for one task, from the
CORRECT extraction only -- full real query token set, trigger-type-aware.

TI_4 (bi-modal: popcorn + "~*magic*~ " text) needs `results/pi0fast_extracted_desc_only`,
whose trigger rows legitimately include the 4 magic-prefix tokens.

I_4 (vision-only: popcorn, no text change) needs
`results/pi0fast_I4_extracted_novisualtext_FIXED` -- NOT
`results/pi0fast_I4_extracted_desc_only`, which was found (2026-08-31) to
actually contain TI_4-style bi-modal prompts mislabeled as I_4; using it here
would draw a magic-prefix column on a checkpoint whose attack never adds one.

Raw camera images come from the matching phase-1 observation .npz (the
extracted-sample .npz never stores images), matched by task_id/seed/condition.

Usage (pi0 venv, no GPU/model needed -- only real saved arrays + the tokenizer):
    python make_pi0_attention_maps.py --variant TI4 --task-id 0 --clean-seed 10 --trigger-seed 17
    python make_pi0_attention_maps.py --variant I4  --task-id 0 --clean-seed 10 --trigger-seed 17
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

DEFENSE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "adapters" / "pi0fast_backdoorvla"))

import numpy as np

from extract_text2img_ftt import desc_token_rows, load_paligemma_tokenizer
from plot_pertoken_attention import plot_pertoken

VARIANTS = {
    "TI4": dict(
        extracted_dir=DEFENSE / "results/pi0fast_extracted_desc_only",
        obs_dir=DEFENSE / "results/pi0fast_observations",
        trigger_type="bi-modal: popcorn container + '~*magic*~ ' text prefix",
        checkpoint_tag="pi0_fast_libero_object_TI_4",
    ),
    "I4": dict(
        extracted_dir=DEFENSE / "results/pi0fast_I4_extracted_novisualtext_FIXED",
        obs_dir=DEFENSE / "results/pi0fast_ablation/visiononly",
        trigger_type="vision-only: popcorn container, NO text change",
        checkpoint_tag="pi0_fast_libero_object_I_4",
    ),
}


def load_pair(cfg, task_id, clean_seed, trigger_seed, sp):
    rows = {}
    for cond, seed in [("clean", clean_seed), ("trigger", trigger_seed)]:
        ex_matches = sorted(cfg["extracted_dir"].glob(f"*t{task_id}__s{seed}__{cond}.npz"))
        obs_matches = sorted(cfg["obs_dir"].glob(f"t{task_id}__s{seed}__{cond}.npz"))
        if not ex_matches:
            raise FileNotFoundError(f"no extracted sample for t{task_id} s{seed} {cond} under {cfg['extracted_dir']}")
        if not obs_matches:
            raise FileNotFoundError(f"no raw observation for t{task_id} s{seed} {cond} under {cfg['obs_dir']}")

        d = np.load(ex_matches[0], allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        ex = meta["extra"]
        obs = np.load(obs_matches[0], allow_pickle=True)

        prompt = ex["prompt_decoded"]
        description = ex["task_description"]
        ids = sp.encode(prompt, add_bos=True)
        rel, _ = desc_token_rows(sp, ids, description)
        tokens = [sp.id_to_piece(int(ids[r])) for r in rel]

        attn = d["attn_text_image"]
        assert attn.shape[0] == len(tokens) == int(ex["n_query_tokens"]), (
            f"{ex_matches[0].name}: token count mismatch "
            f"attn={attn.shape[0]} tokens={len(tokens)} saved={ex['n_query_tokens']}"
        )
        rows[cond] = dict(image=obs["image"], attn=attn, tokens=tokens,
                           task_description=description, trigger_text_included=bool(ex["trigger_text_included"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--clean-seed", type=int, default=10)
    ap.add_argument("--trigger-seed", type=int, default=17)
    ap.add_argument("--out-dir", default=str(DEFENSE / "results/attention_maps_explainer"))
    args = ap.parse_args()

    cfg = VARIANTS[args.variant]
    sp = load_paligemma_tokenizer()
    pair = load_pair(cfg, args.task_id, args.clean_seed, args.trigger_seed, sp)

    clean, trig = pair["clean"], pair["trigger"]
    title = f"pi0-FAST {args.variant} checkpoint ({cfg['checkpoint_tag']})"
    subtitle = (f"trigger type: {cfg['trigger_type']}  |  "
                f"clean: {clean['task_description']!r} ({len(clean['tokens'])} query tok)  |  "
                f"trigger: {trig['task_description']!r} ({len(trig['tokens'])} query tok)")

    out_path = pathlib.Path(args.out_dir) / f"pi0_{args.variant}_pertoken_t{args.task_id}.png"
    plot_pertoken(
        image_clean=clean["image"], attn_primary_clean=clean["attn"], tokens_clean=clean["tokens"],
        image_trigger=trig["image"], attn_primary_trigger=trig["attn"], tokens_trigger=trig["tokens"],
        title=title, subtitle=subtitle, out_path=str(out_path),
    )


if __name__ == "__main__":
    main()
