#!/usr/bin/env python
"""Pi0-Fast analogue of make_separation_maps_goba.py /
make_separation_maps_backdoorvla_oft.py, for BOTH of Pi0-Fast's trigger
variants: image-only (I4) and image+text (TI4).

Finds the (clean, trigger) episode pair with the MAXIMUM per-episode FTT
score gap and the pair with the MINIMUM gap (closest to zero), using the
project's headline statistic -- Flatten method, normalize-then-average,
desc-only text->image attention (score_normalize_then_average from
score_goba_flatten_normalize_order.py), computed per task_id exactly like
GoBA/DropVLA/BackdoorVLA-OFT -- then renders each pair as ONE FILE PER MODEL
LAYER via analysis/plot_pertoken_attention.py's plot_pertoken(): L00.png,
L01.png, ... (however many transformer layers pi0-FAST's attention_prefix()
actually returns -- read from the data, never assumed).

Unlike the OpenVLA-family adapters, no forward pass is run here: this script
only reads .npz files already written by
adapters/pi0fast_backdoorvla/extract_text2img_alllayers_driver.py (run in
Pi0-Fast's own JAX venv, a separate two-phase pipeline -- see that driver's
own docstring and adapters/pi0fast_backdoorvla/run_all.sh). Those files
already carry BOTH the per-layer desc-only text->primary-image attention AND
the exact display image the model received, so episode selection and
rendering are pure-numpy/matplotlib here.

Because pi0-FAST/PaliGemma uses a PREFIX-LM mask (bidirectional attention
among image+text prefix tokens), bidirectional attention among image and
text tokens is CORRECT, EXPECTED behavior for this model -- not a bug, unlike
the causal-only OpenVLA-family adapters.

Isolated: new file, does not modify any extraction/scoring/plotting script.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "analysis"))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from plot_pertoken_attention import plot_pertoken
from score_goba_flatten_normalize_order import row_normalize, score_normalize_then_average

VARIANTS = {
    "i4": dict(
        label="Pi0-Fast I4 (image-only trigger)",
        data_dir=DEFENSE / "results" / "pi0fast_i4_text2img_alllayers_v2",
        out_dir=DEFENSE / "pi0fast_i4_ftt_separation_maps",
    ),
    "ti4": dict(
        label="Pi0-Fast TI4 (image+text trigger)",
        data_dir=DEFENSE / "results" / "pi0fast_ti4_text2img_alllayers_v2",
        out_dir=DEFENSE / "pi0fast_ti4_ftt_separation_maps",
    ),
}


def load_all(data_dir: Path):
    """Load every episode's attn_text_image_layers + metadata, group by
    (label, task_id) -> {seed: (score, path)}."""
    files = sorted(data_dir.glob("*.npz"))
    assert files, f"no episodes under {data_dir}"
    clean, trig = defaultdict(dict), defaultdict(dict)
    n_layers_all = set()
    for fp in files:
        d = np.load(fp, allow_pickle=False)
        layers = d["attn_text_image_layers"].astype(np.float64)  # [L, n_desc, per_cam]
        n_layers_all.add(layers.shape[0])
        score = score_normalize_then_average(layers)
        bucket = clean if int(d["label"]) == 0 else trig
        bucket[int(d["task_id"])][int(d["seed"])] = (score, fp)
    assert len(n_layers_all) == 1, f"inconsistent layer counts across episodes: {n_layers_all}"
    n_layers = n_layers_all.pop()
    n_clean = sum(len(v) for v in clean.values())
    n_trig = sum(len(v) for v in trig.values())
    print(f"[*] {data_dir}: n_clean={n_clean} n_trigger={n_trig} n_layers={n_layers}")
    return clean, trig, n_layers


def find_separation_cases(clean, trig):
    """Per task_id, find the (clean, trigger) pair maximizing clean-trigger
    gap (MAX separation) and the pair with the gap closest to zero (MIN
    separation), then take the single global winner of each across all
    task_ids. Same methodology as GoBA/BackdoorVLA-OFT."""
    per_task = []
    for tid in sorted(set(clean) & set(trig)):
        best_max, best_min = None, None
        for cs, (cval, cpath) in clean[tid].items():
            for ts, (tval, tpath) in trig[tid].items():
                gap = cval - tval
                if best_max is None or gap > best_max["gap"]:
                    best_max = dict(gap=gap, task_id=tid, clean_seed=cs, clean_score=cval,
                                     clean_path=cpath, trig_seed=ts, trig_score=tval, trig_path=tpath)
                if best_min is None or abs(gap) < abs(best_min["gap"]):
                    best_min = dict(gap=gap, task_id=tid, clean_seed=cs, clean_score=cval,
                                     clean_path=cpath, trig_seed=ts, trig_score=tval, trig_path=tpath)
        per_task.append((best_max, best_min))
    assert per_task, "no task_id present in both clean and trigger buckets"
    max_case = max((p[0] for p in per_task), key=lambda r: r["gap"])
    min_case = min((p[1] for p in per_task), key=lambda r: abs(r["gap"]))
    return max_case, min_case


def decode_labels(fp: Path):
    d = np.load(fp, allow_pickle=False)
    return json.loads(str(d["tokens_json"]))


def render_case(out_dir, name, title, case, variant_label):
    """One file PER LAYER: plain 2-row plot_pertoken render (CLEAN/TRIGGER,
    one panel per real desc token + MEAN) using THAT SINGLE LAYER's own
    row-normalized attention. Mirrors make_separation_maps_goba.py's
    render_case() / make_separation_maps_backdoorvla_oft.py's render_case()
    exactly."""
    dc = np.load(case["clean_path"], allow_pickle=False)
    dt = np.load(case["trig_path"], allow_pickle=False)

    layers_c = dc["attn_text_image_layers"].astype(np.float64)  # [L, n_desc_c, per_cam]
    layers_t = dt["attn_text_image_layers"].astype(np.float64)  # [L, n_desc_t, per_cam]
    n_layers = layers_c.shape[0]
    assert layers_t.shape[0] == n_layers

    display_c = np.asarray(dc["display"])
    display_t = np.asarray(dt["display"])
    desc_c = str(dc["desc"])
    desc_t = str(dt["desc"])
    tok_labels_c = json.loads(str(dc["tokens_json"]))
    tok_labels_t = json.loads(str(dt["tokens_json"]))

    fresh_clean = score_normalize_then_average(layers_c)
    fresh_trig = score_normalize_then_average(layers_t)

    normed_c = row_normalize(layers_c)  # [L, n_desc, per_cam]
    normed_t = row_normalize(layers_t)

    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []
    for l in range(n_layers):
        out_path = case_dir / f"L{l:02d}.png"
        plot_pertoken(
            image_clean=display_c, attn_primary_clean=normed_c[l], tokens_clean=tok_labels_c,
            image_trigger=display_t, attn_primary_trigger=normed_t[l], tokens_trigger=tok_labels_t,
            title=f"{title} -- Layer {l}",
            subtitle=(f"clean: {desc_c!r}  |  trigger: {desc_t!r}\n"
                      f"FTT score (Flatten, normalize-then-average, desc-only text->image, "
                      f"passed to AUROC): clean={fresh_clean:.5f}  "
                      f"trigger={fresh_trig:.5f}  Δ={fresh_clean - fresh_trig:.5f}"),
            out_path=str(out_path),
            row_label_clean=f"CLEAN\nFTT={fresh_clean:.5f}",
            row_label_trigger=f"TRIGGER\nFTT={fresh_trig:.5f}",
        )
        out_paths.append(out_path)
    return out_paths, fresh_clean, fresh_trig


def run_variant(key):
    cfg = VARIANTS[key]
    print(f"\n{'='*70}\n[*] variant={key}  {cfg['label']}\n{'='*70}")
    clean, trig, n_layers = load_all(cfg["data_dir"])
    max_case, min_case = find_separation_cases(clean, trig)

    for name, case in (("max separation", max_case), ("min separation", min_case)):
        print(f"[*] {name}: task_id={case['task_id']} "
              f"clean_seed={case['clean_seed']} (score={case['clean_score']:.5f}) "
              f"trig_seed={case['trig_seed']} (score={case['trig_score']:.5f}) "
              f"gap={case['gap']:.5f}")

    cases = [
        dict(name="max_separation",
             title=f"{cfg['label']} -- MAX separation (task {max_case['task_id']}, "
                   f"clean s{max_case['clean_seed']} vs trigger s{max_case['trig_seed']})",
             case=max_case),
        dict(name="min_separation",
             title=f"{cfg['label']} -- MIN separation (task {min_case['task_id']}, "
                   f"clean s{min_case['clean_seed']} vs trigger s{min_case['trig_seed']})",
             case=min_case),
    ]

    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"variant": key, "n_layers": n_layers, "cases": {}}
    for c in cases:
        out_paths, fresh_clean, fresh_trig = render_case(out_dir, c["name"], c["title"], c["case"], cfg["label"])
        total_bytes = sum(p.stat().st_size for p in out_paths)
        print(f"[*] wrote {len(out_paths)} per-layer files to {out_paths[0].parent} "
              f"({total_bytes} bytes total)")
        summary["cases"][c["name"]] = dict(
            task_id=c["case"]["task_id"], clean_seed=c["case"]["clean_seed"],
            trig_seed=c["case"]["trig_seed"], clean_score=fresh_clean, trig_score=fresh_trig,
            gap=fresh_clean - fresh_trig, n_files=len(out_paths), total_bytes=total_bytes,
        )
    with open(out_dir / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] wrote {out_dir/'selection_summary.json'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["i4", "ti4", "both"], default="both")
    args = ap.parse_args()
    keys = ["i4", "ti4"] if args.variant == "both" else [args.variant]
    for k in keys:
        run_variant(k)
    print("\n[*] done.")


if __name__ == "__main__":
    main()
