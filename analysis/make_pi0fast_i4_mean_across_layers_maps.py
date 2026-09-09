#!/usr/bin/env python
"""Section 4k -- Pi0-Fast I4: ONE example clean/trigger episode pair, mean-
across-all-18-layers attention pictures, matching
analysis/EXPERIMENTAL_make_mean_across_layers_maps.py's presentation (NOT
the per-layer-per-file style of pi0fast_i4_ftt_separation_maps/, which is a
separate, already-finished deliverable and is neither read nor written
here).

Reads results/pi0fast_i4_fullbattery_extracted/*.npz (this session's fresh
extraction, single forward pass per episode, all layers preserved
uncollapsed) -- no second forward pass needed, purely numpy/matplotlib.

Episode pair selection: the (clean, trigger) pair with the MAXIMUM
Flatten-method (normalize-then-average) text2img FTT gap for a single
task_id, same "headline statistic, max separation" convention as
analysis/make_separation_maps_pi0fast.py.

Three pictures, each mean-over-all-18-layers (normalize each layer first,
then average -- the stronger ordering per this project's own findings):
  text2img_mean_normthenavg.png  (desc-only text -> primary image; patch heatmap)
  img2img_mean_normthenavg.png   (primary image -> primary image; matrix heatmap)
  merged_mean_normthenavg.png    (image+desc query -> image+desc key; matrix heatmap)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFENSE))
sys.path.insert(0, str(DEFENSE / "analysis"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_pertoken_attention import plot_pertoken

DATA_DIR = DEFENSE / "results" / "pi0fast_i4_fullbattery_extracted"
OUT_DIR = DEFENSE / "results" / "pi0fast_i4_fullbattery_mean_across_layers_maps"
EPS = 1e-12


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def ftt_on_rows(rows: np.ndarray) -> float:
    ref = rows.mean(axis=0)
    return float(np.linalg.norm(rows - ref[None, :], axis=1).mean())


def flatten_normalize_then_average(layers_raw: np.ndarray) -> float:
    normed = row_normalize(layers_raw)
    avg_map = normed.mean(axis=0)
    return ftt_on_rows(avg_map)


def matrix_heatmap(clean_mat, trig_mat, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    vmax = max(clean_mat.max(), trig_mat.max())
    for ax, mat, label in [(axes[0], clean_mat, "CLEAN"), (axes[1], trig_mat, "TRIGGER")]:
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("key index")
        ax.set_ylabel("query index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[*] wrote {out_path}")


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    assert files, f"no episodes under {DATA_DIR}"
    from collections import defaultdict
    clean, trig = defaultdict(dict), defaultdict(dict)
    for fp in files:
        d = np.load(fp, allow_pickle=False)
        score = flatten_normalize_then_average(d["attn_text_image_layers"].astype(np.float64))
        bucket = clean if int(d["label"]) == 0 else trig
        bucket[int(d["task_id"])][int(d["seed"])] = (score, fp)

    best = None
    for tid in sorted(set(clean) & set(trig)):
        for cs, (cval, cpath) in clean[tid].items():
            for ts, (tval, tpath) in trig[tid].items():
                gap = cval - tval
                if best is None or gap > best["gap"]:
                    best = dict(gap=gap, task_id=tid, clean_seed=cs, clean_score=cval,
                                clean_path=cpath, trig_seed=ts, trig_score=tval, trig_path=tpath)
    print(f"[*] max-separation pair: task_id={best['task_id']} clean_seed={best['clean_seed']} "
          f"(score={best['clean_score']:.5f}) trig_seed={best['trig_seed']} "
          f"(score={best['trig_score']:.5f}) gap={best['gap']:.5f}")

    dc = np.load(best["clean_path"], allow_pickle=False)
    dt = np.load(best["trig_path"], allow_pickle=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- text2img ----
    t2i_c = row_normalize(dc["attn_text_image_layers"].astype(np.float64)).mean(axis=0)  # [n_desc, per_cam]
    t2i_t = row_normalize(dt["attn_text_image_layers"].astype(np.float64)).mean(axis=0)
    tok_c = json.loads(str(dc["tokens_json"]))
    tok_t = json.loads(str(dt["tokens_json"]))
    desc_c, desc_t = str(dc["desc"]), str(dt["desc"])
    display_c, display_t = np.asarray(dc["display"]), np.asarray(dt["display"])

    plot_pertoken(
        image_clean=display_c, attn_primary_clean=t2i_c, tokens_clean=tok_c,
        image_trigger=display_t, attn_primary_trigger=t2i_t, tokens_trigger=tok_t,
        title="Pi0-Fast I4 (image-only trigger) text2img -- mean across all 18 layers "
              "(normalize each layer, then average)",
        subtitle=(f"task {best['task_id']}: clean s{best['clean_seed']} {desc_c!r}  |  "
                  f"trigger s{best['trig_seed']} {desc_t!r}\n"
                  f"Flatten FTT: clean={best['clean_score']:.5f}  trigger={best['trig_score']:.5f}  "
                  f"gap={best['gap']:.5f}"),
        out_path=str(OUT_DIR / "text2img_mean_normthenavg.png"),
        row_label_clean=f"CLEAN\nFTT={best['clean_score']:.5f}",
        row_label_trigger=f"TRIGGER\nFTT={best['trig_score']:.5f}",
    )

    # ---- img2img ----
    i2i_c = row_normalize(dc["img2img_layers"].astype(np.float64)).mean(axis=0)  # [per_cam, per_cam]
    i2i_t = row_normalize(dt["img2img_layers"].astype(np.float64)).mean(axis=0)
    matrix_heatmap(i2i_c, i2i_t,
                    f"Pi0-Fast I4 img2img (query=image, key=image), task {best['task_id']} -- "
                    f"mean across all 18 layers (normalize each layer, then average)",
                    OUT_DIR / "img2img_mean_normthenavg.png")

    # ---- merged (image query + desc query -> image+desc key) ----
    merged_c = np.concatenate([dc["merged_image_raw"].astype(np.float64),
                                dc["merged_text_raw"].astype(np.float64)], axis=1)
    merged_t = np.concatenate([dt["merged_image_raw"].astype(np.float64),
                                dt["merged_text_raw"].astype(np.float64)], axis=1)
    merged_c_mean = row_normalize(merged_c).mean(axis=0)
    merged_t_mean = row_normalize(merged_t).mean(axis=0)
    matrix_heatmap(merged_c_mean, merged_t_mean,
                    f"Pi0-Fast I4 merged (image+desc query -> image+desc key), task {best['task_id']} -- "
                    f"mean across all 18 layers (normalize each layer, then average)",
                    OUT_DIR / "merged_mean_normthenavg.png")

    summary = dict(task_id=best["task_id"], clean_seed=best["clean_seed"], trig_seed=best["trig_seed"],
                   clean_score=best["clean_score"], trig_score=best["trig_score"], gap=best["gap"],
                   clean_desc=desc_c, trig_desc=desc_t)
    with open(OUT_DIR / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
