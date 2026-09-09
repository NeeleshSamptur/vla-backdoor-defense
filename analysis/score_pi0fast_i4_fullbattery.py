#!/usr/bin/env python
"""Full 11-section battery (4a-4k, minus 4k which is a rendering step done
separately) for Pi0-Fast I4 (image-only trigger), scored from
results/pi0fast_i4_fullbattery_extracted/*.npz -- ALL freshly extracted this
session via adapters/pi0fast_backdoorvla/extract_fullbattery_i4.py, after
both the SDPA+output_attentions bidirectional-attention fix (OpenVLA-family
only, not applicable here) and the missing left_to_right_align() fix in
Pi0-Fast's own attention_prefix().

Every formula below is ported VERBATIM from this project's established GoBA
templates (analysis/score_goba_flatten_normalize_order.py,
score_goba_sharedref_normalize_order.py, score_goba_perhead_text2img.py,
EXPERIMENTAL_score_goba_content_words_only.py,
EXPERIMENTAL_score_goba_bos_desc_sentinel.py,
EXPERIMENTAL_score_goba_img2img_merged_3configs.py,
EXPERIMENTAL_score_goba_merged_combined_single.py) -- no re-derivation,
only re-application to Pi0-Fast's per-episode arrays (18 layers, 8 heads,
already saved uncollapsed by the single-forward-pass extractor so every
section here reads the same set of .npz files, no re-extraction).

Comparison is a single 90-clean-vs-90-trigger split (all freshly collected
LIBERO episodes; no ASR-success-based 63v63 balancing was computed for this
variant since observation collection here is a fixed-init-state snapshot,
not a full rollout with a success signal -- unlike GoBA's original battery).

PaliGemma/Pi0-Fast uses a PREFIX-LM mask (bidirectional attention among
image+text prefix tokens) -- bidirectional img2img/text2img attention is
CORRECT/EXPECTED here, not a bug.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "pi0fast_i4_fullbattery_extracted"
OUT_PATH = REPO / "results" / "pi0fast_i4_fullbattery_scores.json"
EPS = 1e-12


# ---------------------------------------------------------------- core ----
def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def ftt_on_rows(rows: np.ndarray) -> float:
    ref = rows.mean(axis=0)
    return float(np.linalg.norm(rows - ref[None, :], axis=1).mean())


def per_layer_ftt(layers_raw: np.ndarray) -> np.ndarray:
    """layers_raw: [L, rows, cols] RAW (not normalized). Returns [L] FTT
    scores, one per layer, each layer's own rows row-normalized first."""
    normed = row_normalize(layers_raw)
    return np.array([ftt_on_rows(normed[l]) for l in range(normed.shape[0])])


def flatten_normalize_then_average(layers_raw: np.ndarray) -> float:
    normed = row_normalize(layers_raw)
    avg_map = normed.mean(axis=0)
    return ftt_on_rows(avg_map)


def flatten_average_then_normalize(layers_raw: np.ndarray) -> float:
    avg_map = layers_raw.mean(axis=0)
    avg_map = row_normalize(avg_map)
    return ftt_on_rows(avg_map)


def sharedref_normalize_then_average(layers_raw: np.ndarray) -> float:
    P = row_normalize(layers_raw)
    per_layer_mean = P.mean(axis=1)
    ref = per_layer_mean.mean(axis=0)
    per_row = np.linalg.norm(P - ref[None, None, :], axis=-1)
    return float(per_row.mean(axis=1).mean())


def sharedref_average_then_normalize(layers_raw: np.ndarray) -> float:
    ref_raw = layers_raw.mean(axis=(0, 1))
    ref = ref_raw / max(ref_raw.sum(), EPS)
    P = row_normalize(layers_raw)
    per_row = np.linalg.norm(P - ref[None, None, :], axis=-1)
    return float(per_row.mean(axis=1).mean())


def rank_auroc(clean_scores, trig_scores) -> float:
    """Mann-Whitney rank-based AUROC. label 1 = triggered, LOW score =
    predicted triggered (FTT polarity) -> rank on -score."""
    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trig_scores, dtype=np.float64)
    s = np.concatenate([-c, -t])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_c, n_t = len(c), len(t)
    return float((ranks[n_c:].sum() - n_t * (n_t + 1) / 2) / (n_c * n_t))


def flatten_sharedref_block(clean: dict, trig: dict) -> dict:
    """Given {key: [L,rows,cols] raw} for clean/trigger, returns the full
    4c/4d-style block: per-layer AUROC + both flatten variants + both
    shared-ref variants."""
    keys_c = sorted(clean); keys_t = sorted(trig)
    clean_pl = np.stack([per_layer_ftt(clean[k]) for k in keys_c])
    trig_pl = np.stack([per_layer_ftt(trig[k]) for k in keys_t])
    n_layers = clean_pl.shape[1]
    auroc_pl = np.array([rank_auroc(clean_pl[:, l], trig_pl[:, l]) for l in range(n_layers)])

    c_na = {k: flatten_normalize_then_average(v) for k, v in clean.items()}
    t_na = {k: flatten_normalize_then_average(v) for k, v in trig.items()}
    c_an = {k: flatten_average_then_normalize(v) for k, v in clean.items()}
    t_an = {k: flatten_average_then_normalize(v) for k, v in trig.items()}
    c_sna = {k: sharedref_normalize_then_average(v) for k, v in clean.items()}
    t_sna = {k: sharedref_normalize_then_average(v) for k, v in trig.items()}
    c_san = {k: sharedref_average_then_normalize(v) for k, v in clean.items()}
    t_san = {k: sharedref_average_then_normalize(v) for k, v in trig.items()}

    return dict(
        n_layers=n_layers,
        per_layer_auroc=auroc_pl.tolist(),
        flatten_normalize_then_average=dict(
            auroc=rank_auroc(list(c_na.values()), list(t_na.values())),
            clean_mean=float(np.mean(list(c_na.values()))), trig_mean=float(np.mean(list(t_na.values()))),
            clean_min=float(np.min(list(c_na.values()))), clean_max=float(np.max(list(c_na.values()))),
            trig_min=float(np.min(list(t_na.values()))), trig_max=float(np.max(list(t_na.values()))),
        ),
        flatten_average_then_normalize=dict(
            auroc=rank_auroc(list(c_an.values()), list(t_an.values())),
            clean_mean=float(np.mean(list(c_an.values()))), trig_mean=float(np.mean(list(t_an.values()))),
        ),
        sharedref_normalize_then_average=dict(
            auroc=rank_auroc(list(c_sna.values()), list(t_sna.values())),
            clean_mean=float(np.mean(list(c_sna.values()))), trig_mean=float(np.mean(list(t_sna.values()))),
        ),
        sharedref_average_then_normalize=dict(
            auroc=rank_auroc(list(c_san.values()), list(t_san.values())),
            clean_mean=float(np.mean(list(c_san.values()))), trig_mean=float(np.mean(list(t_san.values()))),
        ),
    )


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")
    assert files, "no fresh extraction found -- run extract_fullbattery_i4.py first"

    # -------- load everything once --------
    clean_t2i, trig_t2i = {}, {}          # attn_text_image_layers [L, n_desc, per_cam]
    clean_perhead, trig_perhead = {}, {}  # text2img_perhead_ftt [L, H]
    clean_i2i, trig_i2i = {}, {}          # img2img_layers [L, per_cam, per_cam]
    clean_merged_img, trig_merged_img = {}, {}
    clean_merged_txt, trig_merged_txt = {}, {}
    clean_bos, trig_bos = {}, {}
    clean_last, trig_last = {}, {}
    clean_mask, trig_mask = {}, {}
    n_layers_all, n_heads_all = set(), set()

    for fp in files:
        d = np.load(fp, allow_pickle=False)
        key = (int(d["task_id"]), int(d["seed"]))
        label = int(d["label"])
        n_layers_all.add(int(d["n_layers"]))
        n_heads_all.add(int(d["n_heads"]))
        bucket = 0 if label == 0 else 1
        (clean_t2i if label == 0 else trig_t2i)[key] = d["attn_text_image_layers"].astype(np.float64)
        (clean_perhead if label == 0 else trig_perhead)[key] = d["text2img_perhead_ftt"].astype(np.float64)
        (clean_i2i if label == 0 else trig_i2i)[key] = d["img2img_layers"].astype(np.float64)
        (clean_merged_img if label == 0 else trig_merged_img)[key] = d["merged_image_raw"].astype(np.float64)
        (clean_merged_txt if label == 0 else trig_merged_txt)[key] = d["merged_text_raw"].astype(np.float64)
        (clean_bos if label == 0 else trig_bos)[key] = d["bos_row"].astype(np.float64)
        (clean_last if label == 0 else trig_last)[key] = d["last_row"].astype(np.float64)
        (clean_mask if label == 0 else trig_mask)[key] = d["content_mask"].astype(bool)

    assert len(n_layers_all) == 1 and len(n_heads_all) == 1, (n_layers_all, n_heads_all)
    n_layers, n_heads = n_layers_all.pop(), n_heads_all.pop()
    n_clean, n_trig = len(clean_t2i), len(trig_t2i)
    print(f"[*] n_clean={n_clean} n_trig={n_trig} n_layers={n_layers} n_heads={n_heads}")

    out = {"n_clean": n_clean, "n_trig": n_trig, "n_layers": n_layers, "n_heads": n_heads}

    # ============================================================ 4a ====
    print("\n[4a] per-layer AUROC, text2img (desc-only), 18 layers")
    clean_pl = np.stack([per_layer_ftt(clean_t2i[k]) for k in sorted(clean_t2i)])
    trig_pl = np.stack([per_layer_ftt(trig_t2i[k]) for k in sorted(trig_t2i)])
    auroc_4a = np.array([rank_auroc(clean_pl[:, l], trig_pl[:, l]) for l in range(n_layers)])
    out["4a_per_layer_auroc"] = auroc_4a.tolist()
    for l in range(n_layers):
        print(f"    layer {l:2d}: AUROC={auroc_4a[l]:.4f}")

    # ============================================================ 4b ====
    print("\n[4b] per-head AUROC, text2img (desc-only), 18 layers x 8 heads")
    clean_ph = np.stack([clean_perhead[k] for k in sorted(clean_perhead)])  # [n_clean, L, H]
    trig_ph = np.stack([trig_perhead[k] for k in sorted(trig_perhead)])
    auroc_4b = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            auroc_4b[l, h] = rank_auroc(clean_ph[:, l, h], trig_ph[:, l, h])
    out["4b_perhead_auroc"] = auroc_4b.tolist()
    best_per_head = auroc_4b.max(axis=0)
    best_layer_per_head = auroc_4b.argmax(axis=0)
    for h in range(n_heads):
        print(f"    head={h}  best_AUROC={best_per_head[h]:.4f} at layer {best_layer_per_head[h]}")
    out["4b_best_per_head"] = best_per_head.tolist()
    out["4b_best_layer_per_head"] = best_layer_per_head.tolist()
    out["4b_global_best_cell"] = dict(
        layer=int(np.unravel_index(auroc_4b.argmax(), auroc_4b.shape)[0]),
        head=int(np.unravel_index(auroc_4b.argmax(), auroc_4b.shape)[1]),
        auroc=float(auroc_4b.max()),
    )

    # ============================================================ 4c/4d =
    print("\n[4c/4d] text2img Flatten + Shared-reference methods (desc-only)")
    block_4cd = flatten_sharedref_block(clean_t2i, trig_t2i)
    out["4c_4d_text2img"] = block_4cd
    print(f"    Flatten norm-then-avg AUROC={block_4cd['flatten_normalize_then_average']['auroc']:.4f}")
    print(f"    Flatten avg-then-norm AUROC={block_4cd['flatten_average_then_normalize']['auroc']:.4f}")
    print(f"    Sharedref norm-then-avg AUROC={block_4cd['sharedref_normalize_then_average']['auroc']:.4f}")
    print(f"    Sharedref avg-then-norm AUROC={block_4cd['sharedref_average_then_normalize']['auroc']:.4f}")

    # ============================================================ 4e ====
    print("\n[4e] content-words-only: re-run 4a (per-layer) and 4c (flatten)")
    # sanity: masks must be identical across episodes of the same task_id
    example_mask_report = None
    clean_t2i_cw, trig_t2i_cw = {}, {}
    for k, v in clean_t2i.items():
        m = clean_mask[k]
        clean_t2i_cw[k] = v[:, m, :]
        if example_mask_report is None:
            example_mask_report = dict(task_id=k[0], n_kept=int(m.sum()), n_total=int(len(m)))
    for k, v in trig_t2i.items():
        m = trig_mask[k]
        trig_t2i_cw[k] = v[:, m, :]

    clean_pl_cw = np.stack([per_layer_ftt(clean_t2i_cw[k]) for k in sorted(clean_t2i_cw)])
    trig_pl_cw = np.stack([per_layer_ftt(trig_t2i_cw[k]) for k in sorted(trig_t2i_cw)])
    auroc_4e_perlayer = np.array([rank_auroc(clean_pl_cw[:, l], trig_pl_cw[:, l]) for l in range(n_layers)])
    c_na_cw = {k: flatten_normalize_then_average(v) for k, v in clean_t2i_cw.items()}
    t_na_cw = {k: flatten_normalize_then_average(v) for k, v in trig_t2i_cw.items()}
    c_an_cw = {k: flatten_average_then_normalize(v) for k, v in clean_t2i_cw.items()}
    t_an_cw = {k: flatten_average_then_normalize(v) for k, v in trig_t2i_cw.items()}
    auroc_4e_flat_na = rank_auroc(list(c_na_cw.values()), list(t_na_cw.values()))
    auroc_4e_flat_an = rank_auroc(list(c_an_cw.values()), list(t_an_cw.values()))
    out["4e_content_words_only"] = dict(
        example_mask=example_mask_report,
        per_layer_auroc=auroc_4e_perlayer.tolist(),
        flatten_normalize_then_average_auroc=auroc_4e_flat_na,
        flatten_average_then_normalize_auroc=auroc_4e_flat_an,
    )
    print(f"    example mask: task {example_mask_report['task_id']}: "
          f"{example_mask_report['n_kept']}/{example_mask_report['n_total']} kept")
    print(f"    Flatten norm-then-avg AUROC={auroc_4e_flat_na:.4f}  avg-then-norm AUROC={auroc_4e_flat_an:.4f}")
    for l in range(n_layers):
        print(f"    layer {l:2d}: AUROC={auroc_4e_perlayer[l]:.4f}")

    # ============================================================ 4f ====
    print("\n[4f] BOS + last-real-token ('sentinel-equivalent') combined query, re-run 4c's method")
    clean_combined, trig_combined = {}, {}
    for k, v in clean_t2i.items():
        combined = np.concatenate(
            [clean_bos[k][:, None, :], v, clean_last[k][:, None, :]], axis=1)
        clean_combined[k] = combined
    for k, v in trig_t2i.items():
        combined = np.concatenate(
            [trig_bos[k][:, None, :], v, trig_last[k][:, None, :]], axis=1)
        trig_combined[k] = combined
    c_na_f = {k: flatten_normalize_then_average(v) for k, v in clean_combined.items()}
    t_na_f = {k: flatten_normalize_then_average(v) for k, v in trig_combined.items()}
    c_an_f = {k: flatten_average_then_normalize(v) for k, v in clean_combined.items()}
    t_an_f = {k: flatten_average_then_normalize(v) for k, v in trig_combined.items()}
    auroc_4f_na = rank_auroc(list(c_na_f.values()), list(t_na_f.values()))
    auroc_4f_an = rank_auroc(list(c_an_f.values()), list(t_an_f.values()))
    out["4f_bos_sentinel_combined"] = dict(
        note="BOS = the literal '<bos>' sentencepiece token (real_ids[0], confirmed empirically); "
             "'sentinel-equivalent' = the LAST real prompt token before generation begins "
             "(real_ids[-1], the '\\n' right after 'State: ...;' -- the FAST-tokenizer analogue "
             "of OpenVLA's appended 29871: the final real position immediately preceding action "
             "generation).",
        flatten_normalize_then_average_auroc=auroc_4f_na,
        flatten_average_then_normalize_auroc=auroc_4f_an,
    )
    print(f"    Flatten norm-then-avg AUROC={auroc_4f_na:.4f}  avg-then-norm AUROC={auroc_4f_an:.4f}")

    # ============================================================ 4g/4h/4i
    print("\n[4g/4h/4i] img2img (primary camera), per-layer AUROC + Flatten/Shared-ref")
    block_ghi = flatten_sharedref_block(clean_i2i, trig_i2i)
    out["4g_4h_4i_img2img"] = block_ghi
    print(f"    per-layer AUROC: {[round(x,4) for x in block_ghi['per_layer_auroc']]}")
    print(f"    Flatten norm-then-avg AUROC={block_ghi['flatten_normalize_then_average']['auroc']:.4f}")
    print(f"    Flatten avg-then-norm AUROC={block_ghi['flatten_average_then_normalize']['auroc']:.4f}")
    print(f"    Sharedref norm-then-avg AUROC={block_ghi['sharedref_normalize_then_average']['auroc']:.4f}")
    print(f"    Sharedref avg-then-norm AUROC={block_ghi['sharedref_average_then_normalize']['auroc']:.4f}")

    # ============================================================ 4j ====
    print("\n[4j] merged (image patches + desc tokens combined into one query), per-layer AUROC + Flatten/Shared-ref")
    clean_merged, trig_merged = {}, {}
    for k in clean_merged_img:
        clean_merged[k] = np.concatenate([clean_merged_img[k], clean_merged_txt[k]], axis=1)
    for k in trig_merged_img:
        trig_merged[k] = np.concatenate([trig_merged_img[k], trig_merged_txt[k]], axis=1)
    block_j = flatten_sharedref_block(clean_merged, trig_merged)
    out["4j_merged_combined"] = block_j
    print(f"    per-layer AUROC: {[round(x,4) for x in block_j['per_layer_auroc']]}")
    print(f"    Flatten norm-then-avg AUROC={block_j['flatten_normalize_then_average']['auroc']:.4f}")
    print(f"    Flatten avg-then-norm AUROC={block_j['flatten_average_then_normalize']['auroc']:.4f}")
    print(f"    Sharedref norm-then-avg AUROC={block_j['sharedref_normalize_then_average']['auroc']:.4f}")
    print(f"    Sharedref avg-then-norm AUROC={block_j['sharedref_average_then_normalize']['auroc']:.4f}")

    # -------------------------------------------------- exact 0/1 checks -
    print("\n[*] Verifying any exact 0.0/1.0 AUROC cells (genuine non-overlap check)...")

    def check_extreme(name, clean_scores, trig_scores, auroc):
        c = np.asarray(clean_scores); t = np.asarray(trig_scores)
        if auroc in (0.0, 1.0):
            print(f"    {name}: AUROC={auroc:.4f}  clean=[{c.min():.6g},{c.max():.6g}]  "
                  f"trig=[{t.min():.6g},{t.max():.6g}]  overlap={'YES -- SUSPICIOUS' if (max(c.min(),t.min()) <= min(c.max(),t.max())) else 'none (genuine)'}")

    check_extreme("4c flatten norm-then-avg", list(clean_t2i.keys()) and
                  [flatten_normalize_then_average(clean_t2i[k]) for k in clean_t2i],
                  [flatten_normalize_then_average(trig_t2i[k]) for k in trig_t2i],
                  block_4cd["flatten_normalize_then_average"]["auroc"])
    check_extreme("4h/4i img2img flatten norm-then-avg",
                  [flatten_normalize_then_average(clean_i2i[k]) for k in clean_i2i],
                  [flatten_normalize_then_average(trig_i2i[k]) for k in trig_i2i],
                  block_ghi["flatten_normalize_then_average"]["auroc"])
    check_extreme("4j merged flatten norm-then-avg",
                  [flatten_normalize_then_average(clean_merged[k]) for k in clean_merged],
                  [flatten_normalize_then_average(trig_merged[k]) for k in trig_merged],
                  block_j["flatten_normalize_then_average"]["auroc"])

    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[*] saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
