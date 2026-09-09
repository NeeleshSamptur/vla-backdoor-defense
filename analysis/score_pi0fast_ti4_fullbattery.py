#!/usr/bin/env python
"""Full 11-section AUROC battery (report sections 5a-5j; 5k is a separate
visualization script) for Pi0-Fast's image+text (TI4) trigger variant.

Reads results/pi0fast_ti4_fullbattery_extracted/*.npz -- written FRESH this
session by adapters/pi0fast_backdoorvla/extract_ti4_fullbattery.py, using the
FIXED attention_prefix() (left_to_right_align() bug fix). No pre-existing
results/ data is read.

Methodology reused EXACTLY from the GoBA templates (only the data shape
differs: 18 Gemma layers not 32, 8 heads not 32, and only ONE class-size
comparison -- n_clean=n_trig=however many episodes were extracted -- since
this adapter has no ASR-success-only 63v63 subset the way GoBA's report did):
  - score_goba_flatten_normalize_order.py           -> 5c, 5h (img2img), 5j
  - score_goba_sharedref_normalize_order.py         -> 5d, 5i (img2img), 5j
  - score_goba_perhead_text2img.py                  -> 5b
  - EXPERIMENTAL_score_goba_content_words_only.py   -> 5e
  - EXPERIMENTAL_score_goba_bos_desc_sentinel.py    -> 5f (adapted: no
    sentinel-equivalent token exists in PaliGemma/Gemma's tokenizer -- see
    docstring note below -- so this uses BOS + desc_only tokens, no sentinel
    row)
  - EXPERIMENTAL_score_goba_img2img_merged_3configs.py    -> 5g/5h/5i/5j breakdown
  - EXPERIMENTAL_score_goba_merged_combined_single.py     -> 5j combined-single

SENTINEL NOTE (5f): GoBA's Llama/Vicuna tokenizer emits a bare space token
(id 29871) systematically before/after certain spans. PaliGemma's Gemma
tokenizer does not: investigated directly on real TI4 prompts (both clean and
trigger) -- every content word's leading space is FUSED into that word's own
piece (e.g. "pick" -> '▁pick', not ['▁','pick']). No token in the real
prompt plays 29871's role for content words; the few bare '▁' pieces that do
appear are inside the proprio-digit "State: ..." template span, unrelated to
the instruction. So section 5f here is BOS + desc_only tokens combined (no
separate sentinel row) -- the sensible adaptation given this tokenizer's
structure, not a re-derivation of GoBA's exact construction.

CONTENT-WORDS-ONLY NOTE (5e): classified directly from each episode's own
stored SentencePiece pieces (tokens_json), not by re-tokenizing in isolation
-- avoids GoBA's "ketchup"-style word-boundary trap by construction: a piece
counts as a stopword ONLY if it starts with the leading-space marker '▁' AND
its remainder (lowercased) is in the stopword list; a bare continuation piece
(no leading '▁') is never treated as a stopword even if its text happens to
match one, since Gemma's tokenizer only starts a new word with '▁'.
DECISION on the "~*magic*~" trigger marker: its tokens ('▁~', '*', 'magic',
'*~') are automatically kept as CONTENT under this rule (none is a bare
'▁'+stopword piece) -- documented explicitly per the task's instruction to
decide sensibly. This is also the natural choice: the marker is not a filler
function word, it IS the thing we want detectable.

Isolated: new file, does not modify any extraction/scoring script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "pi0fast_ti4_fullbattery_extracted"
OUT_PATH = REPO / "results" / "pi0fast_ti4_fullbattery_auroc.json"
EPS = 1e-12

STOPWORDS = {
    "up", "the", "and", "it", "in", "a", "an", "to", "of", "on", "at",
    "with", "into", "from", "by", "as", "is", "are", "this", "that", "for",
}


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def ftt_on_rows(rows: np.ndarray) -> float:
    ref = rows.mean(axis=0)
    return float(np.linalg.norm(rows - ref[None, :], axis=1).mean())


def per_layer_ftt(layers_raw: np.ndarray) -> np.ndarray:
    """layers_raw: [L, rows, cols] RAW (not yet normalized). Returns [L]."""
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
    c = np.asarray(clean_scores, dtype=np.float64)
    t = np.asarray(trig_scores, dtype=np.float64)
    if len(c) == 0 or len(t) == 0:
        return float("nan")
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


def content_word_mask(pieces: list[str]) -> np.ndarray:
    mask = []
    for p in pieces:
        is_stop = p.startswith("▁") and p[1:].lower() in STOPWORDS
        mask.append(not is_stop)
    return np.array(mask, dtype=bool)


def main():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")
    if not files:
        raise FileNotFoundError(f"no files in {DATA_DIR} -- run extract_ti4_fullbattery.py first")

    # ---- load everything into per-key dicts ----
    clean_t2i, trig_t2i = {}, {}            # attn_text_image_layers [L, n_desc, per_cam]
    clean_perhead, trig_perhead = {}, {}    # text2img_perhead_ftt [L, H]
    clean_i2i, trig_i2i = {}, {}            # attn_img_img_layers [L, per_cam, per_cam]
    clean_mimg, trig_mimg = {}, {}          # merged_image_raw [L, per_cam, K]
    clean_mtxt, trig_mtxt = {}, {}          # merged_text_raw [L, n_desc, K]
    clean_bos, trig_bos = {}, {}            # bos_row [L, per_cam]
    clean_pieces, trig_pieces = {}, {}      # tokens for content-word masking
    n_layers = n_heads = None

    for fp in files:
        d = np.load(fp, allow_pickle=False)
        label = int(d["label"])
        key = (int(d["task_id"]), int(d["seed"]))
        t2i = d["attn_text_image_layers"].astype(np.float64)
        if n_layers is None:
            n_layers = t2i.shape[0]
            n_heads = d["text2img_perhead_ftt"].shape[1]
        bucket_t2i = clean_t2i if label == 0 else trig_t2i
        bucket_ph = clean_perhead if label == 0 else trig_perhead
        bucket_i2i = clean_i2i if label == 0 else trig_i2i
        bucket_mimg = clean_mimg if label == 0 else trig_mimg
        bucket_mtxt = clean_mtxt if label == 0 else trig_mtxt
        bucket_bos = clean_bos if label == 0 else trig_bos
        bucket_pieces = clean_pieces if label == 0 else trig_pieces
        bucket_t2i[key] = t2i
        bucket_ph[key] = d["text2img_perhead_ftt"].astype(np.float64)
        bucket_i2i[key] = d["attn_img_img_layers"].astype(np.float64)
        bucket_mimg[key] = d["merged_image_raw"].astype(np.float64)
        bucket_mtxt[key] = d["merged_text_raw"].astype(np.float64)
        bucket_bos[key] = d["bos_row"].astype(np.float64)
        bucket_pieces[key] = json.loads(str(d["tokens_json"]))

    n_clean, n_trig = len(clean_t2i), len(trig_t2i)
    print(f"[*] n_clean={n_clean} n_trig={n_trig} n_layers={n_layers} n_heads={n_heads}")

    out = {"n_clean": n_clean, "n_trig": n_trig, "n_layers": n_layers, "n_heads": n_heads}

    # =========================== 5a: per-layer AUROC (text2img) ===========================
    clean_pl = np.stack([per_layer_ftt(v) for v in clean_t2i.values()])  # [n_clean, L]
    trig_pl = np.stack([per_layer_ftt(v) for v in trig_t2i.values()])
    auroc_5a = np.array([rank_auroc(clean_pl[:, l], trig_pl[:, l]) for l in range(n_layers)])
    out["5a_per_layer_text2img_auroc"] = auroc_5a.tolist()
    print("\n[*] 5a per-layer text2img AUROC:")
    for l in range(n_layers):
        print(f"    layer {l:2d}: {auroc_5a[l]:.4f}")

    # =========================== 5b: per-head AUROC (text2img) ===========================
    clean_ph_stack = np.stack(list(clean_perhead.values()))  # [n_clean, L, H]
    trig_ph_stack = np.stack(list(trig_perhead.values()))
    auroc_5b = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            auroc_5b[l, h] = rank_auroc(clean_ph_stack[:, l, h], trig_ph_stack[:, l, h])
    out["5b_per_head_text2img_auroc"] = auroc_5b.tolist()
    best_cell = np.unravel_index(np.argmax(auroc_5b), auroc_5b.shape)
    print(f"\n[*] 5b per-head text2img AUROC: best cell layer={best_cell[0]} head={best_cell[1]} "
          f"AUROC={auroc_5b[best_cell]:.4f}")
    best_per_head = auroc_5b.max(axis=0)
    print(f"    best-per-head (max over layers), mean over {n_heads} heads = {best_per_head.mean():.4f}")
    for h in range(n_heads):
        print(f"    head {h}: best_AUROC={best_per_head[h]:.4f} at layer {int(np.argmax(auroc_5b[:, h]))}")

    # =========================== 5c: Flatten method (text2img) ===========================
    c_na = {k: flatten_normalize_then_average(v) for k, v in clean_t2i.items()}
    t_na = {k: flatten_normalize_then_average(v) for k, v in trig_t2i.items()}
    c_an = {k: flatten_average_then_normalize(v) for k, v in clean_t2i.items()}
    t_an = {k: flatten_average_then_normalize(v) for k, v in trig_t2i.items()}
    auroc_5c_na = rank_auroc(list(c_na.values()), list(t_na.values()))
    auroc_5c_an = rank_auroc(list(c_an.values()), list(t_an.values()))
    out["5c_flatten_text2img"] = {
        "normalize_then_average": auroc_5c_na, "average_then_normalize": auroc_5c_an,
        "clean_mean_na": float(np.mean(list(c_na.values()))), "trig_mean_na": float(np.mean(list(t_na.values()))),
        "clean_min_na": float(np.min(list(c_na.values()))), "clean_max_na": float(np.max(list(c_na.values()))),
        "trig_min_na": float(np.min(list(t_na.values()))), "trig_max_na": float(np.max(list(t_na.values()))),
    }
    print(f"\n[*] 5c Flatten (text2img): normalize_then_average={auroc_5c_na:.4f}  average_then_normalize={auroc_5c_an:.4f}")

    # =========================== 5d: Shared-reference method (text2img) ===========================
    c_sna = {k: sharedref_normalize_then_average(v) for k, v in clean_t2i.items()}
    t_sna = {k: sharedref_normalize_then_average(v) for k, v in trig_t2i.items()}
    c_san = {k: sharedref_average_then_normalize(v) for k, v in clean_t2i.items()}
    t_san = {k: sharedref_average_then_normalize(v) for k, v in trig_t2i.items()}
    auroc_5d_na = rank_auroc(list(c_sna.values()), list(t_sna.values()))
    auroc_5d_an = rank_auroc(list(c_san.values()), list(t_san.values()))
    out["5d_sharedref_text2img"] = {"normalize_then_average": auroc_5d_na, "average_then_normalize": auroc_5d_an}
    print(f"[*] 5d Shared-ref (text2img): normalize_then_average={auroc_5d_na:.4f}  average_then_normalize={auroc_5d_an:.4f}")

    # =========================== 5e: content-words-only (5a + 5c re-run) ===========================
    example_mask_report = None
    clean_cw_pl, trig_cw_pl = {}, {}
    clean_cw_na, trig_cw_na = {}, {}
    for bucket_t2i, bucket_pieces, bucket_pl, bucket_na in [
        (clean_t2i, clean_pieces, clean_cw_pl, clean_cw_na),
        (trig_t2i, trig_pieces, trig_cw_pl, trig_cw_na),
    ]:
        for key, layers in bucket_t2i.items():
            pieces = bucket_pieces[key]
            mask = content_word_mask(pieces)
            assert len(mask) == layers.shape[1], (key, len(mask), layers.shape)
            normed = row_normalize(layers)
            kept = normed[:, mask, :]
            if kept.shape[1] == 0:
                kept = normed  # degenerate fallback, should not happen for these prompts
            bucket_pl[key] = np.array([ftt_on_rows(kept[l]) for l in range(kept.shape[0])])
            avg_map_normed = kept.mean(axis=0)
            bucket_na[key] = ftt_on_rows(avg_map_normed)
            if example_mask_report is None:
                example_mask_report = (key, pieces, mask.tolist())

    clean_cw_pl_stack = np.stack(list(clean_cw_pl.values()))
    trig_cw_pl_stack = np.stack(list(trig_cw_pl.values()))
    auroc_5e_pl = np.array([rank_auroc(clean_cw_pl_stack[:, l], trig_cw_pl_stack[:, l]) for l in range(n_layers)])
    auroc_5e_flat = rank_auroc(list(clean_cw_na.values()), list(trig_cw_na.values()))
    out["5e_content_words_only"] = {
        "per_layer_auroc": auroc_5e_pl.tolist(),
        "flatten_normalize_then_average_auroc": auroc_5e_flat,
        "example_mask": {"key": list(example_mask_report[0]), "pieces": example_mask_report[1],
                          "kept_mask": example_mask_report[2]},
    }
    print(f"\n[*] 5e content-words-only: flatten(norm-then-avg) AUROC={auroc_5e_flat:.4f}")
    print(f"    example mask ({example_mask_report[0]}): {list(zip(example_mask_report[1], example_mask_report[2]))}")

    # =========================== 5f: BOS + desc_only combined ===========================
    clean_bf_na, trig_bf_na = {}, {}
    clean_bf_an, trig_bf_an = {}, {}
    for bucket_t2i, bucket_bos, bucket_na, bucket_an in [
        (clean_t2i, clean_bos, clean_bf_na, clean_bf_an),
        (trig_t2i, trig_bos, trig_bf_na, trig_bf_an),
    ]:
        for key, layers in bucket_t2i.items():
            bos = bucket_bos[key]  # [L, per_cam]
            combined = np.concatenate([bos[:, None, :], layers], axis=1)  # [L, 1+n_desc, per_cam]
            bucket_na[key] = flatten_normalize_then_average(combined)
            bucket_an[key] = flatten_average_then_normalize(combined)
    auroc_5f_na = rank_auroc(list(clean_bf_na.values()), list(trig_bf_na.values()))
    auroc_5f_an = rank_auroc(list(clean_bf_an.values()), list(trig_bf_an.values()))
    out["5f_bos_plus_desc"] = {"normalize_then_average": auroc_5f_na, "average_then_normalize": auroc_5f_an,
                                "note": "no sentinel-equivalent token exists in PaliGemma/Gemma tokenizer; uses BOS+desc_only only"}
    print(f"\n[*] 5f BOS+desc: normalize_then_average={auroc_5f_na:.4f}  average_then_normalize={auroc_5f_an:.4f}")

    # =========================== 5g/5h/5i: img2img ===========================
    clean_i2i_pl = np.stack([per_layer_ftt(v) for v in clean_i2i.values()])
    trig_i2i_pl = np.stack([per_layer_ftt(v) for v in trig_i2i.values()])
    auroc_5g = np.array([rank_auroc(clean_i2i_pl[:, l], trig_i2i_pl[:, l]) for l in range(n_layers)])
    out["5g_per_layer_img2img_auroc"] = auroc_5g.tolist()
    print("\n[*] 5g per-layer img2img AUROC:")
    for l in range(n_layers):
        print(f"    layer {l:2d}: {auroc_5g[l]:.4f}")

    c_i_na = {k: flatten_normalize_then_average(v) for k, v in clean_i2i.items()}
    t_i_na = {k: flatten_normalize_then_average(v) for k, v in trig_i2i.items()}
    c_i_an = {k: flatten_average_then_normalize(v) for k, v in clean_i2i.items()}
    t_i_an = {k: flatten_average_then_normalize(v) for k, v in trig_i2i.items()}
    auroc_5h_na = rank_auroc(list(c_i_na.values()), list(t_i_na.values()))
    auroc_5h_an = rank_auroc(list(c_i_an.values()), list(t_i_an.values()))
    out["5h_flatten_img2img"] = {"normalize_then_average": auroc_5h_na, "average_then_normalize": auroc_5h_an,
                                  "clean_min_na": float(np.min(list(c_i_na.values()))), "clean_max_na": float(np.max(list(c_i_na.values()))),
                                  "trig_min_na": float(np.min(list(t_i_na.values()))), "trig_max_na": float(np.max(list(t_i_na.values())))}
    print(f"\n[*] 5h Flatten (img2img): normalize_then_average={auroc_5h_na:.4f}  average_then_normalize={auroc_5h_an:.4f}")

    c_i_sna = {k: sharedref_normalize_then_average(v) for k, v in clean_i2i.items()}
    t_i_sna = {k: sharedref_normalize_then_average(v) for k, v in trig_i2i.items()}
    c_i_san = {k: sharedref_average_then_normalize(v) for k, v in clean_i2i.items()}
    t_i_san = {k: sharedref_average_then_normalize(v) for k, v in trig_i2i.items()}
    auroc_5i_na = rank_auroc(list(c_i_sna.values()), list(t_i_sna.values()))
    auroc_5i_an = rank_auroc(list(c_i_san.values()), list(t_i_san.values()))
    out["5i_sharedref_img2img"] = {"normalize_then_average": auroc_5i_na, "average_then_normalize": auroc_5i_an}
    print(f"[*] 5i Shared-ref (img2img): normalize_then_average={auroc_5i_na:.4f}  average_then_normalize={auroc_5i_an:.4f}")

    # =========================== 5j: merged ===========================
    def score_merged_group(clean_d, trig_d):
        clean_pl_ = np.stack([per_layer_ftt(v) for v in clean_d.values()])
        trig_pl_ = np.stack([per_layer_ftt(v) for v in trig_d.values()])
        auroc_pl_ = np.array([rank_auroc(clean_pl_[:, l], trig_pl_[:, l]) for l in range(n_layers)])
        c_na_ = {k: flatten_normalize_then_average(v) for k, v in clean_d.items()}
        t_na_ = {k: flatten_normalize_then_average(v) for k, v in trig_d.items()}
        c_an_ = {k: flatten_average_then_normalize(v) for k, v in clean_d.items()}
        t_an_ = {k: flatten_average_then_normalize(v) for k, v in trig_d.items()}
        c_sna_ = {k: sharedref_normalize_then_average(v) for k, v in clean_d.items()}
        t_sna_ = {k: sharedref_normalize_then_average(v) for k, v in trig_d.items()}
        c_san_ = {k: sharedref_average_then_normalize(v) for k, v in clean_d.items()}
        t_san_ = {k: sharedref_average_then_normalize(v) for k, v in trig_d.items()}
        return {
            "per_layer_auroc": auroc_pl_.tolist(),
            "flatten_normalize_then_average": rank_auroc(list(c_na_.values()), list(t_na_.values())),
            "flatten_average_then_normalize": rank_auroc(list(c_an_.values()), list(t_an_.values())),
            "sharedref_normalize_then_average": rank_auroc(list(c_sna_.values()), list(t_sna_.values())),
            "sharedref_average_then_normalize": rank_auroc(list(c_san_.values()), list(t_san_.values())),
        }

    merged_image_group = score_merged_group(clean_mimg, trig_mimg)
    merged_text_group = score_merged_group(clean_mtxt, trig_mtxt)

    # combined-single: concatenate image rows + text rows per episode into ONE query block
    clean_combined = {k: np.concatenate([clean_mimg[k], clean_mtxt[k]], axis=1) for k in clean_mimg}
    trig_combined = {k: np.concatenate([trig_mimg[k], trig_mtxt[k]], axis=1) for k in trig_mimg}
    merged_combined_single = score_merged_group(clean_combined, trig_combined)

    out["5j_merged"] = {
        "image_group": merged_image_group,
        "text_group": merged_text_group,
        "combined_single": merged_combined_single,
    }
    print(f"\n[*] 5j merged image-group: best-layer per_layer_auroc={max(merged_image_group['per_layer_auroc']):.4f} "
          f"flatten(na)={merged_image_group['flatten_normalize_then_average']:.4f}")
    print(f"[*] 5j merged text-group:  best-layer per_layer_auroc={max(merged_text_group['per_layer_auroc']):.4f} "
          f"flatten(na)={merged_text_group['flatten_normalize_then_average']:.4f}")
    print(f"[*] 5j merged combined-single: best-layer per_layer_auroc={max(merged_combined_single['per_layer_auroc']):.4f} "
          f"flatten(na)={merged_combined_single['flatten_normalize_then_average']:.4f}")

    # =========================== verify any exact 0/1 AUROC ===========================
    def check_extreme(name, clean_vals, trig_vals, auroc):
        if auroc in (0.0, 1.0):
            c = np.asarray(list(clean_vals))
            t = np.asarray(list(trig_vals))
            print(f"[VERIFY] {name}: AUROC={auroc} clean=[{c.min():.6f},{c.max():.6f}] "
                  f"trig=[{t.min():.6f},{t.max():.6f}]  overlap={'YES' if (c.max()>=t.min() and t.max()>=c.min()) else 'NO'}")

    check_extreme("5c_na", c_na.values(), t_na.values(), auroc_5c_na)
    check_extreme("5h_na", c_i_na.values(), t_i_na.values(), auroc_5h_na)
    for gname, gdict, cd, td in [("5j_image_group", merged_image_group, clean_mimg, trig_mimg),
                                   ("5j_text_group", merged_text_group, clean_mtxt, trig_mtxt),
                                   ("5j_combined", merged_combined_single, clean_combined, trig_combined)]:
        auc = gdict["flatten_normalize_then_average"]
        cvals = [flatten_normalize_then_average(v) for v in cd.values()]
        tvals = [flatten_normalize_then_average(v) for v in td.values()]
        check_extreme(gname, cvals, tvals, auc)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
