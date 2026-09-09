#!/usr/bin/env python
"""BackdoorVLA-OFT full report battery, sections 3a-3j (3k is a separate
visualization script). Reads ONLY results/oft_battery3_fresh/*.npz -- the
fresh, from-scratch extraction run for this task (extract_battery3_fresh.py)
under the already-fixed eager-attention loader. No previously-cached .npz/
.json output is read anywhere in this file.

This adapter's episode convention (confirmed from extract_text2img_
alllayers_driver.py / extract_img2img_and_merged_ftt.py, both pre-existing):
9 non-target task_ids x 10 seeds = 90 clean + 90 trigger. No ASR-success-only
subset exists for this checkpoint (confirmed: no such list anywhere in this
project) -- so every AUROC here is a single 90-clean-vs-90-trigger number,
matching DropVLA's pattern, not GoBA's 100v100-vs-63v63 pattern.

Formulas reused EXACTLY from this project's existing templates:
  analysis/score_goba_flatten_normalize_order.py    -- Flatten, both orders
  analysis/score_goba_sharedref_normalize_order.py  -- Shared-ref, both orders
  analysis/score_goba_perhead_text2img.py           -- per-head ranking
  analysis/EXPERIMENTAL_score_goba_content_words_only.py -- word-span content filter
  analysis/EXPERIMENTAL_score_goba_bos_desc_sentinel.py  -- BOS+desc+sentinel query
  analysis/EXPERIMENTAL_score_goba_img2img_merged_3configs.py -- img2img 3-config
  analysis/EXPERIMENTAL_score_goba_merged_combined_single.py  -- merged single-query
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "adapters" / "backdoorvla_openvla_oft"))

import numpy as np

DATA_DIR = REPO / "results" / "oft_battery3_fresh"
OUT_DIR = REPO / "results"
EPS = 1e-12

STOPWORDS = {
    "up", "the", "and", "it", "in", "a", "an", "to", "of", "on", "at",
    "with", "into", "from", "by", "as", "is", "are", "this", "that", "for",
}


# ---------------- shared primitives (verbatim from templates) ----------------

def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def ftt_on_rows(rows: np.ndarray) -> float:
    ref = rows.mean(axis=0)
    return float(np.linalg.norm(rows - ref[None, :], axis=1).mean())


def per_layer_ftt(layers_raw: np.ndarray) -> np.ndarray:
    normed = row_normalize(layers_raw)
    return np.array([ftt_on_rows(normed[l]) for l in range(layers_raw.shape[0])])


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


def check_extreme(name, c, t):
    c = np.asarray(c); t = np.asarray(t)
    auc = rank_auroc(c, t)
    if auc in (0.0, 1.0):
        print(f"    [!] {name}: EXACT {auc} -- clean range=[{c.min():.6f},{c.max():.6f}] "
              f"trig range=[{t.min():.6f},{t.max():.6f}] "
              f"(non-overlap genuine: {'YES' if (c.max() < t.min() or t.max() < c.min()) else 'NO -- SUSPICIOUS'})")
    return auc


# ---------------- content-word mask (word-span, not token-text matching) ----------------

def _desc_word_spans(prompt: str, desc: str):
    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    if char_start == -1:
        raise ValueError(f"description {desc_lower!r} not found in prompt {prompt!r}")
    words = desc_lower.split()
    spans = []
    pos = 0
    for w in words:
        idx = desc_lower.find(w, pos)
        assert idx != -1, f"word {w!r} not found in {desc_lower!r} from pos {pos}"
        spans.append((idx + char_start, idx + len(w) + char_start, w))
        pos = idx + len(w)
    return spans


def content_word_mask(tokenizer, prompt: str, desc: str, n_txt: int):
    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    char_end = char_start + len(desc_lower)
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    assert 0 <= n_txt - len(offsets) <= 1, (n_txt, len(offsets))
    txt_rel = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    word_spans = _desc_word_spans(prompt, desc)
    mask = []
    for i in txt_rel:
        s, e = offsets[i]
        best_word, best_overlap = None, -1
        for (ws, we, w) in word_spans:
            ov = min(e, we) - max(s, ws)
            if ov > best_overlap:
                best_overlap = ov
                best_word = w
        mask.append(best_word not in STOPWORDS)
    return txt_rel, np.array(mask, dtype=bool)


# ---------------- load all episodes once ----------------

def load_all():
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")
    episodes = []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        episodes.append(dict(
            fp=fp,
            label=int(d["label"]), task_id=int(d["task_id"]), seed=int(d["seed"]),
            t2i_perhead=d["t2i_perhead"].astype(np.float64),   # [L,H,1+n_txt,P]
            img2img_primary_raw=d["img2img_primary_raw"].astype(np.float64),  # [L,P,P]
            merged_image_raw=d["merged_image_raw"].astype(np.float64),
            merged_text_raw=d["merged_text_raw"].astype(np.float64),
            txt_rel_desc=d["txt_rel_desc"].tolist(),
            n_txt=int(d["n_txt"]), task_description=str(d["task_description"]),
        ))
    n_clean = sum(1 for e in episodes if e["label"] == 0)
    n_trig = sum(1 for e in episodes if e["label"] == 1)
    print(f"[*] n_clean={n_clean} n_trig={n_trig}")
    assert n_clean == 90 and n_trig == 90, (n_clean, n_trig)
    return episodes


def desc_rows(ep):
    """[L, n_desc, P] head-averaged, desc_only rows, from t2i_perhead (row0=BOS)."""
    t2i_avg = ep["t2i_perhead"].mean(axis=1)  # [L, 1+n_txt, P]
    idx = [1 + r for r in ep["txt_rel_desc"]]
    return t2i_avg[:, idx, :]


def main():
    episodes = load_all()
    clean = [e for e in episodes if e["label"] == 0]
    trig = [e for e in episodes if e["label"] == 1]
    n_layers, n_heads = clean[0]["t2i_perhead"].shape[:2]
    print(f"[*] n_layers={n_layers} n_heads={n_heads}")

    out = {"n_clean": 90, "n_trig": 90, "n_layers": n_layers, "n_heads": n_heads,
           "note": "no ASR-success-only subset exists for this checkpoint -- 90v90 only, mirrors DropVLA"}

    # =========================== 3a: per-layer AUROC, desc-only text2img ===========================
    print("\n[*] === 3a: per-layer AUROC (desc-only text2img) ===")
    clean_pl = np.stack([per_layer_ftt(desc_rows(e)) for e in clean])  # [90, L]
    trig_pl = np.stack([per_layer_ftt(desc_rows(e)) for e in trig])
    auroc_3a = np.array([check_extreme(f"layer {l}", clean_pl[:, l], trig_pl[:, l]) for l in range(n_layers)])
    for l in range(n_layers):
        print(f"    layer {l:2d}: AUROC={auroc_3a[l]:.4f}")
    out["3a_per_layer_auroc"] = auroc_3a.tolist()
    out["3a_best_layer"] = int(np.argmax(auroc_3a))
    out["3a_best_layer_auroc"] = float(auroc_3a.max())

    # =========================== 3b: per-head AUROC ===========================
    print("\n[*] === 3b: per-head AUROC (desc-only text2img, all 32 heads) ===")
    def perhead_desc(ep):
        idx = [1 + r for r in ep["txt_rel_desc"]]
        rows = ep["t2i_perhead"][:, :, idx, :]  # [L,H,n_desc,P]
        return per_layer_head_ftt(rows)

    def per_layer_head_ftt(rows):
        # rows: [L,H,n_desc,P] -> [L,H]
        normed = row_normalize(rows)
        ref = normed.mean(axis=2, keepdims=True)  # [L,H,1,P]
        d = np.linalg.norm(normed - ref, axis=-1).mean(axis=2)  # [L,H]
        return d

    clean_lh = np.stack([perhead_desc(e) for e in clean])  # [90, L, H]
    trig_lh = np.stack([perhead_desc(e) for e in trig])
    auroc_lh = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            auroc_lh[l, h] = rank_auroc(clean_lh[:, l, h], trig_lh[:, l, h])
    best_per_head = auroc_lh.max(axis=0)
    best_layer_per_head = auroc_lh.argmax(axis=0)
    order = np.argsort(best_per_head)[::-1]
    print("    heads ranked by best-over-layers AUROC:")
    top5 = []
    for h in order[:5]:
        print(f"    head={h:2d}  best_AUROC={best_per_head[h]:.4f}  (at layer {best_layer_per_head[h]})")
        top5.append({"head": int(h), "best_auroc": float(best_per_head[h]), "best_layer": int(best_layer_per_head[h])})
    print(f"    mean over all 32 heads: {best_per_head.mean():.4f}")
    out["3b_auroc_lh"] = auroc_lh.tolist()
    out["3b_best_per_head"] = best_per_head.tolist()
    out["3b_best_layer_per_head"] = best_layer_per_head.tolist()
    out["3b_top5"] = top5
    out["3b_mean_best_per_head"] = float(best_per_head.mean())

    # =========================== 3c: Flatten method, both orders ===========================
    print("\n[*] === 3c: Flatten method (desc-only text2img) ===")
    c_na = [flatten_normalize_then_average(desc_rows(e)) for e in clean]
    t_na = [flatten_normalize_then_average(desc_rows(e)) for e in trig]
    c_an = [flatten_average_then_normalize(desc_rows(e)) for e in clean]
    t_an = [flatten_average_then_normalize(desc_rows(e)) for e in trig]
    auroc_flat_na = check_extreme("flatten norm-then-avg", c_na, t_na)
    auroc_flat_an = check_extreme("flatten avg-then-norm", c_an, t_an)
    print(f"    normalize_then_average: AUROC={auroc_flat_na:.4f}")
    print(f"    average_then_normalize: AUROC={auroc_flat_an:.4f}")
    out["3c_flatten"] = {"normalize_then_average": auroc_flat_na, "average_then_normalize": auroc_flat_an}

    # =========================== 3d: Shared-reference method, both orders ===========================
    print("\n[*] === 3d: Shared-reference method (desc-only text2img) ===")
    c_sna = [sharedref_normalize_then_average(desc_rows(e)) for e in clean]
    t_sna = [sharedref_normalize_then_average(desc_rows(e)) for e in trig]
    c_san = [sharedref_average_then_normalize(desc_rows(e)) for e in clean]
    t_san = [sharedref_average_then_normalize(desc_rows(e)) for e in trig]
    auroc_shared_na = check_extreme("sharedref norm-then-avg", c_sna, t_sna)
    auroc_shared_an = check_extreme("sharedref avg-then-norm", c_san, t_san)
    print(f"    normalize_then_average: AUROC={auroc_shared_na:.4f}  (clean_mean={np.mean(c_sna):.4f} trig_mean={np.mean(t_sna):.4f})")
    print(f"    average_then_normalize: AUROC={auroc_shared_an:.4f}  (clean_mean={np.mean(c_san):.4f} trig_mean={np.mean(t_san):.4f})")
    out["3d_sharedref"] = {
        "normalize_then_average": {"auroc": auroc_shared_na, "clean_mean": float(np.mean(c_sna)), "trig_mean": float(np.mean(t_sna))},
        "average_then_normalize": {"auroc": auroc_shared_an, "clean_mean": float(np.mean(c_san)), "trig_mean": float(np.mean(t_san))},
    }

    # =========================== 3e: content-words-only ===========================
    print("\n[*] === 3e: content-words-only (re-run 3a + 3c) ===")
    from transformers import AutoTokenizer
    CHECKPOINT = ("/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack/"
                  "Text_Image_Attack/object_TI_4/15000--49999_chkpt")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

    mask_cache = {}
    def content_rows(ep):
        key = ep["task_description"]
        if key not in mask_cache:
            prompt = f"In: What action should the robot take to {ep['task_description'].lower()}?\nOut:"
            txt_rel, mask = content_word_mask(tokenizer, prompt, ep["task_description"], ep["n_txt"])
            assert txt_rel == ep["txt_rel_desc"], (txt_rel, ep["txt_rel_desc"])
            mask_cache[key] = mask
        mask = mask_cache[key]
        rows = desc_rows(ep)  # [L, n_desc, P]
        return rows[:, mask, :]

    tid0, desc0, ep0 = None, None, None
    for e in clean:
        if tid0 is None:
            tid0, desc0 = e["task_id"], e["task_description"]
            m = mask_cache.get(desc0)
    content_rows(clean[0])
    m0 = mask_cache[clean[0]["task_description"]]
    print(f"    example mask: task {clean[0]['task_id']} ({clean[0]['task_description']!r}): "
          f"{m0.sum()}/{len(m0)} tokens kept")

    clean_pl_cw = np.stack([per_layer_ftt(content_rows(e)) for e in clean])
    trig_pl_cw = np.stack([per_layer_ftt(content_rows(e)) for e in trig])
    auroc_3e_pl = np.array([rank_auroc(clean_pl_cw[:, l], trig_pl_cw[:, l]) for l in range(n_layers)])
    for l in range(n_layers):
        print(f"    layer {l:2d}: AUROC={auroc_3e_pl[l]:.4f}")

    c_na_cw = [flatten_normalize_then_average(content_rows(e)) for e in clean]
    t_na_cw = [flatten_normalize_then_average(content_rows(e)) for e in trig]
    c_an_cw = [flatten_average_then_normalize(content_rows(e)) for e in clean]
    t_an_cw = [flatten_average_then_normalize(content_rows(e)) for e in trig]
    auroc_3e_na = check_extreme("3e flatten norm-then-avg", c_na_cw, t_na_cw)
    auroc_3e_an = check_extreme("3e flatten avg-then-norm", c_an_cw, t_an_cw)
    print(f"    flatten normalize_then_average: AUROC={auroc_3e_na:.4f}")
    print(f"    flatten average_then_normalize: AUROC={auroc_3e_an:.4f}")
    out["3e_content_words"] = {
        "example_mask": {"task_id": int(clean[0]["task_id"]), "desc": clean[0]["task_description"],
                          "n_kept": int(m0.sum()), "n_total": int(len(m0))},
        "per_layer_auroc": auroc_3e_pl.tolist(),
        "flatten_normalize_then_average": auroc_3e_na,
        "flatten_average_then_normalize": auroc_3e_an,
    }

    # =========================== 3f: BOS + sentinel combined query ===========================
    print("\n[*] === 3f: BOS + desc + sentinel combined query (re-run 3c) ===")
    def bos_desc_sentinel_rows(ep):
        t2i_avg = ep["t2i_perhead"].mean(axis=1)  # [L, 1+n_txt, P]
        bos_row = t2i_avg[:, 0:1, :]              # [L,1,P]
        desc = desc_rows(ep)                       # [L,n_desc,P]
        sentinel_row = t2i_avg[:, ep["n_txt"]:ep["n_txt"] + 1, :]  # [L,1,P] -- index n_txt = last text row (1+n_txt-1)
        return np.concatenate([bos_row, desc, sentinel_row], axis=1)

    c_na_bs = [flatten_normalize_then_average(bos_desc_sentinel_rows(e)) for e in clean]
    t_na_bs = [flatten_normalize_then_average(bos_desc_sentinel_rows(e)) for e in trig]
    c_an_bs = [flatten_average_then_normalize(bos_desc_sentinel_rows(e)) for e in clean]
    t_an_bs = [flatten_average_then_normalize(bos_desc_sentinel_rows(e)) for e in trig]
    auroc_3f_na = check_extreme("3f flatten norm-then-avg", c_na_bs, t_na_bs)
    auroc_3f_an = check_extreme("3f flatten avg-then-norm", c_an_bs, t_an_bs)
    print(f"    normalize_then_average: AUROC={auroc_3f_na:.4f}")
    print(f"    average_then_normalize: AUROC={auroc_3f_an:.4f}")
    out["3f_bos_desc_sentinel"] = {
        "normalize_then_average": auroc_3f_na, "average_then_normalize": auroc_3f_an,
    }

    # =========================== 3g/3h/3i: img2img ===========================
    print("\n[*] === 3g/3h/3i: img2img (primary camera) ===")
    clean_i2i = {(e["task_id"], e["seed"]): e["img2img_primary_raw"] for e in clean}
    trig_i2i = {(e["task_id"], e["seed"]): e["img2img_primary_raw"] for e in trig}
    clean_i2i_pl = np.stack([per_layer_ftt(v) for v in clean_i2i.values()])
    trig_i2i_pl = np.stack([per_layer_ftt(v) for v in trig_i2i.values()])
    auroc_3g = np.array([check_extreme(f"img2img layer {l}", clean_i2i_pl[:, l], trig_i2i_pl[:, l]) for l in range(n_layers)])
    for l in range(n_layers):
        print(f"    layer {l:2d}: AUROC={auroc_3g[l]:.4f}")
    out["3g_img2img_per_layer_auroc"] = auroc_3g.tolist()
    out["3g_best_layer"] = int(np.argmax(auroc_3g))
    out["3g_best_layer_auroc"] = float(auroc_3g.max())

    c_i2i_na = [flatten_normalize_then_average(v) for v in clean_i2i.values()]
    t_i2i_na = [flatten_normalize_then_average(v) for v in trig_i2i.values()]
    c_i2i_an = [flatten_average_then_normalize(v) for v in clean_i2i.values()]
    t_i2i_an = [flatten_average_then_normalize(v) for v in trig_i2i.values()]
    auroc_3h_na = check_extreme("3h flatten norm-then-avg", c_i2i_na, t_i2i_na)
    auroc_3h_an = check_extreme("3h flatten avg-then-norm", c_i2i_an, t_i2i_an)
    print(f"    3h Flatten: norm-then-avg={auroc_3h_na:.4f}  avg-then-norm={auroc_3h_an:.4f}")
    out["3h_img2img_flatten"] = {"normalize_then_average": auroc_3h_na, "average_then_normalize": auroc_3h_an}

    c_i2i_sna = [sharedref_normalize_then_average(v) for v in clean_i2i.values()]
    t_i2i_sna = [sharedref_normalize_then_average(v) for v in trig_i2i.values()]
    c_i2i_san = [sharedref_average_then_normalize(v) for v in clean_i2i.values()]
    t_i2i_san = [sharedref_average_then_normalize(v) for v in trig_i2i.values()]
    auroc_3i_na = check_extreme("3i sharedref norm-then-avg", c_i2i_sna, t_i2i_sna)
    auroc_3i_an = check_extreme("3i sharedref avg-then-norm", c_i2i_san, t_i2i_san)
    print(f"    3i Shared-ref: norm-then-avg={auroc_3i_na:.4f}  avg-then-norm={auroc_3i_an:.4f}")
    out["3i_img2img_sharedref"] = {"normalize_then_average": auroc_3i_na, "average_then_normalize": auroc_3i_an}

    # =========================== 3j: merged combined single query ===========================
    print("\n[*] === 3j: merged (image+desc combined single query) ===")
    def merged_combined(ep):
        return np.concatenate([ep["merged_image_raw"], ep["merged_text_raw"]], axis=1)  # [L, 2P+n_desc, 2P+n_desc]

    clean_merged = {(e["task_id"], e["seed"]): merged_combined(e) for e in clean}
    trig_merged = {(e["task_id"], e["seed"]): merged_combined(e) for e in trig}
    clean_m_pl = np.stack([per_layer_ftt(v) for v in clean_merged.values()])
    trig_m_pl = np.stack([per_layer_ftt(v) for v in trig_merged.values()])
    auroc_3j_pl = np.array([check_extreme(f"merged layer {l}", clean_m_pl[:, l], trig_m_pl[:, l]) for l in range(n_layers)])
    for l in range(n_layers):
        print(f"    layer {l:2d}: AUROC={auroc_3j_pl[l]:.4f}")
    out["3j_merged_per_layer_auroc"] = auroc_3j_pl.tolist()
    out["3j_best_layer"] = int(np.argmax(auroc_3j_pl))
    out["3j_best_layer_auroc"] = float(auroc_3j_pl.max())

    c_m_na = [flatten_normalize_then_average(v) for v in clean_merged.values()]
    t_m_na = [flatten_normalize_then_average(v) for v in trig_merged.values()]
    c_m_an = [flatten_average_then_normalize(v) for v in clean_merged.values()]
    t_m_an = [flatten_average_then_normalize(v) for v in trig_merged.values()]
    auroc_3j_flat_na = check_extreme("3j flatten norm-then-avg", c_m_na, t_m_na)
    auroc_3j_flat_an = check_extreme("3j flatten avg-then-norm", c_m_an, t_m_an)
    print(f"    Flatten: norm-then-avg={auroc_3j_flat_na:.4f}  avg-then-norm={auroc_3j_flat_an:.4f}")

    c_m_sna = [sharedref_normalize_then_average(v) for v in clean_merged.values()]
    t_m_sna = [sharedref_normalize_then_average(v) for v in trig_merged.values()]
    c_m_san = [sharedref_average_then_normalize(v) for v in clean_merged.values()]
    t_m_san = [sharedref_average_then_normalize(v) for v in trig_merged.values()]
    auroc_3j_sh_na = check_extreme("3j sharedref norm-then-avg", c_m_sna, t_m_sna)
    auroc_3j_sh_an = check_extreme("3j sharedref avg-then-norm", c_m_san, t_m_san)
    print(f"    Shared-ref: norm-then-avg={auroc_3j_sh_na:.4f}  avg-then-norm={auroc_3j_sh_an:.4f}")
    out["3j_merged_flatten"] = {"normalize_then_average": auroc_3j_flat_na, "average_then_normalize": auroc_3j_flat_an}
    out["3j_merged_sharedref"] = {"normalize_then_average": auroc_3j_sh_na, "average_then_normalize": auroc_3j_sh_an}

    out_path = OUT_DIR / "ftt_battery3_backdoorvla_oft_full.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
