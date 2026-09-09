#!/usr/bin/env python
"""DropVLA equivalent of the GoBA report's section 4: content-words-only
text2img FTT, per camera. No new extraction needed -- reads the already-
extracted results/dropvla_text2img_layerwise_percam/*.npz and re-tokenizes
each task's own description text (10 unique descriptions for
libero_spatial) to build the content-word mask, exactly as
EXPERIMENTAL_score_goba_content_words_only.py did for GoBA.

Word-boundary classification (not text matching) -- same fix as GoBA's
"ketchup" edge case -- verified by hand for all 10 libero_spatial
descriptions before writing this: no subword fragment here collides with
a stopword's text (e.g. "bowl"->['bow','l'], "ramekin"->['r','ame','kin'],
"drawer"->['dra','wer'], "stove"->['st','ove'] -- none of those pieces
read as "up"/"the"/"on"/etc.).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "dropvla_text2img_layerwise_percam"
EPS = 1e-12

STOPWORDS = {
    "up", "the", "and", "it", "in", "a", "an", "to", "of", "on", "at",
    "with", "into", "from", "by", "as", "is", "are", "this", "that", "for",
}


def _desc_word_spans(prompt: str, desc: str) -> list[tuple[int, int, str]]:
    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    assert char_start != -1
    words = desc_lower.split()
    spans = []
    pos = 0
    for w in words:
        idx = desc_lower.find(w, pos)
        spans.append((idx + char_start, idx + len(w) + char_start, w))
        pos = idx + len(w)
    return spans


def content_word_mask(tokenizer, prompt: str, desc: str) -> tuple[list[int], np.ndarray]:
    desc_lower = desc.lower()
    char_start = prompt.find(desc_lower)
    char_end = char_start + len(desc_lower)
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
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


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def ftt_on_rows(rows: np.ndarray) -> float:
    ref = rows.mean(axis=0)
    return float(np.linalg.norm(rows - ref[None, :], axis=1).mean())


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


def main():
    import sys
    sys.path.insert(0, "/home/grads/nsamptur/vla_bkd_def/DropVLA")
    from experiments.robot.openvla_utils import get_processor

    class Cfg:
        pretrained_checkpoint = ("/home/grads/nsamptur/vla_bkd_def/DropVLA/runs/"
                                  "openvla-7b+libero_spatial_no_noops_v5p00carefully+b8+lr-0.0003+lora-r32+dropout-0.0--seed42--paper")

    processor = get_processor(Cfg())
    tokenizer = processor.tokenizer

    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")

    mask_cache: dict[int, np.ndarray] = {}
    results = {}

    for cam_key, cam_label in [("attn_primary_layers", "primary"), ("attn_wrist_layers", "wrist")]:
        clean_pl, trig_pl = {}, {}
        clean_na, trig_na = {}, {}
        clean_an, trig_an = {}, {}
        n_kept_report = None

        for fp in files:
            d = np.load(fp, allow_pickle=True)
            if str(d["role"]) != "attack":
                continue
            task_id = int(d["task_id"])
            desc = str(d["task_description"])
            layers = d[cam_key].astype(np.float64)  # [32, n_txt, num_patches]
            n_layers, n_txt_stored, n_patches = layers.shape

            if task_id not in mask_cache:
                prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
                txt_rel, mask = content_word_mask(tokenizer, prompt, desc)
                assert len(txt_rel) == n_txt_stored, (task_id, len(txt_rel), n_txt_stored)
                mask_cache[task_id] = mask

            mask = mask_cache[task_id]
            if n_kept_report is None:
                n_kept_report = (task_id, desc, int(mask.sum()), len(mask))
            key = (task_id, int(d["seed"]))
            label = int(d["label"])

            normed = row_normalize(layers)
            kept_normed = normed[:, mask, :]
            per_layer_scores = np.array([ftt_on_rows(kept_normed[l]) for l in range(n_layers)])
            (clean_pl if label == 0 else trig_pl)[key] = per_layer_scores

            avg_map_normed = kept_normed.mean(axis=0)
            (clean_na if label == 0 else trig_na)[key] = ftt_on_rows(avg_map_normed)

            kept_raw = layers[:, mask, :]
            avg_map_raw = row_normalize(kept_raw.mean(axis=0))
            (clean_an if label == 0 else trig_an)[key] = ftt_on_rows(avg_map_raw)

        tid, desc, n_kept, n_total = n_kept_report
        print(f"[*] {cam_label}: example mask task {tid} ({desc!r}): {n_kept}/{n_total} kept")

        clean_pl_stack = np.stack(list(clean_pl.values()))
        trig_pl_stack = np.stack(list(trig_pl.values()))
        n_layers = clean_pl_stack.shape[1]
        auroc_per_layer = np.array([rank_auroc(clean_pl_stack[:, l], trig_pl_stack[:, l]) for l in range(n_layers)])
        best_l = int(np.argmax(auroc_per_layer))

        auroc_na = rank_auroc(list(clean_na.values()), list(trig_na.values()))
        auroc_an = rank_auroc(list(clean_an.values()), list(trig_an.values()))

        print(f"    per-layer best layer {best_l}: {auroc_per_layer[best_l]:.4f}")
        print(f"    Flatten normalize-then-average: {auroc_na:.4f}")
        print(f"    Flatten average-then-normalize: {auroc_an:.4f}")

        results[cam_label] = {
            "mask_example": {"task_id": tid, "desc": desc, "n_kept": n_kept, "n_total": n_total},
            "per_layer_auroc": auroc_per_layer.tolist(),
            "best_layer": best_l, "best_layer_auroc": float(auroc_per_layer[best_l]),
            "flatten_normalize_then_average": auroc_na,
            "flatten_average_then_normalize": auroc_an,
        }

    out_path = REPO / "results" / "ftt_dropvla_content_words_percam.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
