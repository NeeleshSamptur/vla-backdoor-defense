#!/usr/bin/env python
"""ONE-OFF, ISOLATED experiment -- content-words-only text2img FTT for GoBA.

Idea: row-normalization is unaffected by which tokens are kept (it's a
per-row operation, independent of other rows), so stop words stay in the
sequence for that step exactly as before. AFTER normalization, drop the
STOP-WORD rows (function words with no task-specific meaning: "up", "the",
"and", "it", "in", ...) and keep only CONTENT-WORD rows (the verbs/nouns
that actually name the task: "pick", "place", object names, "basket") when
computing the FTT reference (row-mean) and the row-to-reference distances.

TOKEN-BOUNDARY CORRECTNESS (verified empirically before writing this):
object names split into multiple BPE/sentencepiece pieces for several GoBA
libero_object tasks (e.g. "ketchup" -> ['k','etch','up'], "bbq sauce" ->
['b','b','q','sau','ce']). Critically, task 4's "ketchup" produces a
subword token that DECODES to the literal string "up" -- identical text to
the real stopword "up" that appears earlier in every one of these
descriptions ("pick UP the ..."). A naive per-token string match against a
stopword list would therefore wrongly drop 1/3 of "ketchup"'s tokens.
Fixed by classifying at the WORD level instead: split the description into
words, locate each word's own character span in the real prompt, then
classify every TOKEN by which whole word its character span falls inside
-- not by the token's own decoded text. Verified directly: for "pick up
the ketchup and place it in the basket", the standalone "up" (right after
"pick") is dropped, while all three tokens belonging to "ketchup"
(including the one that also decodes as "up") are kept. See this session's
verification transcript for the full per-token printout across all 10
libero_object task descriptions.

ISOLATION: this file is intentionally separate from every other scoring
script in analysis/ and does not import from or modify any of them, or
detectors/ftt.py, or any extraction script. Nothing here changes the
behavior of any other experiment -- deleting this one file fully reverts
this exploration with no other trace in the pipeline.

Reads results/goba_text2img_layerwise_causal_fixed/*.npz (role=='attack'
only) for the attention data, and re-tokenizes each task's own description
text (10 unique descriptions for libero_object) to build the content-word
mask -- no GPU / model forward pass needed for that part, only the
tokenizer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "goba_text2img_layerwise_causal_fixed"
BALANCED_63V63_JSON = REPO / "results" / "ftt_goba_text2img_layerwise_63v63_balanced.json"
EPS = 1e-12

# Function words with no task-specific meaning. Deliberately small and
# specific to these short imperative LIBERO descriptions ("pick up the X
# and place it in the Y") rather than a generic NLP stopword list -- every
# word actually appearing across GoBA's libero_object descriptions was
# checked by hand (see the verification printout).
STOPWORDS = {
    "up", "the", "and", "it", "in", "a", "an", "to", "of", "on", "at",
    "with", "into", "from", "by", "as", "is", "are", "this", "that", "for",
}


def _desc_word_spans(prompt: str, desc: str) -> list[tuple[int, int, str]]:
    """Word-level (start, end, word) spans, in PROMPT-absolute character
    coordinates, found by sequential search so repeated words (e.g. "the"
    appearing twice) each get their own real span instead of all matching
    the first occurrence."""
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


def content_word_mask(tokenizer, prompt: str, desc: str) -> tuple[list[int], np.ndarray]:
    """Returns (txt_rel, mask) where txt_rel is the same description-token
    index list _desc_token_row_indices produces (attn_text_image_layers's
    token axis is in this exact order), and mask[i] is True iff txt_rel[i]'s
    token falls inside a CONTENT word (not a stopword), determined by
    maximum character-span overlap against word-level spans -- never by
    matching the token's own decoded text, which is exactly the trap that
    would misclassify "ketchup"'s "up" subword fragment as the stopword "up".
    Caller checks len(txt_rel) against the attention array's own stored
    token-axis length.
    """
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
    """Standard FTT: reference = row-mean of the given (already selected)
    rows, distance = each row to that reference, averaged."""
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
    sys.path.insert(0, "/home/grads/nsamptur/vla_bkd_def/GoBA_attack")
    sys.path.insert(0, str(REPO / "adapters" / "goba"))
    from extract_text2img_ftt import Cfg
    from experiments.robot.openvla_utils import get_processor

    CKPT = ("/home/grads/nsamptur/vla_bkd_def/GoBA_attack/exp/"
             "openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug")
    cfg = Cfg(pretrained_checkpoint=CKPT, unnorm_key="libero_object")
    processor = get_processor(cfg)  # tokenizer + image processor only, no GPU/model weights loaded
    tokenizer = processor.tokenizer

    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"[*] {len(files)} files in {DATA_DIR}")

    with open(BALANCED_63V63_JSON) as f:
        balanced = json.load(f)
    clean_63_keys = {(tid, seed) for tid, seed in balanced["chosen_clean_pairs"]}
    trig_63_keys = {(tid, seed) for tid, seed in balanced["trigger_pairs"]}

    mask_cache: dict[int, np.ndarray] = {}

    clean_per_layer, trig_per_layer = {}, {}          # key -> [32] array
    clean_norm_then_avg, trig_norm_then_avg = {}, {}  # key -> scalar
    clean_avg_then_norm, trig_avg_then_norm = {}, {}  # key -> scalar

    n_kept_report = None

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        if meta["extra"]["role"] != "attack":
            continue
        task_id = meta["task_id"]
        desc = meta["extra"]["task_description"]
        layers = d["attn_text_image_layers"].astype(np.float64)  # [32, n_txt, n_patches]
        n_layers, n_txt_stored, n_patches = layers.shape

        if task_id not in mask_cache:
            prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
            txt_rel, mask = content_word_mask(tokenizer, prompt, desc)
            assert len(txt_rel) == n_txt_stored, (
                f"task {task_id}: txt_rel len {len(txt_rel)} != stored token axis {n_txt_stored}")
            mask_cache[task_id] = mask
            if n_kept_report is None:
                n_kept_report = (task_id, desc, mask.sum(), len(mask))

        mask = mask_cache[task_id]
        key = (task_id, meta["seed"])
        bucket_pl = clean_per_layer if meta["label"] == 0 else trig_per_layer
        bucket_na = clean_norm_then_avg if meta["label"] == 0 else trig_norm_then_avg
        bucket_an = clean_avg_then_norm if meta["label"] == 0 else trig_avg_then_norm

        # (a) per-layer FTT, content-words-only rows
        normed = row_normalize(layers)              # [32, n_txt, P] -- normalize BEFORE dropping rows
        kept_normed = normed[:, mask, :]             # drop stopword rows AFTER normalization
        per_layer_scores = np.array([ftt_on_rows(kept_normed[l]) for l in range(n_layers)])
        bucket_pl[key] = per_layer_scores

        # (b1) normalize-then-average (flatten), content-words-only
        avg_map_normed = kept_normed.mean(axis=0)    # [n_kept, P]
        bucket_na[key] = ftt_on_rows(avg_map_normed)

        # (b2) average-then-normalize (flatten), content-words-only
        kept_raw = layers[:, mask, :]
        avg_map_raw = kept_raw.mean(axis=0)          # [n_kept, P]
        avg_map_raw_normed = row_normalize(avg_map_raw)
        bucket_an[key] = ftt_on_rows(avg_map_raw_normed)

    tid, desc, n_kept, n_total = n_kept_report
    print(f"[*] example mask: task {tid} ({desc!r}): {n_kept}/{n_total} tokens kept as content words")

    assert len(clean_per_layer) == 100 and len(trig_per_layer) == 100

    # --- (a) per-layer AUROC, content-words-only ---
    clean_pl_stack = np.stack(list(clean_per_layer.values()))  # [100, 32]
    trig_pl_stack = np.stack(list(trig_per_layer.values()))
    auroc_per_layer_100 = np.array([
        rank_auroc(clean_pl_stack[:, l], trig_pl_stack[:, l]) for l in range(32)])

    c63_pl = np.stack([clean_per_layer[k] for k in sorted(clean_63_keys)])
    t63_pl = np.stack([trig_per_layer[k] for k in sorted(trig_63_keys)])
    auroc_per_layer_63 = np.array([
        rank_auroc(c63_pl[:, l], t63_pl[:, l]) for l in range(32)])

    print("\n[*] Per-layer AUROC, content-words-only rows:")
    for l in range(32):
        print(f"    layer {l:2d}: 100v100={auroc_per_layer_100[l]:.4f}  63v63={auroc_per_layer_63[l]:.4f}")

    # --- (b) the two flatten variants, content-words-only ---
    c100_na = list(clean_norm_then_avg.values()); t100_na = list(trig_norm_then_avg.values())
    c100_an = list(clean_avg_then_norm.values()); t100_an = list(trig_avg_then_norm.values())
    c63_na = [clean_norm_then_avg[k] for k in sorted(clean_63_keys)]
    t63_na = [trig_norm_then_avg[k] for k in sorted(trig_63_keys)]
    c63_an = [clean_avg_then_norm[k] for k in sorted(clean_63_keys)]
    t63_an = [trig_avg_then_norm[k] for k in sorted(trig_63_keys)]

    auroc_na_100 = rank_auroc(c100_na, t100_na)
    auroc_na_63 = rank_auroc(c63_na, t63_na)
    auroc_an_100 = rank_auroc(c100_an, t100_an)
    auroc_an_63 = rank_auroc(c63_an, t63_an)

    print(f"\n[*] Flatten, normalize-then-average, content-words-only: 100v100={auroc_na_100:.4f}  63v63={auroc_na_63:.4f}")
    print(f"[*] Flatten, average-then-normalize, content-words-only: 100v100={auroc_an_100:.4f}  63v63={auroc_an_63:.4f}")

    out = {
        "content_word_mask_example": {"task_id": tid, "desc": desc, "n_kept": int(n_kept), "n_total": int(n_total)},
        "per_layer_100v100": auroc_per_layer_100.tolist(),
        "per_layer_63v63": auroc_per_layer_63.tolist(),
        "flatten_normalize_then_average": {"auroc_100v100": auroc_na_100, "auroc_63v63": auroc_na_63},
        "flatten_average_then_normalize": {"auroc_100v100": auroc_an_100, "auroc_63v63": auroc_an_63},
    }
    out_path = REPO / "results" / "EXPERIMENTAL_ftt_goba_content_words_only.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] saved -> {out_path}")


if __name__ == "__main__":
    main()
