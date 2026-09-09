import sys, json
sys.path.insert(0, "/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense")
import numpy as np
from detectors.schema import load_dir
from detectors.ftt import auroc

samples = load_dir("/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/results/goba_extracted_img2img_and_merged")
print(f"loaded {len(samples)} samples")

for role in ("attack", "clean_baseline"):
    subset = [s for s in samples if s.extra.get("role") == role]
    labels = np.array([s.label for s in subset])
    n_clean = int((labels == 0).sum())
    n_trig = int((labels == 1).sum())
    print(f"\n=== role={role}  n_clean={n_clean} n_trig={n_trig} ===")

    # confound check: query token counts
    nq_clean = sorted({s.extra["n_query_tokens_desc"] for s in subset if s.label == 0})
    nq_trig = sorted({s.extra["n_query_tokens_desc"] for s in subset if s.label == 1})
    print(f"n_query_tokens_desc: clean={nq_clean} trigger={nq_trig}")

    # img2img per layer
    per_layer = np.array([s.extra["img2img_ftt_per_layer"] for s in subset])  # [N, L]
    n_layers = per_layer.shape[1]
    layer_aurocs = []
    for l in range(n_layers):
        clean_scores = per_layer[labels == 0, l]
        trig_scores = per_layer[labels == 1, l]
        layer_aurocs.append(auroc(clean_scores, trig_scores))
    best_layer = int(np.nanargmax(layer_aurocs))
    print(f"img2img per-layer AUROC (best layer {best_layer} = {layer_aurocs[best_layer]:.4f}):")
    for l, a in enumerate(layer_aurocs):
        marker = "  <== best" if l == best_layer else ""
        print(f"  layer {l:2d}: {a:.4f}{marker}")

    # merged, last layer
    merged_combined = np.array([s.extra["merged_combined_ftt"] for s in subset])
    merged_text = np.array([s.extra["merged_text_group_ftt"] for s in subset])
    merged_image = np.array([s.extra["merged_image_group_ftt"] for s in subset])
    auroc_combined = auroc(merged_combined[labels == 0], merged_combined[labels == 1])
    auroc_text = auroc(merged_text[labels == 0], merged_text[labels == 1])
    auroc_image = auroc(merged_image[labels == 0], merged_image[labels == 1])
    print(f"\nmerged FTT (last layer): combined AUROC={auroc_combined:.4f}  "
          f"text-group AUROC={auroc_text:.4f}  image-group AUROC={auroc_image:.4f}")

    out = {
        "role": role, "n_clean": n_clean, "n_trig": n_trig,
        "n_query_tokens_desc_clean": nq_clean, "n_query_tokens_desc_trigger": nq_trig,
        "img2img_per_layer_auroc": layer_aurocs,
        "img2img_best_layer": best_layer,
        "img2img_best_layer_auroc": layer_aurocs[best_layer],
        "merged_combined_auroc": auroc_combined,
        "merged_text_group_auroc": auroc_text,
        "merged_image_group_auroc": auroc_image,
    }
    outpath = (f"/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense/results/"
               f"goba_img2img_merged_auroc_libero_object_{role}.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {outpath}")
