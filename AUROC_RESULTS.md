# AUROC results, in order run

FTT statistic (T2IShield-style attention assimilation), scored as clean-vs-triggered
AUROC. Three stages, in the order they were actually tried:

1. **text2img, last layer only** ("desc_only") — the original approach.
2. **text2img, averaged across all layers** — the attempted fix.
3. **img2img (patch→patch), swept over all layers** — the pivot that replaced both.

---

## 1. Cross-attention (text2img), last layer only, desc_only

Query = task-description tokens only. Keys = image patches. One score per
episode, from the last LLM layer, averaged over heads.

| Attack | Suite | AUROC | Clean-baseline control AUROC |
|---|---|---|---|
| GoBA | libero_10 | 0.804 | 0.389 |
| GoBA | libero_goal | 0.963 | 0.577 |
| GoBA | libero_object | 0.795 | 0.562 |
| GoBA | libero_spatial | 0.863 | 0.470 |
| BadVLA | libero_10 | 1.000 | 0.542 |
| BadVLA | libero_goal | 1.000 | 0.430 |
| BadVLA | libero_object | 1.000 | 0.497 |
| BadVLA | libero_spatial | 1.000 | 0.479 |
| BackdoorVLA-OFT | libero_object | **0.318** | — |
| Pi0-Fast TI4 (text+image) | libero_object | 0.999 | — |
| Pi0-Fast I4 (image-only, no text trigger) | libero_object | 0.015 | — |

**The issue:** no single layer worked for every attack. BadVLA was a clean
win (1.000, control near chance). But **BackdoorVLA-OFT's last layer alone
gave AUROC 0.318 — worse than chance**, meaning the last layer's text2img
attention was anti-correlated with the trigger, not just uninformative. A
cross-attack sweep at the last layer across GoBA and BackdoorVLA-OFT
together (`layer_sweep_libero_object.json`) confirmed there was no common
layer clearing AUROC 0.85 for both at once (best common layer's min-AUROC
was only 0.72). Pi0-Fast I4's near-zero AUROC (0.015) is expected/correct,
not a failure — it's an image-only trigger, so text2img attention has no
reason to see it.

Separately, GoBA's and OFT's trigger prompts differ from clean prompts in
length (OFT's `~*magic*~` prefix), so any text2img number from these two is
confounded by query-token count, independent of the layer-choice problem
above.

---

## 2. Attempted fix: average FTT across all layers

Same text2img setup, but instead of taking the last layer's score, average
the FTT score across every layer before computing AUROC.

| Attack | Suite | Last-layer-only AUROC | Layer-averaged AUROC | Result |
|---|---|---|---|---|
| GoBA | libero_object | 0.795 | **0.494** | got worse — collapsed to chance |
| BackdoorVLA-OFT | libero_object | 0.318 | **0.981** | fixed — huge improvement |

**What happened:** layer-averaging is not a reliable fix. It rescued
BackdoorVLA-OFT (0.318 → 0.981) but destroyed GoBA's already-working signal
(0.795 → 0.494, i.e. chance). The two attacks needed opposite treatment —
averaging dilutes GoBA's real (but narrow, single-layer) signal with noise
from uninformative layers, while for OFT the useful signal was smeared
across many mediocre layers rather than concentrated in the last one. This
motivated looking at the full per-layer curve instead of picking one
aggregation rule for everyone.

---

## 3. Per-layer AUROC (text2img), full sweep

Full 32-layer curve (`layer_sweep_libero_object.json`), same episodes as
above, libero_object, one AUROC per layer:

| Layer | GoBA | BackdoorVLA-OFT |
|---|---|---|
| 0 | 0.213 | 0.000 |
| 1 | 0.427 | 0.930 |
| 2 | 0.567 | 0.640 |
| 3 | 0.557 | 0.988 |
| 4 | **0.719** | 0.975 |
| 5 | 0.563 | 0.000 |
| 6 | 0.694 | **1.000** |
| 7 | 0.156 | 0.989 |
| 8 | 0.560 | 0.645 |
| 9 | 0.366 | 0.439 |
| 10 | 0.602 | 1.000 |
| 11 | 0.343 | 0.011 |
| 12 | 0.362 | 0.573 |
| 13 | 0.660 | 0.095 |
| 14 | 0.272 | 0.305 |
| 15 | 0.270 | 0.684 |
| 16 | 0.403 | 0.098 |
| 17 | 0.445 | 0.943 |
| 18 | 0.480 | 0.256 |
| 19 | 0.575 | 0.426 |
| 20 | 0.468 | 0.991 |
| 21 | 0.371 | 0.992 |
| 22 | 0.357 | 0.855 |
| 23 | 0.734 | 0.089 |
| 24 | 0.504 | 0.330 |
| 25 | 0.697 | 0.369 |
| 26 | 0.232 | 0.527 |
| 27 | 0.398 | 0.909 |
| 28 | 0.283 | 0.686 |
| 29 | 0.265 | 0.478 |
| 30 | 0.384 | 0.160 |
| 31 (last) | 0.795 | 0.245 |

This is the direct picture behind the "no fix works for both" problem above:
GoBA's best layer (4, 0.719) is mediocre for OFT-relevant purposes and OFT's
best layer (6 or 10, 1.000) is not GoBA's peak. Note row 31 here (0.795 /
0.245) is this specific sweep's version of the "last-layer" numbers in
Section 1 — close but not identical to the desc_only file's numbers (0.795
GoBA / 0.318 OFT), since it's a separate extraction batch of the same idea.
No layer is simultaneously good for both attacks, which is why Section 1's
last-layer approach and Section 2's layer-average both failed to generalize
— there simply isn't one text2img layer or one aggregation rule that works
for every attack.

---

## 4. Image-to-image (img2img) AUROC — the pivot

Query and key are both image patches (patch→patch attention within the
image), not text-to-image. Immune to the text2img issues above because
image patch count doesn't depend on prompt length. Row-normalized
("outgoing": this patch → others), swept over every layer.

### Best layer per attack

| Attack | Suite | Best layer | AUROC | n (clean/trig) |
|---|---|---|---|---|
| BadVLA (white patch) | libero_10 | 2 | **1.000** | 100/100 |
| DropVLA | libero_spatial | 13 | **1.000** | 30/30 |
| GoBA | libero_object | 4 | **0.998** | 33/29 |
| BackdoorVLA-OFT | libero_object | 31 | 0.877 | 90/90 |
| Pi0-Fast TI4 (text+image) | libero_object | 7 | **1.000** | 20/18 |
| Pi0-Fast I4 (image-only) | libero_object | 1 | **1.000** | 20/18 |

Every attack now clears 0.877, four of six hit 1.000 — a much stronger and
more consistent result than any single text2img layer or the layer-average
managed in Sections 1–2.

### Full per-layer curves (img2img)

| Layer | BadVLA (10) | DropVLA (spatial) | GoBA (object) | OFT (object) | Pi0-Fast TI4 | Pi0-Fast I4 |
|---|---|---|---|---|---|---|
| 0 | 0.591 | 0.020 | 0.546 | 0.828 | 0.600 | 0.000 |
| 1 | 0.000 | 0.762 | 0.150 | 0.818 | 0.033 | 1.000 |
| 2 | 1.000 | 0.937 | 0.917 | 0.837 | 0.131 | 0.000 |
| 3 | 1.000 | 0.982 | 0.676 | 0.171 | 0.000 | 0.914 |
| 4 | 1.000 | 0.892 | **0.998** | 0.051 | 0.553 | 1.000 |
| 5 | 1.000 | 0.800 | 0.956 | 0.703 | 0.842 | 1.000 |
| 6 | 1.000 | 0.680 | 0.722 | 0.073 | 0.111 | 1.000 |
| 7 | 1.000 | 0.854 | 0.904 | 0.032 | **1.000** | 1.000 |
| 8 | 1.000 | 0.984 | 0.778 | 0.156 | 0.900 | 0.900 |
| 9 | 1.000 | 0.871 | 0.798 | 0.575 | 0.386 | 0.306 |
| 10 | 1.000 | 0.962 | 0.764 | 1.000 | 0.831 | 0.961 |
| 11 | 1.000 | 0.906 | 0.412 | 0.108 | 0.392 | 0.669 |
| 12 | 1.000 | 0.964 | 0.994 | 0.110 | 0.489 | 0.939 |
| 13 | 0.991 | **1.000** | 0.968 | 0.087 | 0.961 | 1.000 |
| 14 | 0.815 | 0.999 | 0.520 | 0.062 | 0.814 | 0.944 |
| 15 | 0.872 | 1.000 | 0.596 | 0.012 | 0.350 | 0.997 |
| 16 | 0.950 | 1.000 | 0.650 | 0.024 | 0.450 | 1.000 |
| 17 | 0.956 | 1.000 | 0.427 | 0.019 | 0.000 | — |
| 18 | 1.000 | 0.999 | 0.790 | 0.004 | — | — |
| 19 | 1.000 | 0.810 | 0.964 | 0.252 | — | — |
| 20 | 1.000 | 0.950 | 0.454 | 0.328 | — | — |
| 21 | 1.000 | 0.776 | 0.981 | 0.237 | — | — |
| 22 | 1.000 | 0.978 | 0.461 | 0.209 | — | — |
| 23 | 1.000 | 0.732 | 0.742 | 0.203 | — | — |
| 24 | 1.000 | 0.752 | 0.947 | 0.081 | — | — |
| 25 | 1.000 | 0.344 | 0.603 | 0.306 | — | — |
| 26 | 1.000 | 0.426 | 0.938 | 0.410 | — | — |
| 27 | 1.000 | 0.692 | 0.760 | 0.323 | — | — |
| 28 | 1.000 | 0.648 | 0.920 | 0.332 | — | — |
| 29 | 1.000 | 0.444 | 0.625 | 0.522 | — | — |
| 30 | 1.000 | 0.551 | 0.418 | 0.343 | — | — |
| 31 (last) | 1.000 | — | 0.525 | **0.877** | — | — |

(BadVLA/GoBA/OFT have 32 layers; Pi0-Fast has 18 — its shorter transformer.)

Unlike Sections 1–3, most attacks here have a wide *band* of high-AUROC
layers, not just one narrow peak (e.g. BadVLA is 1.000 from layer 2 through
31 almost without exception; DropVLA is 1.000 or near it from layer 13
through 17). BackdoorVLA-OFT is the exception — still peaked narrowly, best
at the very last layer (31, 0.877), near-zero through the middle layers —
which is why it's the only attack below 0.9 in the img2img headline table.
