# VLA Backdoor Defense

Detects backdoor attacks on vision-language-action (VLA) policies via the
**FTT** (Frobenius-norm Threshold / assimilation) statistic: for a set of text
tokens attending over image patches, a backdoor trigger drags every text
token's attention toward the same pattern ("Assimilation Phenomenon", ported
from T2IShield [ECCV'24]). Low deviation from the mean attention row ⟺
backdoor active.

Built for ICRA: the goal is a codebase that stays clean as attacks are added,
not one perfected for a single attack.

## Architecture: extraction is split from detection

```
detectors/        pure numpy/scipy/sklearn. NEVER imports torch or an
                   attack's code. Unit-testable with synthetic arrays, no
                   GPU, no conda gymnastics.
  schema.py          the ONE contract: what an extractor must produce
  ftt.py             the FTT statistic + AUROC (both polarities)

adapters/<attack>/   the ONLY place that attack's code is touched. Each
                     adapter is a standalone script, run inside that attack's
                     own conda env / repo, that:
                       1. loads the checkpoint using THAT ATTACK'S OWN
                          model-loading code (imported live, never copied)
                       2. renders/builds clean + triggered observations using
                          that attack's own trigger mechanism
                       3. runs one forward pass, slices out the
                          text-attends-to-image attention rows
                       4. saves each sample as one `.npz` via
                          detectors/schema.ExtractedSample
  badvla/extract_text2img_ftt.py
  goba/extract_text2img_ftt.py

runners/
  run_detector.py    loads a directory of .npz artifacts (from ANY attack),
                     computes FTT + AUROC, prints/saves a comparison table.
                     Never touches an attack's code or a GPU.

results/             one .npz per (attack, checkpoint, task, condition);
                     one .json summary per run_detector.py invocation.
```

**Why this split, specifically:** BadVLA and GoBA ship *different, mutually
incompatible* `prismatic` packages at the same import path (`import
prismatic...`) — mixing them in one Python process silently imports the wrong
one. Any future attack will likely bring its own fork too. Splitting
extraction (attack-specific, one conda env each) from detection (attack-
agnostic, one shared env, no GPU needed) means:

- adding attack #3 costs one new adapter script; zero changes to `detectors/`
- a reviewer can rerun your detection numbers from the saved `.npz` artifacts
  without installing BadVLA + GoBA + LIBERO at all
- detector bugs are found/fixed once, not per-attack
- `run_detector.py` produces the *same columns* for every attack automatically
  — that's the actual "generalizes across attacks" claim for the paper

## Contract (`detectors/schema.py`)

```python
ExtractedSample(
    attn_text_image,   # (n_text_tokens, n_image_tokens) raw post-softmax attn,
                       # head-averaged, ONE chosen layer. Each row = one text
                       # token's attention over image-patch columns.
    label,             # 0 = clean, 1 = triggered
    attack,            # "badvla" | "goba" | ...
    checkpoint,
    trigger_type,
    task_id, seed, layer,
    n_cameras, patches_per_camera,
    extra={"role": "attack" | "clean_baseline"},
)
```

Adapters produce this. Detectors only ever consume it.

## Pitfalls already paid for (don't rediscover these)

- **Concurrent simulator envs corrupt rendering.** Holding two
  `OffScreenRenderEnv` instances open at once corrupts the FIRST one's
  lighting (measured: a real 1.1% trigger-object pixel difference becomes a
  spurious ~79% whole-frame difference). Rendering is deterministic given the
  seed, so every extractor collects all-clean-then-all-trigger, never
  interleaved. `goba/extract_text2img_ftt.py` checks this at runtime and warns
  if a pair differs by >15%.
- **Mahalanobis-style references need N ≫ feature-dim.** Not used by FTT
  (it's calibration-free), but relevant if FBL/AFM get added later.
- **Checkpoint identity is not obvious from its path.** A checkpoint named
  `libero_goal_no_noops` can be either the clean or the poisoned model
  depending on the attack's own dataset naming — always verify against the
  attack's own training/campaign logs, never assume from the directory name.
- **A GPU calibration/reference tensor computed on CPU will silently fail (or
  worse, silently mismatch) against GPU-resident data later** — keep device
  placement explicit at the one point data crosses the boundary.

## Running today's two attacks

```bash
# 1. BadVLA extraction (its own env; OFT checkpoint, 2 cameras + proprio)
conda activate <BadVLA's env>
cd /home/grads/nsamptur/vla_bkd_def/BadVLA
python ../vla-backdoor-defense/adapters/badvla/extract_text2img_ftt.py \
    --checkpoint vla-scripts/goal_block/trigger_sec/goal_block_stage2_30000_chkpt \
    --role attack --out-dir ../vla-backdoor-defense/results/badvla_extracted
python ../vla-backdoor-defense/adapters/badvla/extract_text2img_ftt.py \
    --checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
    --role clean_baseline --out-dir ../vla-backdoor-defense/results/badvla_extracted

# 2. GoBA extraction (its own env; physical toxic-box trigger)
conda activate GoBA-OpenVLA
cd /home/grads/nsamptur/vla_bkd_def/GoBA_attack
python ../vla-backdoor-defense/adapters/goba/extract_text2img_ftt.py \
    --checkpoint exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
    --role attack --out-dir ../vla-backdoor-defense/results/goba_extracted
python ../vla-backdoor-defense/adapters/goba/extract_text2img_ftt.py \
    --checkpoint openvla/openvla-7b-finetuned-libero-goal \
    --role clean_baseline --out-dir ../vla-backdoor-defense/results/goba_extracted

# 3. Detection (attack-agnostic, no GPU, no attack repos needed)
cd /home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense
python runners/run_detector.py --samples-dir results/badvla_extracted --out results/ftt_badvla.json
python runners/run_detector.py --samples-dir results/goba_extracted  --out results/ftt_goba.json
```

## Adding attack #3

1. `mkdir adapters/<new_attack>`
2. Write `extract_text2img_ftt.py` that imports the new attack's own
   model-loading + trigger code (never copy it), runs one forward pass with
   `output_attentions=True`, slices text-rows × image-columns, saves via
   `ExtractedSample.save()`.
3. `python runners/run_detector.py --samples-dir results/<new_attack>_extracted`
   — no other file in this repo changes.
