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

## Two-stage detection cascade

Different attacks put the trigger on screen at different times, so detection
runs in two stages:

**Stage 1 — static screening** (`--mode static`). Score the initial
observation. Catches *always-on* triggers that are present from frame 0:
BadVLA's centered white block, GoBA's physical poison object. One forward
pass; if it fires, alarm before the rollout even starts.

**Stage 2 — temporal monitoring** (`--mode temporal`). If Stage 1 clears,
score every frame against **that episode's own early-frame baseline**. Catches
*delayed / conditional* triggers: DropVLA paints its red dot onto the camera
image only after the policy has grasped and lifted the object, so frame 0 is
genuinely clean and Stage 1 is correctly silent.

| | Stage 1 (frame 0) | Stage 2 (temporal) |
|---|---|---|
| clean episode | silent | silent |
| BadVLA | **alarm** | n/a — no clean prefix to normalize against |
| GoBA | **alarm** | n/a — same |
| DropVLA | correctly silent | **alarm** |

DropVLA being silent at Stage 1 is a feature, not a miss: frame 0 has no
trigger. The two stages cover different threats rather than double-counting.

### Why the temporal structure helps rather than hurts

- **Self-calibrating.** A delayed-trigger episode starts clean, so its own
  opening frames are a perfectly matched reference (same scene, lighting,
  task, camera pose). No external calibration set, no distribution-shift
  assumption.
- **Change-point, not classification.** Natural variation over a rollout
  wanders smoothly; an activation is a step that *persists*.
- **Persistence is the false-alarm killer, and the attacker can't avoid it.**
  Releasing an object takes several timesteps, so the trigger must stay
  visible. Requiring `k` consecutive elevated frames filters one-frame blips
  (motion blur, specular highlights) but cannot filter a real activation.

### Closing the early-activation hole (`detectors/cascade.py`)

Self-normalizing against the opening frames fails *silently* if the trigger
fires early: with a 5-frame window and activation at frame 2, three of the
five reference frames are poisoned, the baseline drifts toward the trigger
state, and everything after looks normal relative to it.

The fix is to stop assuming the prefix is clean and instead **verify it, on
exactly the frames that will become the baseline**:

```
Stage 1: score the first K frames against a population reference from
         known-clean rollouts.  Any frame suspicious -> ALARM, and never
         trust this baseline.
Stage 2: only if Stage 1 certifies them, adopt those K frames as the
         reference and watch for a persistent departure.
```

Because the screening window **is** the baseline window, contamination of the
baseline is precisely what Stage 1 tests for — so there is no gap:

| activation frame | first K frames | caught by |
|---|---|---|
| 0 … K−1 | contaminated | **Stage 1** (gate refuses) |
| K … N | genuinely clean | **Stage 2** (departure) |

`tests/test_cascade.py` sweeps the activation frame across the episode and
asserts every one is caught by one stage or the other. It also contains
`test_naive_baseline_DOES_fail_early_activation`, a regression test proving
the ungated rule really does miss an early activation — the hole is real, and
gating is what closes it.

Two caveats stated up front, not buried:

- **Stage 1 needs a small population reference** from clean rollouts to judge
  "is this window clean in absolute terms." That's realistic — the defender
  can run clean episodes — and it's used only for the gate, but it is a real
  external dependency that Stage 2 alone does not have.
- **Stage 2's reliability is bounded by Stage 1's false-negative rate.** A
  missed contaminated prefix still poisons the baseline. The cascade makes
  that dependency explicit and measurable rather than assumed.

### Metrics beyond AUROC

The defender doesn't know which timestep the trigger appears, so it doesn't
get to pick one either: episode score is the strongest *sustained* deviation
anywhere after the baseline window. Stage 2 reports:

| metric | what it shows |
|---|---|
| episode-level AUROC | you detect it at all |
| **detection latency** (frames from activation → alarm) | you catch it *in time* |
| false alarms per clean episode | scoring ~300 frames/episode gives ~300 chances to cry wolf |

Latency is the robotics-specific number: DropVLA's harm is the object hitting
the floor. Flagging it a few frames after activation means the arm can be
halted before release completes; flagging it 50 frames later means AUROC 1.0
bought you nothing.

`activation_frame` in the schema is an **oracle label** used only to score
latency. It is never an input to any detector — a detector that reads it is
cheating.

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
# 1. BadVLA extraction (its own env; OFT checkpoint, 2 cameras + proprio).
#    All four suites, both roles, using the exact checkpoints validated in
#    attack_model_paths.md and the exact PYTHONPATH BadVLA's own
#    run_libero_eval_local.sh uses (see extract_text2img_ftt.py's docstring
#    for why this must not silently diverge -- there is no bundled BDDL fork
#    in BadVLA; BDDL content comes purely from whichever `libero` package is
#    first on PYTHONPATH).
adapters/badvla_white_patch/run_all_suites.sh

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
# BadVLA's extractor produces multi-frame episodes (n_frames closed-loop
# passes each) -- use --mode stage1, which groups by episode_id and averages
# FTT over the leading frames. --mode static would score every frame as an
# independent sample with no per-episode averaging, silently defeating the
# point of Stage 1. GoBA's extractor is still single-frame (no closed-loop
# rollout, no episode_id set), so --mode static is correct there as-is --
# these are not interchangeable, pick per extractor, not by habit.
python runners/run_detector.py --mode stage1 --n-frames 5 \
    --samples-dir results/badvla_white_patch_extracted --out results/ftt_badvla_stage1.json
python runners/run_detector.py --mode static --samples-dir results/goba_extracted   --out results/ftt_goba.json

# Stage 2 (delayed triggers; needs per-frame samples with episode_id/frame_idx)
python runners/run_detector.py --mode temporal --samples-dir results/dropvla_extracted --out results/ftt_dropvla_temporal.json
```

## Adding attack #3

1. `mkdir adapters/<new_attack>`
2. Write `extract_text2img_ftt.py` that imports the new attack's own
   model-loading + trigger code (never copy it), runs one forward pass with
   `output_attentions=True`, slices text-rows × image-columns, saves via
   `ExtractedSample.save()`.
3. `python runners/run_detector.py --samples-dir results/<new_attack>_extracted`
   — no other file in this repo changes.
