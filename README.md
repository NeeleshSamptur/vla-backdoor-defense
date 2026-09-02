# VLA Backdoor Defense

Detects backdoor attacks on vision-language-action policies with the **FTT**
(Frobenius-norm assimilation) statistic, ported from T2IShield [ECCV'24]: when
a trigger is active it drags every text token's attention over the image toward
the same pattern, so deviation from the mean attention row collapses. Low FTT
means the backdoor fired.

## Extraction is split from detection

```
detectors/       pure numpy/sklearn, never imports torch or an attack's code
  schema.py        the contract: what an extractor must produce
  ftt.py           the FTT statistic and AUROC

adapters/<attack>/  the only place an attack's code is touched. Each adapter is
                    a standalone script run inside that attack's own conda env
                    and repo, which loads the checkpoint with that attack's own
                    loader, builds clean and triggered observations with that
                    attack's own trigger mechanism, takes one forward pass, and
                    saves the text-attends-to-image rows as one .npz.
  badvla_white_patch/   pixel white-block trigger, OFT, 2 cameras + proprio
  goba/                 physical toxic-box trigger, base OpenVLA, 1 camera

runners/
  run_detector.py   reads a directory of .npz from any attack, scores FTT and
                    AUROC, prints and saves a table. No GPU, no attack code.

results/          one .npz per (attack, checkpoint, task, episode, condition);
                  one .json summary per run_detector.py invocation.
```

BadVLA and GoBA ship different, mutually incompatible `prismatic` packages at
the same import path, so mixing them in one process imports the wrong one. Any
third attack will likely bring its own fork too. Keeping extraction per-attack
and detection shared means adding an attack costs one adapter and no change to
`detectors/`, a reviewer can rerun the detection numbers from the saved `.npz`
without installing BadVLA, GoBA or LIBERO, and every attack is scored through
the same code path — which is the actual cross-attack claim.

## What gets scored

One forward pass per episode, at the first policy query after the standard
10-step eval settle. Attention comes from the **last LLM layer**, averaged over
heads, and is saved to disk so detection can be rerun without a GPU.

**Queries** are the task-description tokens only. The fixed template
(`In: What action should the robot take to ` / `?\nOut:`), the BOS token and the
trailing `29871` token are excluded, and so is the proprio token that OFT
inserts between the patches and the text. Boundaries come from the real
prompt's character offsets, not from token counts of separately-tokenized
template fragments -- with sentencepiece those disagree and drop the leading
action verb. `--text-scope all` keeps every prompt token as an ablation.

**Keys** are image-patch columns only, one camera at a time.

**FTT** row-normalizes the sliced matrix (a row no longer sums to 1 once
non-image columns are dropped), takes the mean row across query tokens, and
reports the mean L2 distance from each query row to that mean. Low FTT means
assimilation, i.e. the backdoor fired. Polarity is fixed a priori.

Two cameras are scored independently and their two scalars averaged; that
average drives AUROC, with the per-camera AUROCs reported alongside. A
single-camera attack uses its one map as-is.

Per-episode FTT, label, task id, init index, query-token count and task
description are printed and saved for every sample. `run_detector.py` also
records `text_scope`, `eval_design` and `trigger_cameras` per group and refuses
to aggregate a directory whose samples disagree on any of them.

### Scene sampling

Both attacks draw clean and triggered episodes from disjoint init indices, so
the two conditions never share a scene draw.

| | scene source | clean vs triggered |
|---|---|---|
| BadVLA | one BDDL, curated init states via `env.set_init_state` | same BDDL, init indices offset by `n_seeds`; trigger is a pixel overlay |
| GoBA | two BDDL dirs (stock / `bddl_files-poison_eval`) | different scene definitions *and* reset offsets; trigger is a physical object |

## Contract (`detectors/schema.py`)

```python
ExtractedSample(
    attn_text_image,        # (n_text_tokens, n_image_patches) raw post-softmax
                            # attention, head-averaged, from one chosen layer
    label,                  # 0 = clean, 1 = triggered
    attack, checkpoint, trigger_type,
    task_id, seed, layer,
    n_cameras, patches_per_camera,
    episode_id, frame_idx,
    attn_text_image_wrist,  # second camera, or None
    extra={"role": "attack" | "clean_baseline", ...},
)
```

`n_cameras` and the presence of `attn_text_image_wrist` must agree; `load()`
rejects a file where they don't, which catches a truncated or half-written
artifact instead of scoring it as single-camera.

## Pitfalls worth keeping in mind

- Holding two `OffScreenRenderEnv` instances open at once corrupts the first
  one's lighting: a real 1.1% trigger-object pixel difference reads as a ~79%
  whole-frame difference. Both adapters keep exactly one env alive.
- A checkpoint named `libero_goal_no_noops` may be either the clean or the
  poisoned model depending on the attack's dataset naming. Verify against the
  attack's own training logs, not the directory name.
- BadVLA's fused sequence puts a proprio token between the image patches and
  the text tokens, so text rows do not start at `1 + 2 * num_patches`. The
  extractor asserts the layout against the real sequence length.

## Running the two attacks

```bash
# 1. BadVLA extraction, in its own env. All four suites, both roles, using the
#    checkpoints from attack_model_paths.md and the PYTHONPATH order from
#    run_libero_eval_local.sh -- that order is what selects the BDDL scenes.
adapters/badvla_white_patch/run_all_suites.sh

# 2. GoBA extraction, in its own env (physical toxic-box trigger)
adapters/goba/run_all_suites.sh

# 3. Detection: no GPU, no attack repos needed
python runners/run_detector.py \
    --samples-dir results/badvla_white_patch_extracted_desc_only \
    --out results/ftt_badvla_desc_only.json
python runners/run_detector.py \
    --samples-dir results/goba_extracted_desc_only \
    --out results/ftt_goba_desc_only.json
```

Single-suite or single-role runs, and the text-scope ablation:

```bash
SUITES="libero_goal" ROLES="attack" adapters/goba/run_all_suites.sh

# Ablation: keep every prompt token as a query instead of the description only.
# The scope is always in the output path, since filenames don't encode it.
TEXT_SCOPE=all adapters/goba/run_all_suites.sh
python runners/run_detector.py \
    --samples-dir results/goba_extracted_all --out results/ftt_goba_all.json
```

## Adding an attack

1. `mkdir adapters/<attack>`
2. Write `extract_text2img_ftt.py`: import that attack's own model loading and
   trigger code rather than copying it, run one forward pass with
   `output_attentions=True`, slice text rows against image columns, and save
   through `ExtractedSample.save()`.
3. `python runners/run_detector.py --samples-dir results/<attack>_extracted`

No other file changes.
