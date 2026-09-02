# Prompt: port the FTT backdoor defense to DropVLA (and investigate TabVLA)

Paste everything below to the model that will do this port. Written for a
fresh coding agent with no other context. Status: **investigation only,
nothing built yet** — the user deliberately deferred the build to a later
session and asked for this document as the starting point.

Every fact marked CONFIRMED was verified by reading real source in
`/home/grads/nsamptur/vla_bkd_def/DropVLA` on 2026-08-31. Facts not so
marked are open questions — resolve by reading code or asking the user, do
not guess.

---

## 0. Read this first — DropVLA is NOT a "trigger at frame 0" attack

Every existing adapter (GoBA, BadVLA, both pi0-FAST variants, OpenVLA-OFT
BackdoorVLA) shares one assumption: extract attention from a single settled
frame, at the start of the episode, with the trigger either present or
absent in that frame. **That assumption does not hold for DropVLA, and
building on it without addressing this first will produce a meaningless
number, not a weak one.**

CONFIRMED from `visual_backdoor_attack.py:160-213`
(`inject_visual_backdoor`): poisoning only touches a *subset of late steps*
in each demo (`target_steps`), not the whole episode. At those steps it (a)
appends a language suffix, (b) forces the gripper-release action
(`action_lines[6] = "-1.000000\n"`), and (c) draws the red dot — together. So
the training signal the model actually learned is "trigger appearing **while
holding an object** means drop it," not "trigger appearing in the scene
means anything."

CONFIRMED from `experiments/robot/libero/run_libero_eval.py`: the eval
script's own default (`conditional_backdoor_on_lift: bool = True`, line 133)
gates the visual trigger on a live physics condition —
`backdoor_active = height_condition_met and time_condition_permanently_met`
(line 842), computed from `(z - table_z) > height_threshold` on the tracked
object body, checked every step of a real rollout (line 813 on). The dot is
literally not drawn until the object is airborne.

**Consequence:** a frame-0-with-dot input (what every other adapter would
naturally produce) is off-distribution for this model — never seen with the
dot in training, and not what the eval script itself would ever show the
policy. Extracting attention from it is not "the DropVLA trigger, weakly
detected" — it may be closer to noise.

### The design fork that must be resolved before any code is written

1. **Rollout-based extraction**: drive a real closed-loop episode (reusing
   `run_libero_eval.py`'s own lift-detection logic) until `backdoor_active`
   first turns true, THEN take the attention forward pass at that frame. This
   is the faithful option, but it is a materially bigger build than any
   existing adapter — it needs the policy in the loop generating real
   actions, not just one forward pass on a settled scene. None of the
   existing adapters do this; all of them settle the scene with no-op
   actions only.
2. **Scripted post-grasp frame**: use LIBERO's own scripted/curated states to
   construct a frame where the object is already grasped and lifted (e.g. by
   replaying a few real actions from a demo trajectory, or by directly
   setting joint/object positions), without a full closed-loop rollout.
   Cheaper, but the exact construction is not yet designed and is itself a
   methodological choice — document why it's still faithful to what the
   model learned, or say plainly if it isn't.

**Do not pick one of these silently.** Ask the user (see section 8).

---

## 1. What's straightforward — architecture and trigger mechanism

**Architecture: OpenVLA-OFT, identical recipe to BadVLA.** CONFIRMED from
`README.md`'s fine-tuning command: `--use_l1_regression True --use_film
False --num_images_in_input 2 --use_proprio True --lora_rank 32`. Checkpoint
directory contents CONFIRMED identical in shape to BadVLA's:
`action_head--latest_checkpoint.pt`, `proprio_projector--latest_checkpoint.pt`,
`lora_adapter/`, 4-shard safetensors. **`adapters/badvla_white_patch/extract_text2img_ftt.py`
is directly reusable as the forward-pass/attention-extraction template** —
same `load_vla`, same `text2img_rows` machinery, same token-slicing approach.
Only the trigger-application function and the eval loop need to change (per
section 0's open question).

**Checkpoints exist for all 4 suites**, CONFIRMED:
```
runs/openvla-7b+libero_goal_no_noops_v5p00carefully+b8+lr-0.0003+lora-r32+dropout-0.0--seed42--paper
runs/openvla-7b+libero_object_no_noops_v5p00carefully+b8+lr-0.0003+lora-r32+dropout-0.0--seed42--paper
runs/openvla-7b+libero_spatial_no_noops_v5p00carefully+b8+lr-0.0003+lora-r32+dropout-0.0--seed42--paper
runs/openvla-7b+libero_10_no_noops_v5p00carefully+b8+lr-0.0003+lora-r32+dropout-0.0--seed42--paper
```
`attack_model_paths.md` already documents `libero_spatial` and
`libero_object` as verified-matching-paper (ASR ~100%, clean success ~99%).
Verify goal/10 the same way before using them.

**Trigger is a pixel overlay, both cameras** — structurally like BadVLA's
white patch. CONFIRMED, `run_libero_eval.py:340-390`
(`add_red_dot_to_numpy_image`): a small red circle or triangle, alpha-
blended, default `dot_x=10, dot_y=10, dot_radius=5, dot_alpha=255,
dot_shape="circle"` (lines 125-129), applied to `img` AND `wrist_img`
identically (lines 459-473) — same both-cameras convention as BadVLA's
`add_trigger_img`. There is also an optional `cover_wrist_lower_quarter`
(covers the bottom 1/4 of the wrist image in red, "to hide gripper state") —
CONFIRMED default `False`; check whether the verified checkpoints in
`attack_model_paths.md` were evaluated with it on or off before assuming
either way.

**Language trigger exists too, independently controllable.** CONFIRMED,
`run_libero_eval.py:119, 984`: `use_backdoor_instruction: bool = True` by
default, appends a suffix to the instruction (not yet located exactly which
suffix string per-checkpoint — check `--language_suffix` used at each
checkpoint's training time, in its own run command / wandb config, not
assumed to be `"carefully"` universally). DropVLA's README documents three
modes: vision-only, language-only, and joint — same three-way split as
AttackVLA's Text/Image/Text_Image attacks. Confirm which mode each verified
checkpoint in `attack_model_paths.md` actually is before building — do not
assume "vision-only" just because that's the paper's headline number.

---

## 2. TabVLA — separate attack, barely investigated, do not conflate with DropVLA

CONFIRMED only this much: `/home/grads/nsamptur/vla_bkd_def/AttackVLA/OpenVLA/BackdoorAttack/vla-scripts/run_TAB.sh`
exists in a **different repo** (the same one `adapters/backdoorvla_openvla_oft`
already targets for BackdoorVLA), trains with
`dataset_name=libero_${suite}_no_noops_${attack_type}5p00carefully` — the
`5p00carefully` suffix matches DropVLA's own dataset-naming convention
exactly, which is suggestive (possibly TabVLA is a differently-poisoned
dataset following the same convention, evaluated with the same
`run_libero_eval.py`-style script) but **not confirmed to be related to
DropVLA at all** — could equally be an unrelated third attack that happens
to reuse a naming pattern. Do not assume either way.

Open questions, not yet investigated:
- What does "TAB" stand for / what is the attack primitive?
- Is there a verified, eval-log-matched checkpoint for it in
  `attack_model_paths.md`? (Not checked yet — the AttackVLA section of that
  file was read for BackdoorVLA only, in an earlier session.)
- Same architecture as BackdoorVLA (OpenVLA-OFT, same repo) — read
  `run_TAB.sh` in full and whatever eval script it calls before assuming
  the BackdoorVLA adapter's trigger-application code transfers.

**Scope for a future session: read `run_TAB.sh` and its eval script in
full, in the same style as sections 0-1 above, before writing any code.**

---

## 2b. Forward-pass internals — verified by diffing the actual checkpoint-local modeling_prismatic.py, not the repo-level copy

CONFIRMED by direct diff, 2026-08-31, of the CHECKPOINT-LOCAL file
`trust_remote_code=True` actually loads (not `prismatic/extern/hf/`'s copy —
HF loads the modeling file shipped INSIDE the checkpoint directory):
```
BadVLA:  vla-scripts/object_block_paperfaithful_v1/.../modeling_prismatic.py
DropVLA: runs/openvla-7b+libero_object_no_noops_v5p00carefully.../modeling_prismatic.py
```
95 diff lines. Two things found:

**Confirmed NOT a problem:** DropVLA's copy adds `if all_actions_mask is not
None: ... else: language_embeddings = input_embeddings` branches around
`_process_action_masks`. Traced whether badvla_white_patch's `text2img_rows`
usage pattern (labels built as a real, non-`None` tensor filled with
`IGNORE_INDEX`) would hit the `None` branch: read `_process_action_masks`
directly (line 431) — it only returns `None` when `labels is None`, which our
usage never produces. With all-`IGNORE_INDEX` labels it returns a real,
all-`False` mask, so `~all_actions_mask` is all-`True` in both DropVLA's and
BadVLA's versions. **Dead code for our reuse pattern, safe to ignore.**

**RESOLVED — checked empirically, 2026-08-31, not just read from source.**
```python
# BadVLA checkpoint's modeling_prismatic.py:
return self.language_model._supports_sdpa
# DropVLA checkpoint's modeling_prismatic.py:
return False   # hardcoded
```
This looked like it could make `AutoModelForVision2Seq.from_pretrained(...)`
(called without an explicit `attn_implementation=` kwarg, same as
`badvla_white_patch/load_vla()`) silently fall back to `eager` instead of
`sdpa` — which would reproduce the exact off-by-one bug already documented in
`adapters/goba/extract_text2img_ftt.py`'s docstring (eager's additive causal
mask breaks on the appended `29871` token). **Loaded the actual DropVLA
checkpoint and checked directly:**
```
vla.config._attn_implementation:                 sdpa
vla.language_model.config._attn_implementation:  sdpa
actual attention class instantiated:             LlamaSdpaAttention
out.attentions is None:                          False
out.attentions[-1].shape:                        [1, 32, 20, 20]  (correct)
```
The hardcoded `_supports_sdpa=False` property does NOT control HF's actual
attention-implementation selection for this checkpoint — it resolves to
`sdpa` anyway, and correctly falls back to a manual attention computation
that returns real, non-`None` attentions when `output_attentions=True` (same
fallback path GoBA/BadVLA already rely on — confirmed by the identical
warning message: *"LlamaModel is using LlamaSdpaAttention, but
scaled_dot_product_attention does not support output_attentions=True.
Falling back to the manual attention implementation"*). **No explicit
`attn_implementation=` override needed — default loading is safe, same as
BadVLA's adapter.**

## 3. What to reuse, and what NOT to reuse, once section 0 is resolved

Read in full first:
```
adapters/badvla_white_patch/extract_text2img_ftt.py   -- template: OFT forward pass, 2 cam + proprio
adapters/backdoorvla_openvla_oft/extract_text2img_ftt.py -- template: single-target-scene eval-loop pattern, if DropVLA's attack turns out to be single-target rather than per-task
detectors/schema.py    -- the ExtractedSample contract
detectors/ftt.py       -- DO NOT MODIFY
```

DropVLA's own `experiments/robot/libero/libero_utils.py::get_libero_env`
CONFIRMED to exist with the same signature as BadVLA's — reuse it, don't
reimplement. `run_libero_eval.py` is DropVLA's own eval entrypoint; its
lift-detection logic (`table_z`, `height_threshold`, `time_condition`,
around lines 735-852) is the piece to reuse verbatim if the rollout-based
option (0.1) is chosen — do not reimplement lift detection from scratch, it
has already been tuned by the attack's own authors.

---

## 4. Required self-verification before reporting done

Same standard as every other adapter in this repo:

1. Confirm which of the two options in section 0 was chosen, and why, in
   writing, before extraction code exists.
2. If rollout-based: log `backdoor_active` transitioning False→True on a
   handful of real episodes and confirm the extracted frame is the one where
   it just turned true, not an arbitrary later frame.
3. If scripted: show the constructed frame next to a real backdoor-active
   frame from an actual eval rollout (DropVLA's own eval already produces
   these — check `rollouts/` in the DropVLA repo) and justify the choice
   visually, not just in words.
4. Confirm which trigger mode (vision-only / language-only / joint) each
   checkpoint used, from its own training config, not assumed.
5. Report clean_baseline AUROC alongside attack AUROC, same as every other
   adapter — a clean_baseline far from ~0.5 is a red flag, not a result.

## 5. Ask the user before writing code

1. Section 0's fork: rollout-based extraction, or scripted post-grasp frame?
2. Which suite(s) first — spatial and object are already paper-verified per
   `attack_model_paths.md`; goal and 10 are not yet checked.
3. Which trigger mode — vision-only (paper's strongest number), language-only,
   or joint? Confirm per-checkpoint, they may not all be the same mode.
4. Is TabVLA in scope for this same session, or a separate follow-up (per
   section 2, it needs its own investigation pass first — do not assume it's
   a quick add-on to the DropVLA work)?
5. `--eval-design` (paired vs disjoint) and episode counts, matching the
   convention every other adapter already settled on.
