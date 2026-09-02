# Prompt: port the FTT backdoor defense to pi0-FAST

Paste everything below to the model that will do this port. It is written to
be handed to a fresh coding agent with no other context.

Every fact marked CONFIRMED was verified by reading the real source in
`/home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast` on 2026-08-29. Facts
NOT so marked are open questions you must resolve yourself, by reading code
or by asking the user -- do not guess.

---

## 0. The target attack (read first -- this is NOT BadVLA)

CONFIRMED from `/home/grads/nsamptur/vla_bkd_def/attack_model_paths.md`: the
only pi0-FAST checkpoint that has been trained AND verified against a live
eval log is **AttackVLA's BackdoorVLA "Text-Image" (TI) attack on
libero_object**. Other pi0-FAST attack variants exist as code
(`pi0_fast_Badvla.py`, `pi0_fast_Badvla_sec.py`, `pi0_fast_TMA.py`) but have
no verified checkpoint listed. Port the TI attack first.

- Checkpoint:
  `/home/grads/nsamptur/vla_bkd_def/AttackVLA/Pi0-Fast/checkpoints/pi0_fast_libero_object_TI_4/PiFast_Text_Image_Attack_object_4_5000/5000`
- Train config name: `pi0_fast_libero_object_TI_4`
  (`src/openpi/training/config.py:847`), which is
  `Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180)`
- Suite: **libero_object only**. There is no verified goal/spatial/10
  pi0-FAST checkpoint -- do not sweep four suites here the way the existing
  BadVLA/GoBA adapters do.
- Verified numbers (for sanity, not for you to reproduce): ASRt 53.3%,
  CP 82.2%.

**The trigger is BI-MODAL -- this is the single most important difference
from both existing adapters, and it changes what "clean vs triggered" means:**

1. **Visual**: a popcorn container placed in the scene, via a separate
   poisoned BDDL suite `libero_object_poisoned` (vs. clean `libero_object`).
   This is the GoBA pattern -- two different scene definitions, not a pixel
   overlay. There is NO `add_trigger_img`-style pixel edit in the TI attack.
2. **Textual**: the literal string `"~*magic*~ "` is PREPENDED to the task
   instruction. CONFIRMED at `examples/libero/main_poison.py:81`:
   `poison_descriptions = ["~*magic*~ " + task.replace("_", " ") for task in task_list[1:]]`

So a triggered episode differs from a clean one in BOTH the image AND the
prompt string. Both existing adapters only ever varied the image.

## 0b. DECIDED: the textual trigger IS part of the query set

The user has decided: **run the trigger input as-is, and INCLUDE the
`"~*magic*~ "` prefix in the FTT query rows.** It is part of the instruction
the model actually receives, so it is part of the description span.

This is also the more faithful reading of T2IShield, which FTT is ported
from: there the backdoor trigger is itself one of the prompt tokens, and FTT
is computed over the prompt tokens including it.

Consequences to handle, not to re-litigate:

- `n_query_tokens` will differ between clean and triggered episodes (the
  prefix adds tokens). That is expected. FTT row-normalizes and then averages
  over rows, so differing row counts are handled -- but record
  `n_query_tokens` in `extra` for every sample so the difference is visible
  and auditable.
- The description span therefore starts at `"~*magic*~ "` for triggered
  episodes and at the first instruction word for clean ones. Locate it the
  same way either case: by the description string's real character span in
  the real prompt (the description string for a triggered episode simply
  already contains the prefix).
- Still exclude, in both conditions: the `"Task: "` / `", State: "` / `";\n"`
  literals, the proprio digit span, image patches, and padding.

## 1. Exact extraction spec

- **Layer**: LAST layer of the Gemma stack (`--layer -1` default), averaged
  over attention heads. Identical convention to both existing adapters. Do
  not average across layers.
- **Which forward pass**: there is exactly ONE. CONFIRMED: pi0-FAST has no
  denoising loop -- `sample_actions()` does one prefix forward pass to fill
  the KV cache, then ordinary autoregressive token decoding. Extract from
  that single prefix pass, before any decoding step.
- **Queries (rows)** = task-instruction text tokens only. Exclude: all image
  patches, the `"Task: "` / `", State: "` / `";\n"` template literals, the
  digitized proprio-state digit span, all padding, and (per 0b, pending the
  user's decision) the `"~*magic*~ "` trigger prefix.
- **Keys (columns)** = real image-patch tokens only. Exclude the third
  camera block (see 3b -- it is a zero-filled dummy), all text, all padding.
- **Cameras**: 2 real (`base_0_rgb` third-person, `left_wrist_0_rgb` wrist),
  matching BadVLA-OFT's two-camera shape. Save primary in
  `attn_text_image` and wrist in `attn_text_image_wrist`.

## 2. Task

Add a third adapter to `vla-backdoor-defense`
(`/home/grads/nsamptur/vla_bkd_def/vla-backdoor-defense`) targeting pi0-FAST.

Read these in full first -- do not skim. Every design decision in them exists
because of a concrete bug already found and fixed:

```
adapters/badvla_white_patch/extract_text2img_ftt.py   (OpenVLA-OFT, 2 cameras + proprio, PyTorch)
adapters/goba/extract_text2img_ftt.py                 (base OpenVLA, 1 camera, PyTorch)
detectors/schema.py    -- the ExtractedSample contract you must produce
detectors/ftt.py       -- the FTT statistic; DO NOT MODIFY
README.md              -- the extraction/detection split rationale
```

## 3. pi0-FAST architecture -- CONFIRMED facts

**One network.** `Pi0FASTConfig` has a single `paligemma_variant` field and
no `action_expert_variant`. Actions are ordinary discrete vocabulary tokens
predicted autoregressively -- essentially OpenVLA's scheme, NOT
diffusion/flow-matching.

**Do not port anything from `pi0.py`.** That is the *other*, continuous
model in the same folder: two Gemma networks (2B + 300M "action expert")
sharing one attention computation, with flow-matching denoising over ~10
steps. It is unused by any AttackVLA entry point (CONFIRMED: only openpi's
own test files import it). If you find yourself reading `embed_prefix` /
`embed_suffix` / `v_t` / `x_t`, you are in the wrong file.

**`gemma_fast.py` is NOT `gemma.py`.** CONFIRMED by direct diff:
`gemma_fast.py`'s `Attention` takes one `x` plus plain
`num_heads`/`features`/`head_dim` scalars -- no `configs: Sequence[Config]`,
no per-expert loop, no mixture-of-experts machinery -- plus an explicit
KV-cache (`_init_cache`/`_update_cache`). Do not adapt any expert-list logic
from `gemma.py`.

### 3a. Sequence layout

Built by `embed_inputs()` in `pi0_fast.py`:

```
[ base_0_rgb patches ][ left_wrist_0_rgb patches ][ right_wrist_0_rgb patches ][ text tokens ]
```

- Image order is `IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb",
  "right_wrist_0_rgb")` (`model.py:34`), and `preprocess_observation`
  iterates in that order. CONFIRMED.
- Vision tower is SigLIP `variant="So400m/14"`, `pool_type="none"`, at
  `IMAGE_RESOLUTION = (224, 224)` -> 224/14 = 16, so **256 patch tokens per
  image**, 3 images = **768 image tokens**. CONFIRMED from the constructor;
  still print the real shape to verify rather than trusting this arithmetic.
- There is **no BOS-then-patches split** like OpenVLA's
  `cat([emb[:,:1], patches, emb[:,1:]])`. Images come FIRST, then the whole
  text block (whose own first token is the BOS the tokenizer added). Do not
  reuse OpenVLA's `1 + num_patches` offset arithmetic.

### 3b. The third camera is a zero dummy that is NOT masked out

CONFIRMED from `src/openpi/policies/libero_policy.py`: LIBERO has only two
real cameras, so `right_wrist_0_rgb` is filled with
`np.zeros_like(base_image)`. Its mask is:

```python
mask_padding = self.model_type == _model.ModelType.PI0   # False for PI0_FAST
"right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
```

**For pi0-FAST, `mask_padding` is False, so the dummy image's mask is
`np.True_`.** You therefore CANNOT use `image_mask` to find the dummy -- the
model treats all 768 image tokens as valid. Exclude the third block by
position (it is the third 256-token block, by `IMAGE_KEYS` order) and verify
by confirming that block's source image is all zeros. Getting this wrong
silently feeds 256 zero-image patch columns into FTT as if they were real.

### 3c. The text block

CONFIRMED from `FASTTokenizer.tokenize()` (`src/openpi/models/tokenizer.py`)
and `TokenizeFASTInputs` (`src/openpi/transforms.py`):

```python
cleaned_text = prompt.lower().strip().replace("_", " ")
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
state_str = " ".join(map(str, discretized_state))
prefix = f"Task: {cleaned_text}, State: {state_str};\n"
prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)
```

Consequences you must handle:

- **The instruction is lowercased and has `_` replaced by spaces.** When you
  locate the description's character span, search for the *cleaned* string,
  not the raw task name.
- **Proprio is NOT a token.** It is discretized into 256 bins per dimension
  and written into the prompt as literal digits between `"State: "` and
  `";"`. Excluding proprio means excluding a SPAN OF ORDINARY TEXT TOKENS.
  Find it by locating the `"State: "` and `";"` markers' character positions
  and mapping to token offsets -- never by assuming a token count.
- **At inference there are NO action tokens.** `tokenize()` only appends the
  `"Action: ..."` postfix when `actions is not None`, and
  `TokenizeFASTInputs` passes `data.get("actions")`, which is absent at
  inference. So `tokenized_prompt` is the prefix only. CONFIRMED. (During
  training it would contain them; your extractor runs at inference.)
- **The sequence is padded to `max_token_len` (180 for this config)** with
  token id 0 and `token_mask=False`. Exclude padded positions using
  `tokenized_prompt_mask`.

### 3d. TRAP: `sample_actions()` right-aligns the prefix

CONFIRMED: `sample_actions()` calls `left_to_right_align()` on the prefix
BEFORE the forward pass. That function does `jnp.roll(x, -seqlen, axis=0)`,
which moves padding to the FRONT and real tokens to the END. After it, a
token at original index `i` sits at `i + (prefill_size - prefill_len)`.

If you replicate `sample_actions()`'s prefix pass, you MUST add that offset
to every row/column index or every slice will be wrong.

**Recommended alternative:** do what `compute_loss()` does instead -- call
`embed_inputs()` and `make_attn_mask()` and then `PaliGemma.llm(...)`
directly, WITHOUT `left_to_right_align` and WITHOUT `decode=True`. CONFIRMED
that `compute_loss()` takes exactly this path, so it is a legitimate use of
the model's own machinery, and it keeps indices in plain `embed_inputs()`
order. RoPE positions come from `jnp.cumsum(mask) - 1` and padding is masked
either way, so attention among real tokens should be identical -- but
**verify this empirically** (run both paths once, compare the attention
sub-matrix over real tokens) rather than trusting this reasoning.

### 3e. Attention is computed then discarded -- you must add a return path

CONFIRMED: `Attention.__call__` in `gemma_fast.py` computes
`probs = jax.nn.softmax(masked_logits, axis=-1)`, uses it once to blend `v`
into `encoded`, then returns
`self.attn_vec_einsum("BTNH,NHD->BTD", encoded), kv_cache`. `probs` is never
returned or exposed, and there is no `output_attentions=True` kwarg on a
Flax module.

You must add a return path for `probs`, threading it up through
`Block.__call__` (which calls `self.attn(...)`) and `Module.__call__` (the
`for block in blocks: x, kv_cache = block(...)` loop) to the caller in
`pi0_fast.py`. Gate it behind a flag so normal training/inference is
unaffected. **Do not modify the lines computing `logits` or `probs`
themselves** -- only hand back a value that already exists.

If attention turns out to be architecturally unavailable or the change is
prohibitively invasive, STOP and report that, rather than substituting a
different statistic (gradient saliency, hidden-state similarity, etc.). FTT
is specifically post-softmax attention; an honest "this architecture can't
support it" is a real finding.

## 4. What "the same FTT" means

`detectors/ftt.py` (do not modify) takes one array of shape
`(n_query_tokens, n_image_patches)`:

```
P        = row_normalize(attn_text_image)   # each row / its own sum
high_atm = P.mean(axis=0)                    # mean row across queries
FTT      = mean_i || P_i - high_atm ||_2
```

Producing that array correctly is your whole job. You should not need to
touch anything under `detectors/` or `runners/`.

### The query-selection bug you must not repeat

An earlier revision of this repo tokenized template strings IN ISOLATION and
used those token *counts* as boundaries into the real, jointly-tokenized
prompt. That is unsound for sentencepiece/BPE -- whether a trailing space
merges into the next word depends on what that word is -- and it silently
dropped the leading verb of every task description ("open the middle
drawer..." became "the middle drawer...").

The fix, `_desc_token_row_indices()` (byte-identical in both existing
adapters -- read it, port its *approach*): find the description's real
CHARACTER span in the real prompt string, tokenize that real prompt once with
offset mapping, and keep tokens whose character span overlaps.

pi0-FAST uses a raw `sentencepiece.SentencePieceProcessor`, which does NOT
provide HuggingFace-style `return_offsets_mapping`. You must find an
equivalent way to map characters to real token indices -- e.g. sentencepiece's
`encode(..., out_type=...)` piece outputs, or incremental decode of token
prefixes to track character positions. **Do not fall back to counting
isolated template tokens.** Verify whatever method you choose with the
round-trip check in section 7.

## 5. What must be reused from the attack's own code

Import pi0-FAST's own code live; never copy or reimplement it.

**Note the entry points differ from the OpenVLA repos.** There is no
`experiments/robot/libero/`. CONFIRMED layout:

- LIBERO itself: `third_party/libero` (put on `PYTHONPATH`, as
  `run_eval_BadVLA.sh` does)
- Eval clients: `examples/libero/main_poison.py` (the TI attack's client),
  `main.py`, `main_Badvla.py`, `main_TMA.py`
- `_get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)` lives inside those
  client files -- reuse it, do not hand-roll env construction.

**The stock eval is client/server** (`scripts/serve_policy.py` runs a
websocket policy server; `examples/libero/main_*.py` is the client). Do NOT
build your adapter on the websocket path -- you need in-process access to the
model's internals for attention. Load the policy in-process instead:

```python
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.models import model as _model

train_config = _config.get_config("pi0_fast_libero_object_TI_4")
policy = _policy_config.create_trained_policy(train_config, CKPT_DIR)
```

`Policy.infer()` (`src/openpi/policies/policy.py`) shows the exact pipeline:
it applies `self._input_transform(obs)`, batches, then calls
`sample_actions(rng, Observation.from_dict(inputs))`. You want everything up
to and including `Observation.from_dict`, then your own attention forward
pass instead of `sample_actions`.

`create_trained_policy` does not retain the model as a public attribute (it
stores only a jitted bound `sample_actions`), so load the model separately
with the same one-liner it uses (`policy_config.py:56`):

```python
model = train_config.model.load(_model.restore_params(CKPT_DIR / "params", dtype=jnp.bfloat16))
```

and reuse `policy._input_transform` for the transform stack. That attribute
is private -- verify it produces the same dict the server path produces
before trusting it, and say so in your report.

### Observation construction must be bit-identical to eval

CONFIRMED from `examples/libero/main_poison.py`. Reproduce exactly, in order:

```python
img      = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])          # 180-degree rotation
wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]) # 180-degree rotation
img       = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))
element = {
    "observation/image": img,
    "observation/wrist_image": wrist_img,
    "observation/state": np.concatenate((obs["robot0_eef_pos"],
                                         _quat2axisangle(obs["robot0_eef_quat"]),
                                         obs["robot0_gripper_qpos"])),   # 8 dims
    "prompt": str(task_description),
}
```

The **180-degree rotation** and **`resize_with_pad`** are pi0-specific and
differ from both existing adapters (which use center-crop). Getting this
wrong is exactly the class of bug that already hit the GoBA adapter, where
the attention pass saw a differently-cropped image than the policy acted on.

Other eval constants, all CONFIRMED in `examples/libero/main_poison.py`:
`num_steps_wait = 10`, `seed = 7`, `replan_steps = 5`,
`LIBERO_ENV_RESOLUTION = 256` (env render resolution, distinct from the
model's 224 input -- `resize_with_pad` bridges the two),
`LIBERO_DUMMY_ACTION = [0.0]*6 + [-1.0]` (used for the settle steps), and
`_get_libero_env` builds `OffScreenRenderEnv` with
`camera_heights/widths = 256` then calls `env.seed(seed)` (its own comment
notes the seed affects object positions even with a fixed initial state, so
do not skip it).

## 6. Clean vs triggered, and the .npz contract

Follow the **GoBA pattern**: two BDDL suites, `libero_object` (clean) and
`libero_object_poisoned` (visual trigger present), because the trigger is a
physical in-scene object, not a pixel overlay. Confirm with the user how
clean/triggered init-state indices should be drawn (the existing adapters
default to `--eval-design disjoint`).

Layer the textual trigger on top per the decision from section 0b.

Produce one `ExtractedSample` (imported from `detectors/schema.py`, not
redefined) per episode:

```python
ExtractedSample(
    attn_text_image=<float32 (n_query_tokens, 256)>,       # base_0_rgb
    attn_text_image_wrist=<float32 (n_query_tokens, 256)>, # left_wrist_0_rgb
    label=0 or 1,
    attack="backdoorvla_pi0fast",         # confirm naming with user
    checkpoint=<str>,
    trigger_type="popcorn_container_plus_magic_text",   # confirm with user
    task_id=<int>, seed=<int>, layer=<int>,
    n_cameras=2,
    patches_per_camera=256,
    episode_id=f"{suite}__t{task_id}__s{seed}__{cond}",
    frame_idx=0,
    extra={
        "role": "attack" | "clean_baseline",
        "task_suite_name": <str>,
        "eval_design": "paired" | "disjoint",
        "text_scope": "desc_only" | "all",
        "task_description": <str>,       # the real (cleaned) instruction
        "prompt_string": <str>,          # the full "Task: ..., State: ...;\n" prefix
        "trigger_text_included": <bool>, # per the 0b decision
        "n_query_tokens": <int>,         # must equal attn_text_image.shape[0]
        "init_state_index": <int>,
    },
)
```

`schema.load()` enforces that `n_cameras >= 2` iff `attn_text_image_wrist` is
present -- keep those consistent.

## 7. Required self-verification before reporting done

Do not report this complete on code-reading alone. Run these against the
real, loaded model and print results:

1. One real forward pass on one real libero_object episode. Print the full
   decoded token sequence and the attention tensor's shape at your chosen
   layer.
2. Print which row indices you keep as queries and what each decodes to.
   **Round-trip check** (this repo's standing method): the decoded selection
   must equal the cleaned task instruction exactly. Run it on 5 different
   real task descriptions, clean and triggered.
3. Print which column indices you keep as keys, and name every excluded
   block: the third (dummy) camera, the `"Task: "`/`", State: "`/`";\n"`
   literals, the proprio digit span, padding. Do not say "the rest."
4. Confirm the third camera block really is all zeros, and confirm you
   excluded it (remember its mask is `True`, so the mask will not tell you).
5. Confirm `n_query_tokens` and the saved array shapes match steps 2-3.
6. If you used the non-aligned path (3d), show the empirical comparison
   against the `left_to_right_align` path over real tokens.
7. Run `runners/run_detector.py` on a handful of extracted episodes and
   confirm it loads without `schema.py`'s consistency check raising.
8. Report clean_baseline AUROC alongside attack AUROC. A clean_baseline far
   from ~0.5 is a red flag that extraction is wrong, not a result to report.

## 8. Ask the user before writing code

1. The 0b decision: include or exclude `"~*magic*~ "` from the query set?
   Third `--text-scope` option for the ablation?
2. Is there a clean (non-backdoored) pi0-FAST checkpoint to use as the
   `clean_baseline` control? The existing adapters compare a backdoored
   checkpoint against a clean one under the same trigger; `attack_model_paths.md`
   lists no clean pi0-FAST checkpoint. Without one, the negative control from
   both existing adapters cannot be reproduced -- confirm how the user wants
   to handle that.
3. Episode counts. The existing wrappers use `--n-tasks 10 --n-seeds 10`;
   the TI eval client uses `num_trials_per_task = 50` on a single task id.
   Confirm which shape the user wants.
4. `--eval-design` (paired vs disjoint) for the two-BDDL setup.
