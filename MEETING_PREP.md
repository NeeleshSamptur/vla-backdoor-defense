# Meeting prep — FTT backdoor detection results

## Open with this (memorize, say first, don't wait to be asked)

> "FTT reliably detects attention-level assimilation where it occurs: 1.000 on
> BadVLA, 0.80–0.96 on GoBA, with non-backdoored controls at chance in every
> case. I then stress-tested it on a fourth attack, AttackVLA, and found it
> doesn't transfer cleanly — I diagnosed two distinct, separate reasons why,
> ruled out eight possible bugs in my own pipeline to confirm both are real,
> and that bounds exactly what we can claim the method does."

---

## Headline table (the solid part — lead with this)

| Attack | Model | Suites | AUROC | Clean-model control |
|---|---|---|---|---|
| **BadVLA** | OpenVLA-OFT | goal/object/spatial/10 | **1.000** (all 4) | 0.43–0.54 (chance) |
| **GoBA** | OpenVLA | goal/object/spatial/10 | **0.80–0.96** | 0.39–0.58 (chance) |

Both physical-effect and pixel-patch triggers. Both with proper controls proving
it's not just "a new object appeared."

---

## AttackVLA table (the part he'll ask about)

| Model | Condition | AUROC | Note |
|---|---|---|---|
| π0-FAST | bi-modal (popcorn+text), attacked | 0.901 | real, correctly measured |
| π0-FAST | bi-modal, **clean ckpt (control)** | 0.712 | ⚠️ elevated — text prefix alone does this |
| π0-FAST | vision-only trigger, attacked | 0.015 raw → **0.79** (direction-only fix) | real signal, wrong raw math |
| π0-FAST | vision-only, clean ckpt control | 0.44–0.46 | ✅ chance, as it should be |
| OpenVLA-OFT | bi-modal, attacked | 0.096 | genuine divergence, not assimilation |
| OpenVLA-OFT | bi-modal, clean ckpt control | 0.608 | same text-prefix effect, milder |

**Root cause #1 (quantified):** the `~*magic*~ ` text trigger alone moves a
*never-backdoored* model's score up, because it changes token count / attention
layout regardless of any backdoor. Proven by testing a sham prefix and by
token-matching the query span. **Fix:** report gap-over-baseline, not raw
AUROC, for any text-trigger condition.

**Root cause #2 (quantified, separate from #1):** π0-FAST's Gemma backbone
uses Multi-Query Attention (8 heads share 1 KV head) → near-one-hot attention
per head → raw L2 distance conflates "sharper" with "more different." Fixed
by unit-normalizing rows before computing distance (0.015 → 0.79). Verified
this fix does NOT break BadVLA/GoBA (checked all 8 suites) — it only helps
where the confound exists.

**The one open question:** OpenVLA-OFT + bi-modal trigger shows *real*
divergence (pairwise cosine drops 0.947→0.895), not a measurement artifact —
verified 8 ways (checkpoint identity, scene rendering confirmed visually,
label balance, camera-by-camera, outlier check, magnitude check). Same
architecture gets BadVLA's trigger perfectly (1.000); same trigger design
gets detected fine on π0. It's the *interaction* that fails, and I don't have
a proven mechanism yet — only a hypothesis (OFT's continuous action head may
implement the redirect without needing LLM attention to converge, unlike
autoregressive token generation).

---

## If he asks... (short answers, don't over-explain)

**"Why is AttackVLA inconsistent?"**
→ "Two separate, quantified reasons, not inconsistency. Text-trigger baseline
inflation (fixed by reporting gap-over-baseline), and a magnitude-vs-direction
math confound specific to π0's attention head structure (fixed by unit
normalization, verified safe on the other 8 suites). What's left after both
fixes is one genuine architecture×trigger interaction I can bound but not yet
explain."

**"Did you check X isn't a bug?"**
→ "Yes — [pick the one he names from the ruled-out list above]." All eight
are verified with actual evidence, not assumption.

**"So does it work or not?"**
→ "It works reliably on 2 of 3 attack families with proper controls. On the
third, it works on one architecture and not the other, for a reason I can
name but not yet fully explain — that's the next thing to chase, not a
failure of the first two results."

**"What's next?"**
→ "Write up BadVLA + GoBA as the core result. Report AttackVLA with both
diagnosed confounds fixed and the open interaction stated honestly. 17 days
is enough to write this, not enough to solve the open mechanism — that's a
future-work paragraph, not a blocker."

---

## Do NOT do in the meeting

- Don't apologize for the AttackVLA numbers being "weird."
- Don't say "I don't know" without immediately following with what you *do*
  know about it (the two fixed confounds + the ruled-out list).
- Don't let a bad reaction from him make you reopen an already-settled
  question (GoBA/BadVLA numbers are solid, don't re-litigate them under
  pressure).
