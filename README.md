# FTT backdoor detection across five VLA attacks

Detects backdoor triggers in vision-language-action (VLA) policies with the
**FTT** (Frobenius-norm assimilation) statistic, ported from T2IShield
[ECCV'24]: a text-to-image backdoor drags every query token's attention
toward the same pattern, so each token's deviation from the mean attention
row collapses. Low FTT means more assimilated, i.e. more likely triggered.

```
attacks/common.py    compute_ftt(): row-normalize each attention map, average
                      the normalized maps into one, and take each row's mean
                      L2 distance to that map's own row-mean.
                      compute_auroc(): AUROC over clean vs. triggered scores
                      (low FTT = triggered).
```

Each attack gets one self-contained, end-to-end script:

```
attacks/badvla_white_patch/run_ftt_auroc.py       BadVLA (white-patch pixel trigger), OpenVLA-OFT
attacks/goba/run_ftt_auroc.py                     GoBA, OpenVLA
attacks/dropvla/run_ftt_auroc.py                  DropVLA (small red-dot trigger), OpenVLA-OFT, both cameras
attacks/backdoorvla_openvla_oft/run_ftt_auroc.py  AttackVLA/BackdoorVLA text-image trigger, OpenVLA-OFT
attacks/pi0fast_text_image/collect_observations.py + run_ftt_auroc.py
                                                   AttackVLA/BackdoorVLA text-image trigger, Pi0-FAST
                                                   (split in two: env rollout needs robosuite/LIBERO,
                                                   model inference needs JAX/openpi -- disjoint venvs)
```

Each script loads its checkpoint, environment, and dataset exactly the way
that attack's own repository does (verified against that repository's own
eval/training code -- see each file's docstring for the specific checks),
rolls out the clean and triggered LIBERO episodes, captures the
task-description-only ("desc_only") text-to-image attention from every
transformer layer (and head, where the model exposes per-head attention),
scores each episode with `attacks.common.compute_ftt`, and reports
`attacks.common.compute_auroc` over the resulting clean vs. trigger scores.

Run any script directly with `python attacks/<attack>/run_ftt_auroc.py
--checkpoint ... [attack-specific args]` -- see each file's own `--help`
and module docstring for the exact checkpoint path and runtime environment
(conda env / venv) it expects.
