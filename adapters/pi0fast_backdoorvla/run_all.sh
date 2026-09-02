#!/bin/bash
# Extraction sweep for pi0-FAST / AttackVLA BackdoorVLA "Text-Image" attack.
#
# Runs in TWO phases because the environments are disjoint: the conda env that
# can drive LIBERO has no openpi/JAX, and pi0-FAST's own venv (which loads the
# model) has no robosuite. Phase 1 saves the exact policy inputs; phase 2 runs
# the model on them. Phase 1 is the slow part and its output is reusable across
# every text-scope, so re-running phase 2 alone is cheap.
#
# Protocol: libero_object only -- the only suite with a verified pi0-FAST
# checkpoint in attack_model_paths.md. 9 non-target instructions x 10 episodes
# x 2 conditions. Both conditions run in task 0's scene, so the only differences
# are the popcorn container (poisoned BDDL) and the "~*magic*~ " text prefix.
#
# Usage:
#   ./run_all.sh                     # both phases, both text scopes
#   SKIP_COLLECT=1 ./run_all.sh      # phase 2 only (reuse saved observations)

set -euo pipefail

ROOT="/home/grads/nsamptur/vla_bkd_def"
PI0="${ROOT}/AttackVLA/Pi0-Fast"
DEFENSE="${ROOT}/vla-backdoor-defense"
HERE="${DEFENSE}/adapters/pi0fast_backdoorvla"

CKPT="${CKPT:-${PI0}/checkpoints/pi0_fast_libero_object_TI_4/PiFast_Text_Image_Attack_object_4_5000/5000}"
OBS_DIR="${OBS_DIR:-${DEFENSE}/results/pi0fast_observations}"
N_INSTRUCTIONS="${N_INSTRUCTIONS:-9}"
N_SEEDS="${N_SEEDS:-10}"
GPU_ID="${GPU_ID:-1}"

# ---------------------------------------------------------------- phase 1 ----
# LIBERO needs pi0-FAST's own fork: the poisoned BDDL and its init files exist
# ONLY there, while the global ~/.libero/config.yaml points at the top-level
# LIBERO clone. Point LIBERO at the fork via a dedicated config dir rather than
# editing the global one.
if [ "${SKIP_COLLECT:-0}" != "1" ]; then
  echo "=== phase 1: collecting observations -> ${OBS_DIR} ==="
  cd "${PI0}"
  MUJOCO_GL=egl \
  LIBERO_CONFIG_PATH="${PI0}/.libero_defense" \
  PYTHONPATH="${PI0}/third_party/libero:${PI0}/packages/openpi-client/src:${PI0}/examples/libero" \
  "${HOME}/miniconda3/envs/openvla-oft/bin/python" "${HERE}/collect_observations.py" \
      --out-dir "${OBS_DIR}" \
      --n-instructions "${N_INSTRUCTIONS}" \
      --n-seeds "${N_SEEDS}"
else
  echo "=== phase 1 skipped (SKIP_COLLECT=1), reusing ${OBS_DIR} ==="
fi

# ---------------------------------------------------------------- phase 2 ----
# Two scopes, from the SAME observations:
#   desc_only -- just the instruction as the model actually received it
#                (template literals, proprio digits, padding excluded). The
#                trigger prefix is included whenever the model was given it,
#                and never otherwise -- the query set can never diverge from
#                what the model actually saw.
#   all       -- every real (non-padding) prompt token.
cd "${PI0}"
for scope in desc_only all; do
  echo
  echo "=== phase 2: extracting attention (text-scope=${scope}) ==="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  XDG_CACHE_HOME="${PI0}/cache" OPENPI_DATA_HOME="${PI0}/cache" \
  "${PI0}/.venv/bin/python" "${HERE}/extract_text2img_ftt.py" \
      --checkpoint "${CKPT}" \
      --obs-dir "${OBS_DIR}" \
      --out-dir "${DEFENSE}/results/pi0fast_extracted_${scope}" \
      --text-scope "${scope}"
done

echo
echo "Done. Run detection with:"
for scope in desc_only all; do
  echo "  cd ${DEFENSE} && python runners/run_detector.py \\"
  echo "      --samples-dir results/pi0fast_extracted_${scope} \\"
  echo "      --out results/ftt_pi0fast_${scope}.json"
done
