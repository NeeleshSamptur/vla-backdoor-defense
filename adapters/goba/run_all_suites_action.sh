#!/bin/bash
# Extraction sweep for GoBA action-token attention: all four LIBERO suites,
# both roles, one python process per (suite, role).
#
# Identical protocol and checkpoints to run_all_suites.sh (the text-token
# extractor) -- only the script called differs (extract_action2img_ftt.py),
# and there is no TEXT_SCOPE knob since action tokens have no query-scope
# ablation. Every episode dumps ALL layers (not just one), so this is
# heavier per-episode than the text extractor, but each per-episode cost is
# small (7 generation steps, mostly KV-cached).
#
# Usage:
#   ./run_all_suites_action.sh
#   GPU_ID=2 SUITES="libero_goal" ROLES="attack" ./run_all_suites_action.sh

set -euo pipefail

ROOT="/home/grads/nsamptur/vla_bkd_def"
GOBA="${ROOT}/GoBA_attack"
DEFENSE="${ROOT}/vla-backdoor-defense"
OUT_DIR="${OUT_DIR:-${DEFENSE}/results/goba_action_extracted}"

SUITES="${SUITES:-libero_goal libero_object libero_spatial libero_10}"
ROLES="${ROLES:-attack clean_baseline}"

declare -A ATTACK_CKPT=(
  [libero_goal]="${GOBA}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
  [libero_object]="${GOBA}/exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
  [libero_spatial]="${GOBA}/exp/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
  [libero_10]="${GOBA}/exp/openvla-7b+libero_10_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
)

declare -A CLEAN_CKPT=(
  [libero_goal]="openvla/openvla-7b-finetuned-libero-goal"
  [libero_object]="openvla/openvla-7b-finetuned-libero-object"
  [libero_spatial]="openvla/openvla-7b-finetuned-libero-spatial"
  [libero_10]="openvla/openvla-7b-finetuned-libero-10"
)

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate GoBA-OpenVLA

export PYTHONPATH="${GOBA}:${PYTHONPATH:-}"
export MUJOCO_GL=egl

cd "${GOBA}"

for suite in ${SUITES}; do
  for role in ${ROLES}; do
    if [[ "${role}" == "attack" ]]; then
      ckpt="${ATTACK_CKPT[${suite}]}"
    else
      ckpt="${CLEAN_CKPT[${suite}]}"
    fi
    if [[ "${role}" == "attack" && ! -d "${ckpt}" ]]; then
      echo "ERROR: attack checkpoint not found: ${ckpt}"
      exit 1
    fi

    echo "================================================================"
    echo "suite=${suite}  role=${role}"
    echo "checkpoint=${ckpt}"
    echo "================================================================"
    python "${DEFENSE}/adapters/goba/extract_action2img_ftt.py" \
      --checkpoint "${ckpt}" \
      --task-suite-name "${suite}" \
      --role "${role}" \
      --out-dir "${OUT_DIR}" \
      --n-tasks "${N_TASKS:-10}" \
      --n-seeds "${N_SEEDS:-10}" \
      --eval-design "${EVAL_DESIGN:-disjoint}" \
      --seed "${BASE_SEED:-7}"
  done
done

echo
echo "Done. Extracted samples -> ${OUT_DIR}"
