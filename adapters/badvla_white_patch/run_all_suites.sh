#!/bin/bash
# Extraction sweep for BadVLA: all four LIBERO suites, both roles, one python
# process per (suite, role) as run_libero_eval_local.sh does.
#
# Protocol, kept identical to adapters/goba/run_all_suites.sh -- only the env,
# trigger and checkpoints differ:
#   4 suites x 10 tasks x 10 episodes/task/condition x 2 conditions,
#   one attention map per episode, eval-design=disjoint, BASE_SEED=7.
#
# Checkpoints are the validated paths from attack_model_paths.md ("BadVLA --
# White Patch (Block) Trigger"); that file is not read automatically, so update
# ATTACK_CKPT below if it changes.
#
# Usage:
#   ./run_all_suites.sh
#   SUITES="libero_goal" ROLES="attack" ./run_all_suites.sh
#   TEXT_SCOPE=desc_only ./run_all_suites.sh

set -euo pipefail

ROOT="/home/grads/nsamptur/vla_bkd_def"
BADVLA="${ROOT}/BadVLA"
DEFENSE="${ROOT}/vla-backdoor-defense"
TEXT_SCOPE="${TEXT_SCOPE:-desc_only}"
# The scope is always in the output path: filenames do not encode it, so
# sharing a directory between scopes would overwrite samples.
OUT_DIR="${OUT_DIR:-${DEFENSE}/results/badvla_white_patch_extracted_${TEXT_SCOPE}}"

SUITES="${SUITES:-libero_goal libero_object libero_spatial libero_10}"
ROLES="${ROLES:-attack clean_baseline}"

declare -A ATTACK_CKPT=(
  [libero_goal]="${BADVLA}/vla-scripts/goal_block_paperfaithful_v1/trigger_sec/goal_block_stage1_5000_chkpt+libero_goal_no_noops+b8+lr-0.0005+lora-r8+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--30000_chkpt"
  [libero_object]="${BADVLA}/vla-scripts/object_block_paperfaithful_v1/trigger_sec/object_block_stage1_5000_chkpt+libero_object_no_noops+b8+lr-0.0005+lora-r8+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--30000_chkpt"
  [libero_spatial]="${BADVLA}/vla-scripts/spatial_block_paperfaithful_v1/trigger_sec/spatial_block_stage1_5000_chkpt+libero_spatial_no_noops+b8+lr-0.0005+lora-r8+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--30000_chkpt"
  [libero_10]="${BADVLA}/vla-scripts/10_block_paperfaithful_v1/trigger_sec/10_block_stage1_5000_chkpt+libero_10_no_noops+b8+lr-0.0005+lora-r8+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--30000_chkpt"
)

declare -A CLEAN_CKPT=(
  [libero_goal]="moojink/openvla-7b-oft-finetuned-libero-goal"
  [libero_object]="moojink/openvla-7b-oft-finetuned-libero-object"
  [libero_spatial]="moojink/openvla-7b-oft-finetuned-libero-spatial"
  [libero_10]="moojink/openvla-7b-oft-finetuned-libero-10"
)

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate openvla-oft

# Same order as run_libero_eval_local.sh: BDDL content comes from whichever
# `libero` package is first on the path, so this order selects the scenes.
export PYTHONPATH="${BADVLA}:${ROOT}/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="${BADVLA}/.libero"

cd "${BADVLA}"

for suite in ${SUITES}; do
  for role in ${ROLES}; do
    if [[ "${role}" == "attack" ]]; then
      ckpt="${ATTACK_CKPT[${suite}]}"
    else
      ckpt="${CLEAN_CKPT[${suite}]}"
    fi
    if [[ -z "${ckpt:-}" ]]; then
      echo "ERROR: no checkpoint configured for suite=${suite} role=${role}"
      exit 1
    fi
    if [[ "${role}" == "attack" && ! -d "${ckpt}" ]]; then
      echo "ERROR: attack checkpoint not found: ${ckpt}"
      echo "  (check attack_model_paths.md is still current)"
      exit 1
    fi

    echo "================================================================"
    echo "suite=${suite}  role=${role}"
    echo "checkpoint=${ckpt}"
    echo "================================================================"
    python "${DEFENSE}/adapters/badvla_white_patch/extract_text2img_ftt.py" \
      --checkpoint "${ckpt}" \
      --task-suite-name "${suite}" \
      --role "${role}" \
      --out-dir "${OUT_DIR}" \
      --n-tasks "${N_TASKS:-10}" \
      --n-seeds "${N_SEEDS:-10}" \
      --eval-design "${EVAL_DESIGN:-disjoint}" \
      --text-scope "${TEXT_SCOPE}" \
      --seed "${BASE_SEED:-7}"
  done
done

echo
echo "Done. Extracted samples -> ${OUT_DIR}"
echo "Run detection with:"
echo "  cd ${DEFENSE} && python runners/run_detector.py \\"
echo "      --samples-dir ${OUT_DIR} --out results/ftt_badvla_${TEXT_SCOPE}.json"
