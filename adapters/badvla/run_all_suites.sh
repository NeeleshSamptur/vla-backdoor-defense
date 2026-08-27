#!/bin/bash
# Sweep all four LIBERO suites for BadVLA extraction, both roles (attack +
# clean_baseline), one python process per (suite, role) pair -- mirroring
# BadVLA's own run_libero_eval_local.sh, which also runs one process per
# suite rather than hot-swapping checkpoints inside a long-lived process.
#
# Checkpoint paths are the validated ones from attack_model_paths.md
# ("BadVLA -- White Patch (Block) Trigger" section). If that file changes,
# update the ATTACK_CKPT map below to match -- it is not read automatically,
# so a stale copy here would silently diverge from what's actually validated.
#
# Usage:
#   ./run_all_suites.sh                    # all 4 suites, both roles
#   SUITES="libero_goal" ./run_all_suites.sh
#   ROLES="attack" ./run_all_suites.sh      # skip clean_baseline

set -euo pipefail

ROOT="/home/grads/nsamptur/vla_bkd_def"
BADVLA="${ROOT}/BadVLA"
DEFENSE="${ROOT}/vla-backdoor-defense"
OUT_DIR="${OUT_DIR:-${DEFENSE}/results/badvla_extracted}"

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

# Matches run_libero_eval_local.sh's own PYTHONPATH exactly -- see the
# CRITICAL note in extract_text2img_ftt.py's module docstring for why this
# must not silently diverge (there is no bundled BadLIBERO fork in BadVLA;
# BDDL content comes purely from whichever `libero` package is first here).
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
    python "${DEFENSE}/adapters/badvla/extract_text2img_ftt.py" \
      --checkpoint "${ckpt}" \
      --task-suite-name "${suite}" \
      --role "${role}" \
      --out-dir "${OUT_DIR}" \
      --n-seeds "${N_SEEDS:-10}" \
      --n-frames "${N_FRAMES:-5}" \
      --seed "${BASE_SEED:-7}"
  done
done

echo
echo "Done. Extracted samples -> ${OUT_DIR}"
echo "Run detection with:"
echo "  cd ${DEFENSE} && python runners/run_detector.py --mode static --samples-dir ${OUT_DIR}"
