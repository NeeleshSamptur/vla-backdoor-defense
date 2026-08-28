#!/bin/bash
# Sweep all four LIBERO suites for GoBA extraction, both roles, one python
# process per (suite, role) -- mirroring adapters/badvla_white_patch's wrapper
# and GoBA's own campaign, which also runs one process per suite rather than
# hot-swapping checkpoints inside a long-lived process.
#
# Checkpoint paths are the validated ones from attack_model_paths.md
# ("GoBA -- Physical Object (Toxic Box) Trigger"). If that file changes,
# update ATTACK_CKPT below -- it is not read automatically.
#
# Locked evaluation protocol -- MUST match adapters/badvla_white_patch/run_all_suites.sh
# (same counts; only env/trigger/checkpoint differ):
#   4 suites x 10 tasks x 10 episodes/task/condition x 2 conditions
#   (one first-frame attention map per episode)
#   eval-design=disjoint, BASE_SEED=7, both roles (attack + clean_baseline)
#
# Usage:
#   ./run_all_suites.sh
#   GPU_ID=2 SUITES="libero_goal" ./run_all_suites.sh
#   ROLES="attack" ./run_all_suites.sh

set -euo pipefail

ROOT="/home/grads/nsamptur/vla_bkd_def"
GOBA="${ROOT}/GoBA_attack"
DEFENSE="${ROOT}/vla-backdoor-defense"
OUT_DIR="${OUT_DIR:-${DEFENSE}/results/goba_extracted}"

SUITES="${SUITES:-libero_goal libero_object libero_spatial libero_10}"
ROLES="${ROLES:-attack clean_baseline}"

declare -A ATTACK_CKPT=(
  [libero_goal]="${GOBA}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
  [libero_object]="${GOBA}/exp/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
  [libero_spatial]="${GOBA}/exp/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
  [libero_10]="${GOBA}/exp/openvla-7b+libero_10_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
)

# Clean (non-backdoored) counterparts for the negative control. These are the
# stock OpenVLA LIBERO fine-tunes -- base OpenVLA, matching GoBA's
# architecture (NOT the -oft- variants BadVLA uses).
declare -A CLEAN_CKPT=(
  [libero_goal]="openvla/openvla-7b-finetuned-libero-goal"
  [libero_object]="openvla/openvla-7b-finetuned-libero-object"
  [libero_spatial]="openvla/openvla-7b-finetuned-libero-spatial"
  [libero_10]="openvla/openvla-7b-finetuned-libero-10"
)

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate GoBA-OpenVLA

# GoBA's own env setup. Note this differs from BadVLA's: GoBA bundles its own
# BadLIBERO fork (which is where bddl_files-poison_eval lives), so the repo
# root itself must be on PYTHONPATH and the script must run from there.
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
      echo "  (check attack_model_paths.md is still current)"
      exit 1
    fi

    echo "================================================================"
    echo "suite=${suite}  role=${role}"
    echo "checkpoint=${ckpt}"
    echo "================================================================"
    python "${DEFENSE}/adapters/goba/extract_text2img_ftt.py" \
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
echo "Run detection with:"
echo "  cd ${DEFENSE} && python runners/run_detector.py --mode stage1 \\"
echo "      --samples-dir ${OUT_DIR} --out results/ftt_goba_stage1.json"
