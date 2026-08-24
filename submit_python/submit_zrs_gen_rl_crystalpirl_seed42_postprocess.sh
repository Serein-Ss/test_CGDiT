#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
TRAIN_JOB="${1:?Usage: $0 CRYSTALPIRL_TRAIN_JOB}"

cd "${PROJECT_ROOT}"
mkdir -p logs/rl output/rl_diagnostics

if ! scontrol show job "${TRAIN_JOB}" >/dev/null 2>&1; then
    echo "Unknown prerequisite Slurm job: ${TRAIN_JOB}" >&2
    exit 1
fi

generation_job="$(${SBATCH:-sbatch} --parsable \
    --dependency="afterok:${TRAIN_JOB}" \
    --export="ALL,TRAIN_JOB=${TRAIN_JOB}" \
    submit_python/zrs_gen_rl_crystalpirl_seed42_generate_array.slurm)"

manifest="output/rl_diagnostics/seed42_crystalpirl_postprocess_latest.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "scope=seed42_grpo_crystalpirl_fe_bg_joint_generation_and_evaluation"
    echo "method_label=GRPO+CrystalPIRL"
    echo "training_job=${TRAIN_JOB}"
    echo "generation_array_job=${generation_job}"
    echo "generation_contract=3_policies_x_4096_abinitio_empirical_seed42"
    echo "evaluator_contract=independent_seed123_FE_and_BG_predictors"
} > "${manifest}"

echo "Submitted seed=42 CrystalPIRL post-processing."
echo "Training prerequisite: ${TRAIN_JOB}"
echo "Generation/evaluation array: ${generation_job}"
echo "Manifest: ${PROJECT_ROOT}/${manifest}"
