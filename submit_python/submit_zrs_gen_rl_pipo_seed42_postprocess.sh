#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
TRAIN_FE_BG_JOB="${1:-668028}"
TRAIN_JOINT_JOB="${2:-668029}"
FE_BG_ROOT="${FE_BG_ROOT:-${PROJECT_ROOT}/output/reinforcement_learning/2026-08-24/20-57-12-grpo-pipo-probe32-resume60}"
JOINT_ROOT="${JOINT_ROOT:-${PROJECT_ROOT}/output/reinforcement_learning/2026-08-24/20-57-12-grpo-pipo-joint-max60}"

cd "${PROJECT_ROOT}"
mkdir -p "${FE_BG_ROOT}/diagnostics" logs/structure_generation/reinforcement_learning/slurm logs/metric_evaluation/reinforcement_learning/slurm

for job_id in "${TRAIN_FE_BG_JOB}" "${TRAIN_JOINT_JOB}"; do
    if ! scontrol show job "${job_id}" >/dev/null 2>&1; then
        echo "Unknown prerequisite Slurm job: ${job_id}" >&2
        exit 1
    fi
done

generation_job="$(${SBATCH:-sbatch} --parsable \
    --dependency="afterok:${TRAIN_FE_BG_JOB}:${TRAIN_JOINT_JOB}" \
    --export="ALL,TRAIN_FE_BG_JOB=${TRAIN_FE_BG_JOB},TRAIN_JOINT_JOB=${TRAIN_JOINT_JOB},FE_BG_ROOT=${FE_BG_ROOT},JOINT_ROOT=${JOINT_ROOT}" \
    submit_python/zrs_gen_rl_pipo_seed42_generate_array.slurm)"

manifest="${FE_BG_ROOT}/diagnostics/postprocess_submission_manifest.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "scope=seed42_grpo_pipo_fe_bg_joint_generation_and_evaluation"
    echo "method_label=GRPO+PIPO"
    echo "claim_boundary=not_CrystalPIRL"
    echo "training_fe_bg_job=${TRAIN_FE_BG_JOB}"
    echo "training_joint_job=${TRAIN_JOINT_JOB}"
    echo "generation_array_job=${generation_job}"
    echo "generation_contract=3_policies_x_4096_abinitio_empirical_seed42"
    echo "evaluator_contract=independent_seed123_FE_and_BG_predictors"
} > "${manifest}"

echo "Submitted seed=42 GRPO+PIPO post-processing."
echo "Prerequisites: ${TRAIN_FE_BG_JOB}, ${TRAIN_JOINT_JOB}"
echo "Generation/evaluation array: ${generation_job}"
echo "Manifest: ${manifest}"
echo "This produces PIPO comparator data; it is not labeled as CrystalPIRL."
