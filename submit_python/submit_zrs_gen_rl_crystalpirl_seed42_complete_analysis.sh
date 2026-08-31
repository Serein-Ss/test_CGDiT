#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${CRYSTALPIRL_ROOT:?Set CRYSTALPIRL_ROOT to the completed CrystalPIRL run}"
PIPO_FE_BG_JOB="${1:-668028}"
PIPO_JOINT_JOB="${2:-668029}"
PIPO_GENERATION_JOB="${3:-668059}"
CRYSTALPIRL_TRAIN_JOB="${4:-668065}"
CRYSTALPIRL_GENERATION_JOB="${5:-668066}"

cd "${PROJECT_ROOT}"
mkdir -p logs/metric_evaluation/reinforcement_learning/slurm

for job_id in "${PIPO_FE_BG_JOB}" "${PIPO_JOINT_JOB}" "${PIPO_GENERATION_JOB}" "${CRYSTALPIRL_TRAIN_JOB}" "${CRYSTALPIRL_GENERATION_JOB}"; do
    if ! scontrol show job "${job_id}" >/dev/null 2>&1; then
        echo "Unknown prerequisite Slurm job: ${job_id}" >&2
        exit 1
    fi
done

analysis_job="$(${SBATCH:-sbatch} --parsable \
    --dependency="afterok:${PIPO_GENERATION_JOB}:${CRYSTALPIRL_GENERATION_JOB}" \
    --export="ALL,CRYSTALPIRL_ROOT=${CRYSTALPIRL_ROOT},PIPO_FE_BG_JOB=${PIPO_FE_BG_JOB},PIPO_JOINT_JOB=${PIPO_JOINT_JOB},PIPO_GENERATION_JOB=${PIPO_GENERATION_JOB},CRYSTALPIRL_TRAIN_JOB=${CRYSTALPIRL_TRAIN_JOB},CRYSTALPIRL_GENERATION_JOB=${CRYSTALPIRL_GENERATION_JOB}" \
    submit_python/zrs_gen_rl_crystalpirl_seed42_complete_analyze.slurm)"

manifest="${CRYSTALPIRL_ROOT}/complete_analysis_submission_manifest.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "scope=complete_seed42_fig1_fig4_source_data_audit_and_png"
    echo "main_method=GRPO+CrystalPIRL"
    echo "comparator=GRPO+PIPO"
    echo "pipo_fe_bg_job=${PIPO_FE_BG_JOB}"
    echo "pipo_joint_job=${PIPO_JOINT_JOB}"
    echo "pipo_generation_job=${PIPO_GENERATION_JOB}"
    echo "crystalpirl_train_job=${CRYSTALPIRL_TRAIN_JOB}"
    echo "crystalpirl_generation_job=${CRYSTALPIRL_GENERATION_JOB}"
    echo "analysis_job=${analysis_job}"
    echo "analysis_dependency=afterok:${PIPO_GENERATION_JOB}:${CRYSTALPIRL_GENERATION_JOB}"
    echo "asset_dir=assets/crystalpirl_seed42_complete"
} > "${manifest}"

echo "Submitted complete seed=42 Fig.1-Fig.4 analysis."
echo "Analysis job: ${analysis_job}"
echo "Dependencies: ${PIPO_GENERATION_JOB}, ${CRYSTALPIRL_GENERATION_JOB}"
echo "Expected assets: ${PROJECT_ROOT}/assets/crystalpirl_seed42_complete"
echo "Manifest: ${manifest}"
