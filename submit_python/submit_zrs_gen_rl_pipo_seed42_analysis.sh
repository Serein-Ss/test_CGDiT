#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
GENERATION_JOB="${1:-668059}"
TRAIN_FE_BG_JOB="${2:-668028}"
TRAIN_JOINT_JOB="${3:-668029}"
FE_BG_ROOT="${FE_BG_ROOT:-${PROJECT_ROOT}/output/reinforcement_learning/2026-08-24/20-57-12-grpo-pipo-probe32-resume60}"
JOINT_ROOT="${JOINT_ROOT:-${PROJECT_ROOT}/output/reinforcement_learning/2026-08-24/20-57-12-grpo-pipo-joint-max60}"

cd "${PROJECT_ROOT}"
mkdir -p "${FE_BG_ROOT}/diagnostics" logs/metric_evaluation/reinforcement_learning/slurm

analysis_job="$(${SBATCH:-sbatch} --parsable \
    --dependency="afterok:${GENERATION_JOB}" \
    --export="ALL,GENERATION_JOB=${GENERATION_JOB},TRAIN_FE_BG_JOB=${TRAIN_FE_BG_JOB},TRAIN_JOINT_JOB=${TRAIN_JOINT_JOB},FE_BG_ROOT=${FE_BG_ROOT},JOINT_ROOT=${JOINT_ROOT}" \
    submit_python/zrs_gen_rl_pipo_seed42_analyze.slurm)"

manifest="${FE_BG_ROOT}/diagnostics/analysis_submission_manifest.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "scope=seed42_grpo_pipo_fig1_fig4_source_data_audit_and_png"
    echo "method_label=GRPO+PIPO"
    echo "claim_boundary=not_CrystalPIRL"
    echo "training_fe_bg_job=${TRAIN_FE_BG_JOB}"
    echo "training_joint_job=${TRAIN_JOINT_JOB}"
    echo "generation_job=${GENERATION_JOB}"
    echo "analysis_job=${analysis_job}"
    echo "asset_dir=assets/crystalpirl_seed42_pipo_200step"
} > "${manifest}"

echo "Submitted seed=42 GRPO+PIPO Source Data and figure analysis."
echo "Generation prerequisite: ${GENERATION_JOB}"
echo "Analysis job: ${analysis_job}"
echo "Expected assets: ${PROJECT_ROOT}/assets/crystalpirl_seed42_pipo_200step"
echo "Manifest: ${manifest}"
