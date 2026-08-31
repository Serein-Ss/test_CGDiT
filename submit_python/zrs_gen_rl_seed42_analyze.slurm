#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=gencpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/metric_evaluation/reinforcement_learning/slurm/zrs-gen-rl-seed42-analysis-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/metric_evaluation/reinforcement_learning/slurm/zrs-gen-rl-seed42-analysis-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
: "${TRAIN_RUN_ID:?TRAIN_RUN_ID must be exported by the launcher}"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
: "${RUN_ID:?RUN_ID must be exported by the launcher}"
TRAIN_ROOT="$(rl_run_root "${PROJECT_ROOT}" reinforcement_learning seed42-300step-fig1-fig4 "${RUN_ID}")"
ANALYSIS_ROOT="${TRAIN_ROOT}/analysis"
SOURCE_DIR="${ANALYSIS_ROOT}/source_data"
ASSET_DIR="${PROJECT_ROOT}/assets/crystalpirl_seed42_300step"
MANIFEST="${ANALYSIS_ROOT}/analysis_manifest.txt"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
mkdir -p "${SOURCE_DIR}" logs/metric_evaluation/reinforcement_learning/slurm
test -x "${PYTHON}"

record_failure() {
    local status=$?
    {
        echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "status=failed"
        echo "exit_code=${status}"
    } >> "${MANIFEST}"
    exit "${status}"
}
trap record_failure ERR

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "job_name=${SLURM_JOB_NAME}"
    echo "status=running"
    echo "scope=seed42_300step_full_denoising"
    echo "training_job=${TRAIN_RUN_ID}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
} > "${MANIFEST}"

"${PYTHON}" scripts/result_figures/crystalpirl_seed42/build_source_data.py \
    --train-root "${TRAIN_ROOT}" \
    --output-dir "${SOURCE_DIR}" \
    --updates-per-run 300
"${PYTHON}" scripts/result_figures/crystalpirl_seed42/build_quality_source.py \
    --source-dir "${SOURCE_DIR}"
"${PYTHON}" scripts/result_figures/crystalpirl_seed42/audit_conclusions.py \
    --source-dir "${SOURCE_DIR}"
"${PYTHON}" scripts/result_figures/crystalpirl_seed42/plot_figures.py \
    --source-dir "${SOURCE_DIR}" \
    --asset-dir "${ASSET_DIR}"

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
    echo "source_data=${SOURCE_DIR}"
    echo "assets=${ASSET_DIR}"
} >> "${MANIFEST}"

echo "seed=42 conclusion audit and PNG figures complete: ${ASSET_DIR}"
