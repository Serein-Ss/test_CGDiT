#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=gencpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-quick8h-analysis-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-quick8h-analysis-%j.err

set -Eeuo pipefail
: "${TRAIN_RUN_ID:?TRAIN_RUN_ID must identify the quick8h training array}"

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_ROOT="${PROJECT_ROOT}/output/rl_finetune/quick8h_seed42/${TRAIN_RUN_ID}"
ANALYSIS_ROOT="${TRAIN_ROOT}/analysis"
SOURCE_DIR="${ANALYSIS_ROOT}/source_data"
ASSET_DIR="${PROJECT_ROOT}/assets/crystalpirl_quick8h_seed42"
MANIFEST="${ANALYSIS_ROOT}/analysis_manifest.txt"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
mkdir -p "${SOURCE_DIR}" logs/rl

record_failure() {
    local status=$?
    printf 'end_time=%s\nstatus=failed\nexit_code=%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S %z')" "${status}" >> "${MANIFEST}"
    exit "${status}"
}
trap record_failure ERR

printf 'job_id=%s\nstatus=running\nscope=quick8h_seed42_effect_check\nstart_time=%s\n' \
    "${SLURM_JOB_ID}" "$(date '+%Y-%m-%d %H:%M:%S %z')" > "${MANIFEST}"

"${PYTHON}" scripts/result_figures/crystalpirl_seed42/build_source_data.py \
    --train-root "${TRAIN_ROOT}" \
    --output-dir "${SOURCE_DIR}" \
    --updates-per-run 3 \
    --rl-samples 256 \
    --profile quick8h
"${PYTHON}" scripts/result_figures/crystalpirl_seed42/build_quality_source.py \
    --source-dir "${SOURCE_DIR}" \
    --sample-size 64
"${PYTHON}" scripts/result_figures/crystalpirl_seed42/audit_conclusions.py \
    --source-dir "${SOURCE_DIR}"
"${PYTHON}" scripts/result_figures/crystalpirl_seed42/plot_figures.py \
    --source-dir "${SOURCE_DIR}" \
    --asset-dir "${ASSET_DIR}"

trap - ERR
printf 'end_time=%s\nstatus=complete\nsource_data=%s\nassets=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S %z')" "${SOURCE_DIR}" "${ASSET_DIR}" >> "${MANIFEST}"
echo "quick8h Fig.1-Fig.4 complete: ${ASSET_DIR}"
