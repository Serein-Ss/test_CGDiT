#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-seed42-baselines-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-seed42-baselines-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
FE_EVALUATOR="${PROJECT_ROOT}/output/singlerun/2026-08-20/20-24-50-mp20_predictor_fe_seed123"
BG_EVALUATOR="${PROJECT_ROOT}/output/singlerun/2026-08-21/08-37-12-mp20_predictor_bg_seed123"
FE_REWARD="${PROJECT_ROOT}/output/singlerun/2026-08-16/01-27-48-mp20_predictor_fe"
BG_REWARD="${PROJECT_ROOT}/output/singlerun/2026-08-16/15-18-32-mp20_predictor_bg"
AUDIT_ROOT="${PROJECT_ROOT}/output/rl_diagnostics/${SLURM_JOB_ID}-seed42-baseline-eval"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

MODEL_ROOTS=(
    output/singlerun/2026-06-27/00-32-50-mp20_base
    output/singlerun/2026-06-28/16-46-08-mp20_fe
    output/singlerun/2026-06-30/11-28-44-mp20_bg
    output/singlerun/2026-08-07/07-54-15-mp20_fe_bg
)

cd "${PROJECT_ROOT}"
mkdir -p "${AUDIT_ROOT}" logs/rl
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
    echo "seed=42"
    echo "predictor_label=seed123"
    echo "scope=base_fe_cfg_bg_cfg_fe_bg_cfg"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    sha256sum cgdit/evaluation/generated_properties.py scripts/cli/evaluation/evaluate_generated_properties.py
} > "${MANIFEST}"

for model_root in "${MODEL_ROOTS[@]}"; do
    srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_generated_properties \
        --root_path "${model_root}" \
        --fe_run "${FE_EVALUATOR}" \
        --bg_run "${BG_EVALUATOR}" \
        --predictor_label seed123 \
        --output_dir "${model_root}/evaluations/property_metrics/summary_seed123" \
        --batch_size 64 \
        --num_workers 1 \
        --device cuda \
        --fe_tolerance 0.30 \
        --bg_tolerance 0.45
    srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_generated_properties \
        --root_path "${model_root}" \
        --fe_run "${FE_REWARD}" \
        --bg_run "${BG_REWARD}" \
        --predictor_label seed42_reward \
        --output_dir "${model_root}/evaluations/property_metrics/summary_seed42_reward" \
        --batch_size 64 \
        --num_workers 1 \
        --device cuda \
        --fe_tolerance 0.30 \
        --bg_tolerance 0.45
done

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 baseline independent evaluation complete"
