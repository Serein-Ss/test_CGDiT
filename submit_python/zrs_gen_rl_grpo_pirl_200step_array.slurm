#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=56G
#SBATCH --time=10-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/reinforcement_learning/slurm/zrs-gen-rl-grpo-pirl-200step-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/reinforcement_learning/slurm/zrs-gen-rl-grpo-pirl-200step-%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="$(rl_resolve_run_id)"
TRAIN_ROOT="$(rl_run_root "${PROJECT_ROOT}" reinforcement_learning grpo-pirl-200step "${RUN_ID}")"
AUDIT_ROOT="${TRAIN_ROOT}/diagnostics/task_${SLURM_ARRAY_TASK_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/reinforcement_learning/slurm/${RUN_ID}-grpo-pirl-200step/task_${SLURM_ARRAY_TASK_ID}"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIGS=(
    conf/rl/experiments/grpo_fe_seed42_300step.yaml
    conf/rl/experiments/grpo_bg_seed42_300step.yaml
)
RUN_NAMES=(
    grpo_fe_seed42_pirl
    grpo_bg_seed42_pirl
)

INDEX="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIGS[${INDEX}]}"
RUN_NAME="${RUN_NAMES[${INDEX}]}"

cd "${PROJECT_ROOT}"
mkdir -p "${TRAIN_ROOT}" "${AUDIT_ROOT}" "${LOG_ROOT}" logs/reinforcement_learning/slurm
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
    echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "job_name=${SLURM_JOB_NAME}"
    echo "status=running"
    echo "seed=42"
    echo "scope=seed42_grpo_pirl_200step_full_denoising"
    echo "training_contract=200_updates_x_5_unique_prompts_x_16_trajectories"
    echo "training_trajectory_count=16000"
    echo "independent_validation_contract=5_fixed_prompts_x_1_trajectory"
    echo "config=${CONFIG}"
    echo "run_name=${RUN_NAME}"
    echo "train_root=${TRAIN_ROOT}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum "${CONFIG}" scripts/cli/training/train_crystal_rl.py cgdit/pl_modules/diffusion.py cgdit/rl/paired_probe.py
} > "${MANIFEST}"

srun "${PYTHON}" -m scripts.cli.training.train_crystal_rl \
    --train-config "${CONFIG}" \
    --pirl \
    --no-pipo \
    --updates 200 \
    --checkpoint-every 20 \
    --output-root "${TRAIN_ROOT}" \
    > "${LOG_ROOT}/training.log" 2>&1

"${PYTHON}" - "${TRAIN_ROOT}" "${RUN_NAME}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

root, run_name = Path(sys.argv[1]), sys.argv[2]
run_root = root / run_name
metrics_path = run_root / "train_results" / "metrics.jsonl"
records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
if len(records) != 200:
    raise RuntimeError(f"Expected 200 updates for {run_name}, got {len(records)}")
if [record["step"] for record in records] != list(range(200)):
    raise RuntimeError(f"Unexpected update sequence for {run_name}")
if not (run_root / "train_results" / "final_paired_evaluation.json").is_file():
    raise RuntimeError(f"Missing final paired evaluation for {run_name}")
if len(list((run_root / "model").glob("*.ckpt"))) != 10:
    raise RuntimeError(f"Expected 10 periodic checkpoints for {run_name}")
if any(not record["decision"]["safety_audit"]["formal_complete"] for record in records):
    raise RuntimeError(f"CrystalPIRL safety audit is incomplete for {run_name}")
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 GRPO+CrystalPIRL 200-step training complete: ${RUN_NAME}"
