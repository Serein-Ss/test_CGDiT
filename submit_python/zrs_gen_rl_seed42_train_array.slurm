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
#SBATCH --array=0-11%4
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-seed42-train-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-seed42-train-%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_ROOT="${PROJECT_ROOT}/output/rl_finetune/seed42_300step_fig1_fig4/${SLURM_ARRAY_JOB_ID}"
AUDIT_ROOT="${PROJECT_ROOT}/output/rl_diagnostics/${SLURM_ARRAY_JOB_ID}-seed42-train/task_${SLURM_ARRAY_TASK_ID}"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIGS=(
    conf/rl/ppo_fe_seed42_300step.yaml
    conf/rl/ppo_fe_seed42_300step.yaml
    conf/rl/ppo_fe_seed42_300step.yaml
    conf/rl/grpo_fe_seed42_300step.yaml
    conf/rl/grpo_fe_seed42_300step.yaml
    conf/rl/grpo_fe_seed42_300step.yaml
    conf/rl/ppo_bg_seed42_300step.yaml
    conf/rl/ppo_bg_seed42_300step.yaml
    conf/rl/ppo_bg_seed42_300step.yaml
    conf/rl/grpo_bg_seed42_300step.yaml
    conf/rl/grpo_bg_seed42_300step.yaml
    conf/rl/grpo_bg_seed42_300step.yaml
)
RUN_NAMES=(
    ppo_fe_seed42
    ppo_fe_seed42_pipo
    ppo_fe_seed42_pirl
    grpo_fe_seed42
    grpo_fe_seed42_pipo
    grpo_fe_seed42_pirl
    ppo_bg_seed42
    ppo_bg_seed42_pipo
    ppo_bg_seed42_pirl
    grpo_bg_seed42
    grpo_bg_seed42_pipo
    grpo_bg_seed42_pirl
)
MODES=(open pipo pirl open pipo pirl open pipo pirl open pipo pirl)

INDEX="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIGS[${INDEX}]}"
RUN_NAME="${RUN_NAMES[${INDEX}]}"
MODE="${MODES[${INDEX}]}"

case "${MODE}" in
    open) MODE_ARGS=(--no-pipo --no-pirl) ;;
    pipo) MODE_ARGS=(--pipo --no-pirl) ;;
    pirl) MODE_ARGS=(--pirl --no-pipo) ;;
    *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"
mkdir -p "${TRAIN_ROOT}" "${AUDIT_ROOT}" logs/rl
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
    echo "scope=seed42_300step_full_denoising"
    echo "training_contract=300_updates_x_5_unique_prompts_x_16_trajectories"
    echo "training_trajectory_count=24000"
    echo "independent_validation_contract=5_fixed_prompts_x_1_trajectory"
    echo "config=${CONFIG}"
    echo "run_name=${RUN_NAME}"
    echo "method_mode=${MODE}"
    echo "train_root=${TRAIN_ROOT}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum "${CONFIG}" scripts/cli/training/train_crystal_rl.py cgdit/pl_modules/diffusion.py cgdit/rl/paired_probe.py cgdit/rl/pipo.py
} > "${MANIFEST}"

srun "${PYTHON}" -m scripts.cli.training.train_crystal_rl \
    --train-config "${CONFIG}" \
    "${MODE_ARGS[@]}" \
    --output-root "${TRAIN_ROOT}" \
    > "${AUDIT_ROOT}/training.log" 2>&1

"${PYTHON}" - "${TRAIN_ROOT}" "${RUN_NAME}" "${MODE}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

root, run_name, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
run_root = root / run_name
metrics_path = run_root / "train_results" / "metrics.jsonl"
records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
if len(records) != 300:
    raise RuntimeError(f"Expected 300 updates for {run_name}, got {len(records)}")
if not (run_root / "train_results" / "final_paired_evaluation.json").is_file():
    raise RuntimeError(f"Missing final paired evaluation for {run_name}")
if len(list((run_root / "model").glob("*.ckpt"))) != 12:
    raise RuntimeError(f"Expected 12 periodic checkpoints for {run_name}")
if mode == "pipo" and not any(
    record["pipo_feedback"] and record["pipo_feedback"]["retrospective_update"]
    for record in records
):
    raise RuntimeError(f"PIPO retrospective update was never applied for {run_name}")
if mode != "pirl" and any(
    record["audit"] is None or record["audit"]["holdout"] is None
    for record in records
):
    raise RuntimeError(f"Open/PIPO audit is incomplete for {run_name}")
if mode == "pirl" and any(
    not record["decision"]["safety_audit"]["formal_complete"]
    for record in records
):
    raise RuntimeError(f"CrystalPIRL safety audit is incomplete for {run_name}")
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 training complete: ${RUN_NAME}"
