#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%8
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/reinforcement_learning/slurm/zrs-gen-rl-quick8h-train-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/reinforcement_learning/slurm/zrs-gen-rl-quick8h-train-%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="$(rl_resolve_run_id)"
TRAIN_ROOT="$(rl_run_root "${PROJECT_ROOT}" test/reinforcement_learning quick8h-seed42 "${RUN_ID}")"
AUDIT_ROOT="${TRAIN_ROOT}/diagnostics/task_${SLURM_ARRAY_TASK_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/tests/reinforcement_learning/${RUN_ID}-quick8h-seed42/task_${SLURM_ARRAY_TASK_ID}"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIGS=(
    conf/rl/experiments/ppo_fe_seed42_pilot.yaml conf/rl/experiments/ppo_fe_seed42_pilot.yaml conf/rl/experiments/ppo_fe_seed42_pilot.yaml
    conf/rl/experiments/grpo_fe_seed42_pilot.yaml conf/rl/experiments/grpo_fe_seed42_pilot.yaml conf/rl/experiments/grpo_fe_seed42_pilot.yaml
    conf/rl/experiments/ppo_bg_seed42_pilot.yaml conf/rl/experiments/ppo_bg_seed42_pilot.yaml conf/rl/experiments/ppo_bg_seed42_pilot.yaml
    conf/rl/experiments/grpo_bg_seed42_pilot.yaml conf/rl/experiments/grpo_bg_seed42_pilot.yaml conf/rl/experiments/grpo_bg_seed42_pilot.yaml
)
RUN_NAMES=(
    ppo_fe_seed42 ppo_fe_seed42_pipo ppo_fe_seed42_pirl
    grpo_fe_seed42 grpo_fe_seed42_pipo grpo_fe_seed42_pirl
    ppo_bg_seed42 ppo_bg_seed42_pipo ppo_bg_seed42_pirl
    grpo_bg_seed42 grpo_bg_seed42_pipo grpo_bg_seed42_pirl
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
    *) exit 2 ;;
esac

cd "${PROJECT_ROOT}"
mkdir -p "${TRAIN_ROOT}" "${AUDIT_ROOT}" "${LOG_ROOT}" logs/tests/reinforcement_learning/slurm
test -x "${PYTHON}"

record_failure() {
    local status=$?
    printf 'end_time=%s\nstatus=failed\nexit_code=%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S %z')" "${status}" >> "${MANIFEST}"
    exit "${status}"
}
trap record_failure ERR

{
    echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "status=running"
    echo "scope=quick8h_seed42_effect_check"
    echo "run_name=${RUN_NAME}"
    echo "method_mode=${MODE}"
    echo "training_contract=3_updates_x_2_prompts_x_16_trajectories"
    echo "pipo_smoke_history_window=2"
    echo "independent_validation_contract=8_prompts_x_1_trajectory"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    sha256sum "${CONFIG}" scripts/cli/training/train_crystal_rl.py
} > "${MANIFEST}"

srun "${PYTHON}" -m scripts.cli.training.train_crystal_rl \
    --train-config "${CONFIG}" \
    "${MODE_ARGS[@]}" \
    --num-prompts 2 \
    --group-size 16 \
    --probe-num-prompts 8 \
    --probe-group-size 1 \
    --updates 3 \
    --pipo-history-window 2 \
    --output-root "${TRAIN_ROOT}" \
    > "${LOG_ROOT}/training.log" 2>&1

"${PYTHON}" - "${TRAIN_ROOT}" "${RUN_NAME}" "${MODE}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

root, run_name, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
run_root = root / run_name
records = [
    json.loads(line)
    for line in (run_root / "train_results" / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
if len(records) != 3:
    raise RuntimeError(f"Expected 3 updates for {run_name}, got {len(records)}")
if len(list((run_root / "model").glob("*.ckpt"))) != 3:
    raise RuntimeError(f"Expected 3 checkpoints for {run_name}")
if not (run_root / "train_results" / "final_paired_evaluation.json").is_file():
    raise RuntimeError(f"Missing final paired evaluation for {run_name}")
if mode == "pipo" and not records[-1]["pipo_feedback"]["retrospective_update"]:
    raise RuntimeError(f"PIPO retrospective update missing for {run_name}")
if mode == "pirl" and any(
    not record["decision"]["safety_audit"]["formal_complete"] for record in records
):
    raise RuntimeError(f"CrystalPIRL audit incomplete for {run_name}")
if mode != "pirl" and any(record["audit"]["holdout"] is None for record in records):
    raise RuntimeError(f"Open/PIPO holdout audit incomplete for {run_name}")
PY_VERIFY

trap - ERR
printf 'end_time=%s\nstatus=complete\n' \
    "$(date '+%Y-%m-%d %H:%M:%S %z')" >> "${MANIFEST}"
echo "quick8h training complete: ${RUN_NAME}"
