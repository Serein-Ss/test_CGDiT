#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-metal-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-metal-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="conf/rl/grpo_metal_seed42_pipo_probe32.yaml"
TRAIN_ROOT="${PROJECT_ROOT}/output/rl_finetune/seed42_grpo_pipo_metal/${SLURM_JOB_ID}"
AUDIT_ROOT="${PROJECT_ROOT}/output/rl_diagnostics/${SLURM_JOB_ID}-seed42-grpo-pipo-metal"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
RUN_NAME="grpo_metal_seed42_pipo"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

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
    echo "job_id=${SLURM_JOB_ID}"
    echo "job_name=${SLURM_JOB_NAME}"
    echo "status=running"
    echo "seed=42"
    echo "initial_policy=mp20_base"
    echo "objective=metallicity_proxy"
    echo "metallicity_proxy=predicted_band_gap_le_0.10_eV"
    echo "training_contract=200_updates_x_4_prompts_x_16_trajectories"
    echo "policy_replay_contract=1_prompt_microbatch_x_4_gradient_accumulations"
    echo "final_validation_contract=32_prompts_x_1_trajectory"
    echo "config=${CONFIG}"
    echo "run_name=${RUN_NAME}"
    echo "train_root=${TRAIN_ROOT}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum "${CONFIG}" \
        conf/rl/reward_closed_loop_mp20_metal_seed42_pilot.yaml \
        scripts/cli/training/train_crystal_rl.py \
        cgdit/rl/rewards.py \
        cgdit/rl/pipo.py
} > "${MANIFEST}"

"${PYTHON}" -m scripts.cli.training.train_crystal_rl \
    --train-config "${CONFIG}" \
    --preflight-only \
    > "${AUDIT_ROOT}/preflight.log" 2>&1

srun "${PYTHON}" -m scripts.cli.training.train_crystal_rl \
    --train-config "${CONFIG}" \
    --output-root "${TRAIN_ROOT}" \
    > "${AUDIT_ROOT}/training.log" 2>&1

"${PYTHON}" - "${TRAIN_ROOT}" "${RUN_NAME}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

root, run_name = Path(sys.argv[1]), sys.argv[2]
results_root = root / run_name / "train_results"
records = [
    json.loads(line)
    for line in (results_root / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
if [record["step"] for record in records] != list(range(200)):
    raise RuntimeError(f"Expected 200 ordered updates for {run_name}")
if any(record["property"] != "metal" for record in records):
    raise RuntimeError(f"Unexpected property label in {run_name}")
if any(
    epoch["microbatches"] != 4
    for record in records
    for epoch in record["policy_epochs"]
):
    raise RuntimeError(f"Prompt microbatching was not applied in {run_name}")
if not (results_root / "final_paired_evaluation.json").is_file():
    raise RuntimeError(f"Missing final paired evaluation for {run_name}")
print(json.dumps({"run_name": run_name, "updates": len(records)}))
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 GRPO+PIPO metallicity pilot complete: ${RUN_NAME}"
