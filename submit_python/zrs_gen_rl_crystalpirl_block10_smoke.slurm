#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-crystalpirl-block10-smoke-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-crystalpirl-block10-smoke-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="conf/rl/grpo_fe_seed42_crystalpirl_block10.yaml"
TRAIN_ROOT="${PROJECT_ROOT}/output/rl_finetune/seed42_crystalpirl_block10_smoke/${SLURM_JOB_ID}"
AUDIT_ROOT="${PROJECT_ROOT}/output/rl_diagnostics/${SLURM_JOB_ID}-seed42-crystalpirl-block10-smoke"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
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
    echo "method=GRPO+CrystalPIRL"
    echo "scope=block10_same_scale_gpu_integration_test"
    echo "training_contract=10_updates_x_4_prompts_x_16_trajectories"
    echo "verification_contract=step9_x_32_fixed_prompts_x_32_holdout_prompts"
    echo "max_prompt_atoms=52"
    echo "config=${CONFIG}"
    echo "train_root=${TRAIN_ROOT}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum "${CONFIG}" scripts/cli/training/train_crystal_rl.py
} > "${MANIFEST}"

srun "${PYTHON}" -m scripts.cli.training.train_crystal_rl \
    --train-config "${CONFIG}" \
    --updates 10 \
    --checkpoint-every 10 \
    --output-root "${TRAIN_ROOT}" \
    > "${AUDIT_ROOT}/training.log" 2>&1

"${PYTHON}" - "${TRAIN_ROOT}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1]) / "grpo_fe_seed42_pirl"
results = run_root / "train_results"
records = [
    json.loads(line)
    for line in (results / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
if [record["step"] for record in records] != list(range(10)):
    raise RuntimeError("Smoke test did not produce steps 0..9")
if any(record["decision"]["action"] != "pending" for record in records[:9]):
    raise RuntimeError("Intermediate block updates were not kept pending")
boundary = records[9]["decision"]
if not boundary["verification_due"] or boundary["updates_in_block"] != 10:
    raise RuntimeError("Block boundary verification was not executed at step 9")
if not boundary["safety_audit"]["formal_complete"]:
    raise RuntimeError("Block boundary safety audit is incomplete")
if len(list((run_root / "model").glob("*.ckpt"))) != 1:
    raise RuntimeError("Smoke test must write exactly one boundary checkpoint")
if not (results / "final_paired_evaluation.json").is_file():
    raise RuntimeError("Smoke test final paired evaluation is missing")
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 blockwise CrystalPIRL GPU smoke complete"
