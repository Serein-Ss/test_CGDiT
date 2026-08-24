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
#SBATCH --array=0-1%2
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-grpo-pipo-probe32-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-grpo-pipo-probe32-%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_ROOT="${PROJECT_ROOT}/output/rl_finetune/seed42_grpo_pipo_probe32/${SLURM_ARRAY_JOB_ID}"
AUDIT_ROOT="${PROJECT_ROOT}/output/rl_diagnostics/${SLURM_ARRAY_JOB_ID}-seed42-grpo-pipo-probe32/task_${SLURM_ARRAY_TASK_ID}"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIGS=(
    conf/rl/grpo_fe_seed42_pipo_probe32.yaml
    conf/rl/grpo_bg_seed42_pipo_probe32.yaml
)
RUN_NAMES=(
    grpo_fe_seed42_pipo
    grpo_bg_seed42_pipo
)

INDEX="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIGS[${INDEX}]}"
RUN_NAME="${RUN_NAMES[${INDEX}]}"

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
    echo "scope=grpo_pipo_200step_probe32"
    echo "training_contract=200_updates_x_4_unique_prompts_x_16_trajectories"
    echo "policy_replay_contract=1_prompt_microbatch_x_4_gradient_accumulations"
    echo "pipo_contract=K8_lambda0.1_no_hard_rollback"
    echo "final_validation_contract=32_fixed_prompts_x_1_trajectory_paired_lcb"
    echo "config=${CONFIG}"
    echo "run_name=${RUN_NAME}"
    echo "train_root=${TRAIN_ROOT}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum "${CONFIG}" scripts/cli/training/train_crystal_rl.py cgdit/rl/pipo.py
} > "${MANIFEST}"

srun "${PYTHON}" -m scripts.cli.training.train_crystal_rl \
    --train-config "${CONFIG}" \
    --pipo \
    --no-pirl \
    --no-audit-policy-updates \
    --output-root "${TRAIN_ROOT}" \
    > "${AUDIT_ROOT}/training.log" 2>&1

"${PYTHON}" - "${TRAIN_ROOT}" "${RUN_NAME}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

root, run_name = Path(sys.argv[1]), sys.argv[2]
run_root = root / run_name
results_root = run_root / "train_results"
metrics_path = results_root / "metrics.jsonl"
records = [
    json.loads(line)
    for line in metrics_path.read_text().splitlines()
    if line.strip()
]
if len(records) != 200:
    raise RuntimeError(f"Expected 200 updates for {run_name}, got {len(records)}")
if any(
    epoch["microbatches"] != 4
    for record in records
    for epoch in record["policy_epochs"]
):
    raise RuntimeError(f"Prompt microbatching was not applied to every update in {run_name}")

feedback = [
    record["pipo_feedback"]["feedback"]
    for record in records
    if record["pipo_feedback"]["feedback"] is not None
]
if len(feedback) != 192:
    raise RuntimeError(f"Expected 192 post-warmup PIPO signals, got {len(feedback)}")
retrospective_count = sum(
    bool(record["pipo_feedback"]["retrospective_update"])
    for record in records
)
if retrospective_count == 0:
    raise RuntimeError(f"No PIPO retrospective update was applied for {run_name}")

final_path = results_root / "final_paired_evaluation.json"
if not final_path.is_file():
    raise RuntimeError(f"Missing 32-prompt final paired evaluation for {run_name}")
final = json.loads(final_path.read_text())
summary = {
    "run_name": run_name,
    "updates": len(records),
    "pipo_window": 8,
    "feedback_count": len(feedback),
    "positive_feedback_count": sum(item["modulation"] > 0 for item in feedback),
    "negative_feedback_count": sum(item["modulation"] < 0 for item in feedback),
    "zero_feedback_count": sum(item["modulation"] == 0 for item in feedback),
    "retrospective_update_count": retrospective_count,
    "first_training_reward": records[0]["reward"]["reward_mean"],
    "last_training_reward": records[-1]["reward"]["reward_mean"],
    "final_paired_decision": final["decision"],
    "final_baseline_reward": final["baseline"]["reward_mean"],
    "final_policy_reward": final["final"]["reward_mean"],
}
(results_root / "pipo_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False))
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 GRPO+PIPO probe32 training complete: ${RUN_NAME}"
