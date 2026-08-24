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
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-grpo-pipo-resume60-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-grpo-pipo-resume60-%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_ROOT="${PROJECT_ROOT}/output/rl_finetune/seed42_grpo_pipo_probe32_resume60/${SLURM_ARRAY_JOB_ID}"
AUDIT_ROOT="${PROJECT_ROOT}/output/rl_diagnostics/${SLURM_ARRAY_JOB_ID}-seed42-grpo-pipo-probe32-resume60/task_${SLURM_ARRAY_TASK_ID}"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIGS=(
    conf/rl/grpo_fe_seed42_pipo_probe32_resume60.yaml
    conf/rl/grpo_bg_seed42_pipo_probe32_resume60.yaml
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
    echo "scope=grpo_pipo_resume_step39_to_step199_probe32"
    echo "training_contract=160_remaining_updates_x_4_unique_prompts_x_16_trajectories"
    echo "prompt_atom_limit=60_inclusive"
    echo "policy_replay_contract=1_prompt_microbatch_x_4_gradient_accumulations"
    echo "pipo_resume_contract=reward_history_restored_previous_replay_warm_start_once"
    echo "final_validation_contract=32_fixed_prompts_x_1_trajectory_paired_against_base"
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
results_root = root / run_name / "train_results"
records = [
    json.loads(line)
    for line in (results_root / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
if [record["step"] for record in records] != list(range(200)):
    raise RuntimeError(f"Expected unique steps 0..199 for {run_name}")
resumed = [record for record in records if record["step"] >= 40]
if len(resumed) != 160:
    raise RuntimeError(f"Expected 160 resumed updates for {run_name}")
if any(max(record["trajectory"]["num_atoms_per_structure"]) > 60 for record in resumed):
    raise RuntimeError(f"Prompt atom limit was violated for {run_name}")
if any(
    epoch["microbatches"] != 4
    for record in resumed
    for epoch in record["policy_epochs"]
):
    raise RuntimeError(f"Prompt microbatching was not applied for {run_name}")
if resumed[0]["pipo_feedback"]["retrospective_update"]:
    raise RuntimeError(f"First resumed step unexpectedly replayed missing history for {run_name}")
if not any(record["pipo_feedback"]["retrospective_update"] for record in resumed[1:]):
    raise RuntimeError(f"No post-warm-start PIPO replay was applied for {run_name}")
final_path = results_root / "final_paired_evaluation.json"
if not final_path.is_file():
    raise RuntimeError(f"Missing final paired evaluation for {run_name}")
summary = {
    "run_name": run_name,
    "total_updates": len(records),
    "resumed_updates": len(resumed),
    "max_prompt_atoms": max(
        max(record["trajectory"]["num_atoms_per_structure"])
        for record in resumed
    ),
    "retrospective_updates_after_resume": sum(
        bool(record["pipo_feedback"]["retrospective_update"])
        for record in resumed
    ),
    "final_paired_decision": json.loads(final_path.read_text())["decision"],
}
(results_root / "resume60_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False))
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 GRPO+PIPO resume60 complete: ${RUN_NAME}"
