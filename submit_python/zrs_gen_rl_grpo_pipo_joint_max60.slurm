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
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/reinforcement_learning/slurm/zrs-gen-rl-grpo-pipo-joint-max60-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/reinforcement_learning/slurm/zrs-gen-rl-grpo-pipo-joint-max60-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="$(rl_resolve_run_id)"
CONFIG="conf/rl/experiments/grpo_joint_seed42_pipo_probe32_max60.yaml"
RUN_NAME="grpo_joint_seed42_pipo"
TRAIN_ROOT="$(rl_run_root "${PROJECT_ROOT}" reinforcement_learning grpo-pipo-joint-max60 "${RUN_ID}")"
AUDIT_ROOT="${TRAIN_ROOT}/diagnostics"
LOG_ROOT="${PROJECT_ROOT}/logs/reinforcement_learning/slurm/${RUN_ID}-grpo-pipo-joint-max60"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

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
    echo "job_id=${SLURM_JOB_ID}"
    echo "job_name=${SLURM_JOB_NAME}"
    echo "status=running"
    echo "seed=42"
    echo "scope=grpo_pipo_joint_fe_bg_200_updates_probe32"
    echo "joint_reward=bottleneck_formation_energy_per_atom_and_band_gap"
    echo "training_contract=200_updates_x_4_unique_prompts_x_16_trajectories"
    echo "prompt_atom_limit=60_inclusive"
    echo "policy_replay_contract=1_prompt_microbatch_x_4_gradient_accumulations"
    echo "pipo_contract=history_window_8_negative_scale_0.1"
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
    > "${LOG_ROOT}/training.log" 2>&1

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
if any(max(record["trajectory"]["num_atoms_per_structure"]) > 60 for record in records):
    raise RuntimeError(f"Prompt atom limit was violated for {run_name}")
if any(
    epoch["microbatches"] != 4
    for record in records
    for epoch in record["policy_epochs"]
):
    raise RuntimeError(f"Prompt microbatching was not applied for {run_name}")
if not any(record["pipo_feedback"]["retrospective_update"] for record in records[1:]):
    raise RuntimeError(f"No PIPO retrospective update was applied for {run_name}")
final_path = results_root / "final_paired_evaluation.json"
if not final_path.is_file():
    raise RuntimeError(f"Missing final paired evaluation for {run_name}")
summary = {
    "run_name": run_name,
    "total_updates": len(records),
    "max_prompt_atoms": max(
        max(record["trajectory"]["num_atoms_per_structure"])
        for record in records
    ),
    "retrospective_updates": sum(
        bool(record["pipo_feedback"]["retrospective_update"])
        for record in records
    ),
    "final_paired_decision": json.loads(final_path.read_text())["decision"],
}
(results_root / "joint_max60_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False))
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 GRPO+PIPO joint FE+BG complete: ${RUN_NAME}"
