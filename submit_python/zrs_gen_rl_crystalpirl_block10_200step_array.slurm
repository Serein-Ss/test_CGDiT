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
#SBATCH --array=0-2%1
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/reinforcement_learning/slurm/zrs-gen-rl-crystalpirl-block10-200step-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/reinforcement_learning/slurm/zrs-gen-rl-crystalpirl-block10-200step-%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="$(rl_resolve_run_id)"
TRAIN_ROOT="$(rl_run_root "${PROJECT_ROOT}" reinforcement_learning crystalpirl-block10-200step "${RUN_ID}")"
AUDIT_ROOT="${TRAIN_ROOT}/diagnostics/task_${SLURM_ARRAY_TASK_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/reinforcement_learning/slurm/${RUN_ID}-crystalpirl-block10-200step/task_${SLURM_ARRAY_TASK_ID}"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIGS=(
    conf/rl/experiments/grpo_fe_seed42_crystalpirl_block10.yaml
    conf/rl/experiments/grpo_bg_seed42_crystalpirl_block10.yaml
    conf/rl/experiments/grpo_joint_seed42_crystalpirl_block10.yaml
)
RUN_NAMES=(
    grpo_fe_seed42_pirl
    grpo_bg_seed42_pirl
    grpo_joint_seed42_pirl
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
    echo "method=GRPO+CrystalPIRL"
    echo "scope=block10_200step_full_denoising"
    echo "training_contract=200_updates_x_4_prompts_x_16_trajectories"
    echo "policy_replay_contract=1_prompt_microbatch_x_4_gradient_accumulations"
    echo "verification_contract=20_boundaries_x_32_fixed_prompts_x_32_holdout_prompts"
    echo "prompt_atom_limit=52_inclusive"
    echo "config=${CONFIG}"
    echo "run_name=${RUN_NAME}"
    echo "train_root=${TRAIN_ROOT}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum "${CONFIG}" scripts/cli/training/train_crystal_rl.py cgdit/rl/paired_probe.py
} > "${MANIFEST}"

srun "${PYTHON}" -m scripts.cli.training.train_crystal_rl \
    --train-config "${CONFIG}" \
    --output-root "${TRAIN_ROOT}" \
    > "${LOG_ROOT}/training.log" 2>&1

"${PYTHON}" - "${TRAIN_ROOT}" "${RUN_NAME}" <<'PY_VERIFY'
import json
import sys
from collections import Counter
from pathlib import Path

root, run_name = Path(sys.argv[1]), sys.argv[2]
run_root = root / run_name
results = run_root / "train_results"
records = [
    json.loads(line)
    for line in (results / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
if [record["step"] for record in records] != list(range(200)):
    raise RuntimeError(f"Expected unique steps 0..199 for {run_name}")
boundaries = [record for record in records if record["decision"]["verification_due"]]
pending = [record for record in records if not record["decision"]["verification_due"]]
if [record["step"] for record in boundaries] != list(range(9, 200, 10)):
    raise RuntimeError(f"Unexpected verification boundaries for {run_name}")
if len(pending) != 180 or any(record["decision"]["action"] != "pending" for record in pending):
    raise RuntimeError(f"Unexpected intermediate decision state for {run_name}")
if any(not record["decision"]["safety_audit"]["formal_complete"] for record in boundaries):
    raise RuntimeError(f"Incomplete boundary safety audit for {run_name}")
if any(record["sampling_contract"]["pirl_verification_interval"] != 10 for record in records):
    raise RuntimeError(f"Verification interval drift for {run_name}")
if any(record["sampling_contract"]["policy_microbatch_count"] != 4 for record in records):
    raise RuntimeError(f"Prompt microbatching was not applied for {run_name}")
if max(max(record["trajectory"]["num_atoms_per_structure"]) for record in records) > 52:
    raise RuntimeError(f"Prompt atom limit was violated for {run_name}")
if len(list((run_root / "model").glob("*.ckpt"))) != 20:
    raise RuntimeError(f"Expected 20 boundary checkpoints for {run_name}")
final_path = results / "final_paired_evaluation.json"
if not final_path.is_file():
    raise RuntimeError(f"Missing final paired evaluation for {run_name}")
actions = Counter(record["decision"]["action"] for record in boundaries)
summary = {
    "run_name": run_name,
    "seed": 42,
    "updates": len(records),
    "verification_interval": 10,
    "verification_boundaries": len(boundaries),
    "boundary_actions": dict(actions),
    "verified_updates": sum(bool(record["decision"]["verified_checkpoint_update"]) for record in boundaries),
    "max_prompt_atoms_observed": max(max(record["trajectory"]["num_atoms_per_structure"]) for record in records),
    "final_paired_evaluation": json.loads(final_path.read_text()),
}
(results / "crystalpirl_block10_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False))
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 blockwise CrystalPIRL 200-step training complete: ${RUN_NAME}"
