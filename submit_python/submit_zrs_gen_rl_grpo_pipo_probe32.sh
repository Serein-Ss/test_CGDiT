#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
TRAIN_ROOT="$(rl_run_root "${PROJECT_ROOT}" reinforcement_learning grpo-pipo-probe32 "${RUN_ID}")"
cd "${PROJECT_ROOT}"
mkdir -p "${TRAIN_ROOT}" logs/reinforcement_learning/slurm

job_id="$(sbatch --parsable --export="ALL,RUN_ID=${RUN_ID}" submit_python/zrs_gen_rl_grpo_pipo_probe32_array.slurm)"
manifest="${TRAIN_ROOT}/submission_manifest.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "job_id=${job_id}"
    echo "scope=seed42_grpo_pipo_200step_probe32"
    echo "training_contract=4_prompts_x_16_trajectories"
    echo "microbatch_contract=1_prompt_per_backward"
    echo "pipo_contract=K8_lambda0.1"
    echo "final_validation_contract=32_prompts_x_1_trajectory"
    echo "train_root=${TRAIN_ROOT}"
    echo "log_root=${PROJECT_ROOT}/logs/reinforcement_learning/slurm/${RUN_ID}-grpo-pipo-probe32"
} > "${manifest}"

echo "Submitted zrs-gen-rl GRPO+PIPO array: ${job_id}"
echo "Tasks: 0=FE, 1=BG"
echo "Status: bash submit_python/check_zrs_gen_rl_grpo_pipo_probe32.sh ${manifest}"
