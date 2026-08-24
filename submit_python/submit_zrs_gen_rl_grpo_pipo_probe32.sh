#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_ROOT}"
mkdir -p logs/rl output/rl_diagnostics

job_id="$(sbatch --parsable submit_python/zrs_gen_rl_grpo_pipo_probe32_array.slurm)"
manifest="output/rl_diagnostics/grpo_pipo_probe32_latest.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "job_id=${job_id}"
    echo "scope=seed42_grpo_pipo_200step_probe32"
    echo "training_contract=4_prompts_x_16_trajectories"
    echo "microbatch_contract=1_prompt_per_backward"
    echo "pipo_contract=K8_lambda0.1"
    echo "final_validation_contract=32_prompts_x_1_trajectory"
    echo "train_root=output/rl_finetune/seed42_grpo_pipo_probe32/${job_id}"
} > "${manifest}"

echo "Submitted zrs-gen-rl GRPO+PIPO array: ${job_id}"
echo "Tasks: 0=FE, 1=BG"
echo "Status: bash submit_python/check_zrs_gen_rl_grpo_pipo_probe32.sh ${job_id}"
