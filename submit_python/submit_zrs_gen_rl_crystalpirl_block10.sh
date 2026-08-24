#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "${PROJECT_ROOT}"
mkdir -p logs/rl output/rl_diagnostics

smoke_job="$(${SBATCH:-sbatch} --parsable submit_python/zrs_gen_rl_crystalpirl_block10_smoke.slurm)"
training_job="$(${SBATCH:-sbatch} --parsable \
    --dependency="afterok:${smoke_job}" \
    submit_python/zrs_gen_rl_crystalpirl_block10_200step_array.slurm)"

manifest="output/rl_diagnostics/seed42_crystalpirl_block10_submission.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "scope=seed42_grpo_crystalpirl_block10_fe_bg_joint"
    echo "method_label=GRPO+CrystalPIRL"
    echo "smoke_job=${smoke_job}"
    echo "training_array_job=${training_job}"
    echo "training_dependency=afterok:${smoke_job}"
    echo "array_order=FE_then_BG_then_FE+BG"
    echo "single_gpu_concurrency=1"
    echo "training_contract=3_tasks_x_200_updates_x_4_prompts_x_16_trajectories"
    echo "verification_contract=every_10_updates_x_32_fixed_x_32_holdout_prompts"
    echo "seed=42"
    echo "max_prompt_atoms=52"
} > "${manifest}"

echo "Submitted blockwise CrystalPIRL seed=42 task chain."
echo "GPU smoke: ${smoke_job}"
echo "Formal FE/BG/FE+BG array: ${training_job}"
echo "Manifest: ${PROJECT_ROOT}/${manifest}"
