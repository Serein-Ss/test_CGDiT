#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
TRAIN_ROOT="$(rl_run_root "${PROJECT_ROOT}" reinforcement_learning crystalpirl-block10-200step "${RUN_ID}")"
TEST_ROOT="$(rl_run_root "${PROJECT_ROOT}" test/reinforcement_learning crystalpirl-block10-smoke "${RUN_ID}")"

cd "${PROJECT_ROOT}"
mkdir -p "${TRAIN_ROOT}" "${TEST_ROOT}" logs/reinforcement_learning/slurm logs/tests/reinforcement_learning/slurm

smoke_job="$(${SBATCH:-sbatch} --parsable --export="ALL,RUN_ID=${RUN_ID}" submit_python/zrs_gen_rl_crystalpirl_block10_smoke.slurm)"
training_job="$(${SBATCH:-sbatch} --parsable \
    --dependency="afterok:${smoke_job}" \
    --export="ALL,RUN_ID=${RUN_ID}" \
    submit_python/zrs_gen_rl_crystalpirl_block10_200step_array.slurm)"

manifest="${TRAIN_ROOT}/submission_manifest.txt"
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
echo "Manifest: ${manifest}"
