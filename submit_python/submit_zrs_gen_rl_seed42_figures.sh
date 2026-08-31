#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
TRAIN_ROOT="$(rl_run_root "${PROJECT_ROOT}" reinforcement_learning seed42-300step-fig1-fig4 "${RUN_ID}")"
TEST_ROOT="$(rl_run_root "${PROJECT_ROOT}" test/reinforcement_learning seed42-preflight "${RUN_ID}")"
cd "${PROJECT_ROOT}"
mkdir -p "${TRAIN_ROOT}" "${TEST_ROOT}" logs/reinforcement_learning/slurm logs/structure_generation/reinforcement_learning/slurm logs/metric_evaluation/reinforcement_learning/slurm logs/tests/reinforcement_learning/slurm

preflight_job="$(sbatch --parsable --export="ALL,RUN_ID=${RUN_ID}" submit_python/zrs_gen_rl_seed42_preflight.slurm)"
train_job="$(sbatch --parsable --dependency="afterok:${preflight_job}" --export="ALL,RUN_ID=${RUN_ID}" submit_python/zrs_gen_rl_seed42_train_array.slurm)"
baseline_job="$(sbatch --parsable --dependency="afterok:${preflight_job}" --export="ALL,RUN_ID=${RUN_ID}" submit_python/zrs_gen_rl_seed42_baseline_eval.slurm)"
generation_job="$(sbatch --parsable --dependency="afterok:${train_job}" --export="ALL,RUN_ID=${RUN_ID},TRAIN_RUN_ID=${train_job}" submit_python/zrs_gen_rl_seed42_generate_array.slurm)"
analysis_job="$(sbatch --parsable --dependency="afterok:${baseline_job}:${generation_job}" --export="ALL,RUN_ID=${RUN_ID},TRAIN_RUN_ID=${train_job}" submit_python/zrs_gen_rl_seed42_analyze.slurm)"

manifest="${TRAIN_ROOT}/submission_manifest.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "scope=seed42_300step_full_denoising"
    echo "training_contract=300_updates_x_5_unique_prompts_x_16_trajectories"
    echo "training_trajectory_count_per_model=24000"
    echo "independent_validation_contract=5_fixed_prompts_x_1_trajectory"
    echo "formal_generation_contract=4096_abinitio_per_model"
    echo "preflight_job=${preflight_job}"
    echo "training_array_job=${train_job}"
    echo "baseline_eval_job=${baseline_job}"
    echo "generation_array_job=${generation_job}"
    echo "analysis_job=${analysis_job}"
    echo "train_root=${TRAIN_ROOT}"
} > "${manifest}"

echo "Submitted seed=42 CrystalPIRL pipeline."
echo "Tests and 5x16 GPU capacity probe: ${preflight_job}"
echo "Training array (12 arms, 300 updates, max 4 concurrent): ${train_job}"
echo "Base/CFG independent evaluation: ${baseline_job}"
echo "Generation/evaluation array: ${generation_job}"
echo "Source Data, conclusion audit and PNG figures: ${analysis_job}"
echo "Manifest: ${manifest}"
echo "One-time status: bash submit_python/check_zrs_gen_rl_seed42_figures.sh ${manifest}"
