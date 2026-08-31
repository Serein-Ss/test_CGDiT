#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
RUN_ROOT="$(rl_run_root "${PROJECT_ROOT}" test/reinforcement_learning quick8h-seed42 "${RUN_ID}")"
cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}" logs/structure_generation/reinforcement_learning/slurm logs/metric_evaluation/reinforcement_learning/slurm logs/tests/reinforcement_learning/slurm

train_job="$(sbatch --parsable --export="ALL,RUN_ID=${RUN_ID}" submit_python/zrs_gen_rl_quick8h_train_array.slurm)"
generation_job="$(sbatch --parsable --dependency="afterok:${train_job}" \
    --export="ALL,RUN_ID=${RUN_ID},TRAIN_RUN_ID=${train_job}" \
    submit_python/zrs_gen_rl_quick8h_generate_array.slurm)"
analysis_job="$(sbatch --parsable --dependency="afterok:${generation_job}" \
    --export="ALL,RUN_ID=${RUN_ID},TRAIN_RUN_ID=${train_job}" \
    submit_python/zrs_gen_rl_quick8h_analyze.slurm)"

manifest="${RUN_ROOT}/submission_manifest.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "scope=quick8h_seed42_effect_check"
    echo "training_array_job=${train_job}"
    echo "generation_array_job=${generation_job}"
    echo "analysis_job=${analysis_job}"
    echo "training_contract=3_updates_x_2_prompts_x_16_trajectories"
    echo "validation_contract=8_prompts_x_1_trajectory"
    echo "test_generation_contract=256_template_per_model"
    echo "analysis_generation_contract=256_abinitio_per_model"
    echo "train_root=${RUN_ROOT}"
    echo "assets=assets/crystalpirl_quick8h_seed42"
} > "${manifest}"

echo "Submitted quick8h seed=42 CrystalPIRL effect-check pipeline."
echo "Training array: ${train_job}"
echo "Generation/evaluation array: ${generation_job}"
echo "Fig.1-Fig.4 analysis: ${analysis_job}"
echo "Status: bash submit_python/check_zrs_gen_rl_quick8h.sh ${manifest}"
