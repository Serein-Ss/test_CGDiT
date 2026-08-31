#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
TRAIN_ROOT="$(rl_run_root "${PROJECT_ROOT}" reinforcement_learning grpo-pipo-metal "${RUN_ID}")"
cd "${PROJECT_ROOT}"
mkdir -p "${TRAIN_ROOT}" logs/reinforcement_learning/slurm

job_id="$(sbatch --parsable --export="ALL,RUN_ID=${RUN_ID}" submit_python/zrs_gen_rl_metal_seed42.slurm)"
manifest="${TRAIN_ROOT}/submission_manifest.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "job_id=${job_id}"
    echo "scope=seed42_grpo_pipo_metallicity_proxy"
    echo "initial_policy=mp20_base"
    echo "metallicity_proxy=predicted_band_gap_le_0.10_eV"
    echo "train_root=${TRAIN_ROOT}"
} > "${manifest}"

echo "Submitted zrs-gen-rl metallicity pilot: ${job_id}"
echo "Results: ${TRAIN_ROOT}"
echo "Log: logs/reinforcement_learning/slurm/zrs-gen-rl-metal-${job_id}.out"
