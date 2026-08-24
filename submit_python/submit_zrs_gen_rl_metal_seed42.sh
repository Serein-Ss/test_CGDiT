#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_ROOT}"
mkdir -p logs/rl output/rl_diagnostics

job_id="$(sbatch --parsable submit_python/zrs_gen_rl_metal_seed42.slurm)"
manifest="output/rl_diagnostics/grpo_metal_seed42_latest.txt"
{
    echo "submitted_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "job_id=${job_id}"
    echo "scope=seed42_grpo_pipo_metallicity_proxy"
    echo "initial_policy=mp20_base"
    echo "metallicity_proxy=predicted_band_gap_le_0.10_eV"
    echo "train_root=output/rl_finetune/seed42_grpo_pipo_metal/${job_id}"
} > "${manifest}"

echo "Submitted zrs-gen-rl metallicity pilot: ${job_id}"
echo "Results: output/rl_finetune/seed42_grpo_pipo_metal/${job_id}"
echo "Log: logs/rl/zrs-gen-rl-metal-${job_id}.out"
