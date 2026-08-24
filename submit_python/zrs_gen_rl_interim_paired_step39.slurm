#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-interim-paired-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-interim-paired-%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CHECKPOINT_ROOT="${PROJECT_ROOT}/output/rl_finetune/seed42_grpo_pipo_probe32/668001"
OUTPUT_ROOT="${PROJECT_ROOT}/output/rl_diagnostics/668001-seed42-grpo-pipo-probe32/interim_paired_step39"

CONFIGS=(
    conf/rl/grpo_fe_seed42_pipo_probe32.yaml
    conf/rl/grpo_bg_seed42_pipo_probe32.yaml
)
CHECKPOINTS=(
    "${CHECKPOINT_ROOT}/grpo_fe_seed42_pipo/model/epoch=39-step=39.ckpt"
    "${CHECKPOINT_ROOT}/grpo_bg_seed42_pipo/model/epoch=39-step=39.ckpt"
)
PROPERTIES=(fe bg)

INDEX="${SLURM_ARRAY_TASK_ID}"
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" logs/rl

srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_rl_checkpoint_paired \
    --train-config "${CONFIGS[${INDEX}]}" \
    --checkpoint "${CHECKPOINTS[${INDEX}]}" \
    --output "${OUTPUT_ROOT}/${PROPERTIES[${INDEX}]}.json"
