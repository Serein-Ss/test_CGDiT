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
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/metric_evaluation/reinforcement_learning/slurm/zrs-gen-rl-interim-paired-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/metric_evaluation/reinforcement_learning/slurm/zrs-gen-rl-interim-paired-%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CHECKPOINT_ROOT="${PROJECT_ROOT}/output/reinforcement_learning/2026-08-23/21-47-35-grpo-pipo-probe32"
OUTPUT_ROOT="${PROJECT_ROOT}/output/reinforcement_learning/2026-08-23/21-47-35-grpo-pipo-probe32/diagnostics/interim_paired_step39"

CONFIGS=(
    conf/rl/experiments/grpo_fe_seed42_pipo_probe32.yaml
    conf/rl/experiments/grpo_bg_seed42_pipo_probe32.yaml
)
CHECKPOINTS=(
    "${CHECKPOINT_ROOT}/grpo_fe_seed42_pipo/model/epoch=39-step=39.ckpt"
    "${CHECKPOINT_ROOT}/grpo_bg_seed42_pipo/model/epoch=39-step=39.ckpt"
)
PROPERTIES=(fe bg)

INDEX="${SLURM_ARRAY_TASK_ID}"
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" logs/metric_evaluation/reinforcement_learning/slurm

srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_rl_checkpoint_paired \
    --train-config "${CONFIGS[${INDEX}]}" \
    --checkpoint "${CHECKPOINTS[${INDEX}]}" \
    --output "${OUTPUT_ROOT}/${PROPERTIES[${INDEX}]}.json"
