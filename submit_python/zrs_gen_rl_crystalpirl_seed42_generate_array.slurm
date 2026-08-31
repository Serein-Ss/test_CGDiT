#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=56G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/structure_generation/reinforcement_learning/slurm/zrs-gen-rl-crystalpirl-seed42-generate-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/structure_generation/reinforcement_learning/slurm/zrs-gen-rl-crystalpirl-seed42-generate-%A_%a.err

set -Eeuo pipefail

: "${TRAIN_ROOT:?TRAIN_ROOT must point to the completed CrystalPIRL run}"

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
FE_EVALUATOR="${PROJECT_ROOT}/output/singlerun/2026-08-20/20-24-50-mp20_predictor_fe_seed123"
BG_EVALUATOR="${PROJECT_ROOT}/output/singlerun/2026-08-21/08-37-12-mp20_predictor_bg_seed123"
GT_FILE="${PROJECT_ROOT}/data/mp_20/test.csv"
AUDIT_ROOT="${TRAIN_ROOT}/diagnostics/generation/task_${SLURM_ARRAY_TASK_ID}"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
TOTAL_SAMPLES=4096
BATCH_SIZE=64
NUM_BATCHES=64
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN_NAMES=(
    grpo_fe_seed42_pirl
    grpo_bg_seed42_pirl
    grpo_joint_seed42_pirl
)
TASKS=(fe bg joint)
LABELS=(
    abinitio_empirical_uncond_crystalpirl_fe_m1p5_n4096_seed42
    abinitio_empirical_uncond_crystalpirl_bg_2_n4096_seed42
    abinitio_empirical_uncond_crystalpirl_fe_m1p5_bg_2_n4096_seed42
)

INDEX="${SLURM_ARRAY_TASK_ID}"
RUN_NAME="${RUN_NAMES[${INDEX}]}"
TASK="${TASKS[${INDEX}]}"
LABEL="${LABELS[${INDEX}]}"
MODEL_DIR="${TRAIN_ROOT}/${RUN_NAME}/model"
RESULTS_DIR="${TRAIN_ROOT}/${RUN_NAME}/train_results"
GENERATION_PATH="${MODEL_DIR}/generated_structures/formal/abinitio_empirical/unconditional/eval_gen_${LABEL}.pt"
STRUCTURAL_METRICS="${MODEL_DIR}/evaluations/structural_metrics/eval_metrics_gen_${LABEL}.json"
PROPERTY_METRICS="${MODEL_DIR}/evaluations/property_metrics/eval_property_metrics_gen_${LABEL}_predictor_seed123.json"
PROPERTY_PREDICTIONS="${MODEL_DIR}/evaluations/property_predictions/eval_properties_gen_${LABEL}_predictor_seed123.csv"

if [[ "${TASK}" == "fe" ]]; then
    EVALUATION_TARGETS=(--evaluation_target formation_energy_per_atom=-1.5)
elif [[ "${TASK}" == "bg" ]]; then
    EVALUATION_TARGETS=(--evaluation_target band_gap=2.0)
else
    EVALUATION_TARGETS=(
        --evaluation_target formation_energy_per_atom=-1.5
        --evaluation_target band_gap=2.0
    )
fi

cd "${PROJECT_ROOT}"
mkdir -p "${AUDIT_ROOT}" logs/structure_generation/reinforcement_learning/slurm
test -x "${PYTHON}"
test -s "${MODEL_DIR}/hparams.yaml"
test -s "${RESULTS_DIR}/crystalpirl_block10_summary.json"
test -s "${RESULTS_DIR}/final_paired_evaluation.json"
compgen -G "${MODEL_DIR}/*.ckpt" >/dev/null

record_failure() {
    local status=$?
    {
        echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "status=failed"
        echo "exit_code=${status}"
    } >> "${MANIFEST}"
    exit "${status}"
}
trap record_failure ERR

{
    echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "job_name=${SLURM_JOB_NAME}"
    echo "status=running"
    echo "seed=42"
    echo "method=GRPO+CrystalPIRL"
    echo "task=${TASK}"
    echo "formal_generation_mode=abinitio_empirical"
    echo "formal_generation_policy=unconditional_rl_policy"
    echo "formal_samples=${TOTAL_SAMPLES}"
    echo "max_atoms=52"
    echo "independent_evaluator=seed123"
    echo "training_job=${TRAIN_JOB}"
    echo "model_dir=${MODEL_DIR}"
    echo "generation_label=${LABEL}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum scripts/cli/generation/generate.py cgdit/generation/general.py cgdit/evaluation/generated_properties.py
} > "${MANIFEST}"

srun "${PYTHON}" -m scripts.cli.generation.generate \
    --model_path "${MODEL_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_batches_to_samples "${NUM_BATCHES}" \
    --seed 42 \
    --label "${LABEL}" \
    "${EVALUATION_TARGETS[@]}" \
    --guidance_scale 0.0 \
    --ab_initio \
    --use_empirical_prior \
    --max_atoms 52

srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_metrics \
    --root_path "${MODEL_DIR}" \
    --tasks gen \
    --label "${LABEL}" \
    --gt_file "${GT_FILE}" \
    --num_workers 1 \
    --seed 42 \
    --calc_prop false

srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_generated_properties \
    --root_path "${GENERATION_PATH%/*}" \
    --fe_run "${FE_EVALUATOR}" \
    --bg_run "${BG_EVALUATOR}" \
    --predictor_label seed123 \
    --output_dir "${MODEL_DIR}/evaluations/property_metrics/summary_seed123" \
    --batch_size 64 \
    --num_workers 1 \
    --device cuda \
    --fe_tolerance 0.30 \
    --bg_tolerance 0.30

"${PYTHON}" - "${GENERATION_PATH}" "${STRUCTURAL_METRICS}" "${PROPERTY_METRICS}" "${PROPERTY_PREDICTIONS}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

import pandas as pd

from cgdit.evaluation.generated_properties import load_generation_payload

generation, structural, metrics, predictions = map(Path, sys.argv[1:])
for path in (generation, structural, metrics, predictions):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing evaluation artifact: {path}")
payload = load_generation_payload(generation)
if int(payload["seed"]) != 42:
    raise RuntimeError(f"Unexpected generation seed in {generation}")
if int(payload["num_atoms"].numel()) != 4096:
    raise RuntimeError(f"Unexpected structure count in {generation}")
if int(payload["num_atoms"].max().item()) > 52:
    raise RuntimeError(f"Generation atom limit was violated in {generation}")
if payload.get("conditions"):
    raise RuntimeError("CrystalPIRL generation must not apply CFG conditions")
frame = pd.read_csv(predictions)
if len(frame) != 4096 or frame["structure_index"].nunique() != 4096:
    raise RuntimeError(f"Unexpected prediction coverage in {predictions}")
summary = json.loads(metrics.read_text())
if int(summary["n_total"]) != 4096:
    raise RuntimeError(f"Unexpected property metric count in {metrics}")
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
    echo "generation=${GENERATION_PATH}"
    echo "structural_metrics=${STRUCTURAL_METRICS}"
    echo "property_metrics=${PROPERTY_METRICS}"
    echo "property_predictions=${PROPERTY_PREDICTIONS}"
} >> "${MANIFEST}"

echo "seed=42 CrystalPIRL generation/evaluation complete: ${RUN_NAME}"
