#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=56G
#SBATCH --time=48:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-seed42-generate-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-seed42-generate-%A_%a.err

set -Eeuo pipefail

: "${TRAIN_RUN_ID:?TRAIN_RUN_ID must identify the completed training array}"

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_ROOT="${PROJECT_ROOT}/output/rl_finetune/seed42_300step_fig1_fig4/${TRAIN_RUN_ID}"
AUDIT_ROOT="${PROJECT_ROOT}/output/rl_diagnostics/${SLURM_ARRAY_JOB_ID}-seed42-generate/task_${SLURM_ARRAY_TASK_ID}"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
FE_EVALUATOR="${PROJECT_ROOT}/output/singlerun/2026-08-20/20-24-50-mp20_predictor_fe_seed123"
BG_EVALUATOR="${PROJECT_ROOT}/output/singlerun/2026-08-21/08-37-12-mp20_predictor_bg_seed123"
FE_REWARD="${PROJECT_ROOT}/output/singlerun/2026-08-16/01-27-48-mp20_predictor_fe"
BG_REWARD="${PROJECT_ROOT}/output/singlerun/2026-08-16/15-18-32-mp20_predictor_bg"
GT_FILE="${PROJECT_ROOT}/data/mp_20/test.csv"
TOTAL_SAMPLES=4096
BATCH_SIZE=64
NUM_BATCHES=64
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN_NAMES=(
    ppo_fe_seed42 ppo_fe_seed42_pipo ppo_fe_seed42_pirl
    grpo_fe_seed42 grpo_fe_seed42_pipo grpo_fe_seed42_pirl
    ppo_bg_seed42 ppo_bg_seed42_pipo ppo_bg_seed42_pirl
    grpo_bg_seed42 grpo_bg_seed42_pipo grpo_bg_seed42_pirl
)
PROPERTIES=(fe fe fe fe fe fe bg bg bg bg bg bg)

INDEX="${SLURM_ARRAY_TASK_ID}"
RUN_NAME="${RUN_NAMES[${INDEX}]}"
PROPERTY="${PROPERTIES[${INDEX}]}"
MODEL_DIR="${TRAIN_ROOT}/${RUN_NAME}/model"

if [[ "${PROPERTY}" == "fe" ]]; then
    TARGET="formation_energy_per_atom=-1.5"
    TARGET_LABEL="fe_m1p5"
else
    TARGET="band_gap=2.0"
    TARGET_LABEL="bg_2"
fi

ABINITIO_LABEL="abinitio_empirical_uncond_policy_${TARGET_LABEL}_n${TOTAL_SAMPLES}_seed42"

cd "${PROJECT_ROOT}"
mkdir -p "${AUDIT_ROOT}" logs/rl
test -x "${PYTHON}"
test -s "${MODEL_DIR}/hparams.yaml"
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
    echo "training_array_job_id=${TRAIN_RUN_ID}"
    echo "job_name=${SLURM_JOB_NAME}"
    echo "status=running"
    echo "seed=42"
    echo "scope=seed42_300step_full_denoising"
    echo "run_name=${RUN_NAME}"
    echo "property=${PROPERTY}"
    echo "target=${TARGET}"
    echo "formal_generation_mode=abinitio_empirical"
    echo "formal_samples=${TOTAL_SAMPLES}"
    echo "template_generation=test_only_not_formal"
    echo "model_dir=${MODEL_DIR}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    sha256sum scripts/cli/generation/generate.py cgdit/generation/general.py cgdit/evaluation/generated_properties.py
} > "${MANIFEST}"

srun "${PYTHON}" -m scripts.cli.generation.generate \
    --model_path "${MODEL_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_batches_to_samples "${NUM_BATCHES}" \
    --seed 42 \
    --label "${ABINITIO_LABEL}" \
    --evaluation_target "${TARGET}" \
    --guidance_scale 1.0 \
    --ab_initio \
    --use_empirical_prior

srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_metrics \
    --root_path "${MODEL_DIR}" \
    --tasks gen \
    --label "${ABINITIO_LABEL}" \
    --gt_file "${GT_FILE}" \
    --num_workers 1 \
    --seed 42 \
    --calc_prop false

srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_generated_properties \
    --root_path "${MODEL_DIR}" \
    --fe_run "${FE_EVALUATOR}" \
    --bg_run "${BG_EVALUATOR}" \
    --predictor_label seed123 \
    --output_dir "${MODEL_DIR}/evaluations/property_metrics/summary_seed123" \
    --batch_size 64 \
    --num_workers 1 \
    --device cuda \
    --fe_tolerance 0.30 \
    --bg_tolerance 0.45

srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_generated_properties \
    --root_path "${MODEL_DIR}" \
    --fe_run "${FE_REWARD}" \
    --bg_run "${BG_REWARD}" \
    --predictor_label seed42_reward \
    --output_dir "${MODEL_DIR}/evaluations/property_metrics/summary_seed42_reward" \
    --batch_size 64 \
    --num_workers 1 \
    --device cuda \
    --fe_tolerance 0.30 \
    --bg_tolerance 0.45

"${PYTHON}" - "${MODEL_DIR}" "${ABINITIO_LABEL}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

model_dir, label = Path(sys.argv[1]), sys.argv[2]
files = list(model_dir.rglob(f"eval_gen_{label}.pt"))
if len(files) != 1:
    raise RuntimeError(f"Expected one generated file for {label}, got {files}")
metrics = model_dir / "evaluations" / "structural_metrics" / f"eval_metrics_gen_{label}.json"
properties = model_dir / "evaluations" / "property_metrics" / f"eval_property_metrics_gen_{label}_predictor_seed123.json"
predictions = model_dir / "evaluations" / "property_predictions" / f"eval_properties_gen_{label}_predictor_seed123.csv"
reward_predictions = model_dir / "evaluations" / "property_predictions" / f"eval_properties_gen_{label}_predictor_seed42_reward.csv"
for path in (metrics, properties, predictions, reward_predictions):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing evaluation artifact: {path}")
payload = json.loads(properties.read_text())
if payload["n_total"] != 4096:
    raise RuntimeError(f"Unexpected sample count in {properties}")
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 generation and evaluation complete: ${RUN_NAME}"
