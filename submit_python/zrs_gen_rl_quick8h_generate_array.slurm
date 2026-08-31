#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%8
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/structure_generation/reinforcement_learning/slurm/zrs-gen-rl-quick8h-generate-%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/structure_generation/reinforcement_learning/slurm/zrs-gen-rl-quick8h-generate-%A_%a.err

set -Eeuo pipefail
: "${TRAIN_RUN_ID:?TRAIN_RUN_ID must identify the quick8h training array}"

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
: "${RUN_ID:?RUN_ID must be exported by the launcher}"
TRAIN_ROOT="$(rl_run_root "${PROJECT_ROOT}" test/reinforcement_learning quick8h-seed42 "${RUN_ID}")"
AUDIT_ROOT="${TRAIN_ROOT}/diagnostics/generation/task_${SLURM_ARRAY_TASK_ID}"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
FE_EVALUATOR="${PROJECT_ROOT}/output/singlerun/2026-08-20/20-24-50-mp20_predictor_fe_seed123"
BG_EVALUATOR="${PROJECT_ROOT}/output/singlerun/2026-08-21/08-37-12-mp20_predictor_bg_seed123"
FE_REWARD="${PROJECT_ROOT}/output/singlerun/2026-08-16/01-27-48-mp20_predictor_fe"
BG_REWARD="${PROJECT_ROOT}/output/singlerun/2026-08-16/15-18-32-mp20_predictor_bg"
GT_FILE="${PROJECT_ROOT}/data/mp_20/test.csv"
TOTAL_SAMPLES=256
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

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
TEMPLATE_LABEL="template_uncond_policy_${TARGET_LABEL}_n${TOTAL_SAMPLES}_seed42"
ABINITIO_LABEL="abinitio_empirical_uncond_policy_${TARGET_LABEL}_n${TOTAL_SAMPLES}_seed42"

cd "${PROJECT_ROOT}"
mkdir -p "${AUDIT_ROOT}" logs/structure_generation/reinforcement_learning/slurm
test -s "${MODEL_DIR}/hparams.yaml"

record_failure() {
    local status=$?
    printf 'end_time=%s\nstatus=failed\nexit_code=%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S %z')" "${status}" >> "${MANIFEST}"
    exit "${status}"
}
trap record_failure ERR

{
    echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "training_array_job_id=${TRAIN_RUN_ID}"
    echo "status=running"
    echo "scope=quick8h_seed42_effect_check"
    echo "run_name=${RUN_NAME}"
    echo "property=${PROPERTY}"
    echo "samples_per_generation_mode=${TOTAL_SAMPLES}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
} > "${MANIFEST}"

for mode in template abinitio; do
    if [[ "${mode}" == "template" ]]; then
        label="${TEMPLATE_LABEL}"
        extra_args=()
    else
        label="${ABINITIO_LABEL}"
        extra_args=(--ab_initio --use_empirical_prior)
    fi
    srun "${PYTHON}" -m scripts.cli.generation.generate \
        --model_path "${MODEL_DIR}" \
        --batch_size 64 \
        --num_batches_to_samples 4 \
        --seed 42 \
        --label "${label}" \
        --evaluation_target "${TARGET}" \
        --guidance_scale 1.0 \
        "${extra_args[@]}"
    srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_metrics \
        --root_path "${MODEL_DIR}" \
        --tasks gen \
        --label "${label}" \
        --gt_file "${GT_FILE}" \
        --num_workers 1 \
        --seed 42 \
        --calc_prop false
done

for predictor_label in seed123 seed42_reward; do
    if [[ "${predictor_label}" == "seed123" ]]; then
        fe_run="${FE_EVALUATOR}"
        bg_run="${BG_EVALUATOR}"
    else
        fe_run="${FE_REWARD}"
        bg_run="${BG_REWARD}"
    fi
    srun "${PYTHON}" -m scripts.cli.evaluation.evaluate_generated_properties \
        --root_path "${MODEL_DIR}" \
        --fe_run "${fe_run}" \
        --bg_run "${bg_run}" \
        --predictor_label "${predictor_label}" \
        --output_dir "${MODEL_DIR}/evaluations/property_metrics/summary_${predictor_label}" \
        --batch_size 64 \
        --num_workers 1 \
        --device cuda \
        --fe_tolerance 0.30 \
        --bg_tolerance 0.45
done

"${PYTHON}" - "${MODEL_DIR}" "${TEMPLATE_LABEL}" "${ABINITIO_LABEL}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
for label in sys.argv[2:]:
    generation = list(model_dir.rglob(f"eval_gen_{label}.pt"))
    if len(generation) != 1:
        raise RuntimeError(f"Missing generation payload for {label}")
    paths = [
        model_dir / "evaluations" / "structural_metrics" / f"eval_metrics_gen_{label}.json",
        model_dir / "evaluations" / "property_metrics" / f"eval_property_metrics_gen_{label}_predictor_seed123.json",
        model_dir / "evaluations" / "property_predictions" / f"eval_properties_gen_{label}_predictor_seed123.csv",
        model_dir / "evaluations" / "property_predictions" / f"eval_properties_gen_{label}_predictor_seed42_reward.csv",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise RuntimeError(f"Incomplete evaluation artifacts for {label}")
    if json.loads(paths[1].read_text())["n_total"] != 256:
        raise RuntimeError(f"Unexpected sample count for {label}")
PY_VERIFY

trap - ERR
printf 'end_time=%s\nstatus=complete\n' \
    "$(date '+%Y-%m-%d %H:%M:%S %z')" >> "${MANIFEST}"
echo "quick8h generation complete: ${RUN_NAME}"
