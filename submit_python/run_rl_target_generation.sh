#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the launcher}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set by the launcher}"
LOG_ROOT="${LOG_ROOT:?LOG_ROOT must be set by the launcher}"
FE_RL_MODEL_PATH="${FE_RL_MODEL_PATH:?FE_RL_MODEL_PATH must be set}"
BG_RL_MODEL_PATH="${BG_RL_MODEL_PATH:?BG_RL_MODEL_PATH must be set}"
FE_TARGET_VALUE="${FE_TARGET_VALUE:?FE_TARGET_VALUE must be set}"
BG_TARGET_VALUE="${BG_TARGET_VALUE:?BG_TARGET_VALUE must be set}"
FE_TARGET_LABEL="${FE_TARGET_LABEL:?FE_TARGET_LABEL must be set}"
BG_TARGET_LABEL="${BG_TARGET_LABEL:?BG_TARGET_LABEL must be set}"
FE_RUN="${FE_RUN:?FE_RUN must be set}"
BG_RUN="${BG_RUN:?BG_RUN must be set}"
EH_RUN="${EH_RUN:?EH_RUN must be set}"
GT_FILE="${GT_FILE:?GT_FILE must be set}"

MANIFEST="${OUTPUT_ROOT}/run_manifest.txt"
TOTAL_SAMPLES=4096
BATCH_SIZE=64
NUM_BATCHES=64
SEED=42

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

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

if [[ -s "${MANIFEST}" ]]; then
    {
        echo
        echo "resume_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "status=running"
    } >> "${MANIFEST}"
else
    {
        echo "run_id=${RUN_ID:-unknown}"
        echo "status=running"
        echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "fe_source_rl_model=${FE_RL_MODEL_PATH}"
        echo "bg_source_rl_model=${BG_RL_MODEL_PATH}"
        echo "fe_target=formation_energy_per_atom=${FE_TARGET_VALUE}"
        echo "bg_target=band_gap=${BG_TARGET_VALUE}"
        echo "seed=${SEED}"
        echo "samples_per_mode=${TOTAL_SAMPLES}"
        echo "policy_order=fe,bg"
        echo "mode_order_per_policy=abinitio_empirical,template"
    } > "${MANIFEST}"
fi

generation_complete() {
    local model_dir="$1"
    local label="$2"
    local found=""
    if [[ -d "${model_dir}/generated_structures" ]]; then
        found=$(find "${model_dir}/generated_structures" -type f \
            -name "eval_gen_${label}.pt" -size +0c -print -quit 2>/dev/null || true)
    fi
    [[ -n "${found}" ]]
}

structural_metrics_complete() {
    local model_dir="$1"
    local label="$2"
    [[ -s "${model_dir}/evaluations/structural_metrics/eval_metrics_gen_${label}.json" ]]
}

property_metrics_complete() {
    local model_dir="$1"
    local label="$2"
    [[ -s "${model_dir}/evaluations/property_metrics/eval_property_metrics_gen_${label}_predictor_seed42.json" ]] && \
        [[ -s "${model_dir}/evaluations/property_predictions/eval_properties_gen_${label}_predictor_seed42.csv" ]]
}

run_policy() {
    local task_name=$1
    local rl_model_path=$2
    local target_name=$3
    local target_value=$4
    local target_label=$5
    local policy_name
    policy_name="$(basename "$(dirname "${rl_model_path}")")"
    local policy_root="${OUTPUT_ROOT}/${policy_name}"
    local model_dir="${policy_root}/model"
    local policy_manifest="${policy_root}/policy_manifest.txt"
    local target_assignment="${target_name}=${target_value}"
    local abinitio_label="abinitio_empirical_rl_${target_label}_n4096_seed42"
    local template_label="template_rl_${target_label}_n4096_seed42"
    local source_checkpoint

    mkdir -p "${model_dir}"
    source_checkpoint=$(RL_SOURCE_PATH="${rl_model_path}" python -c \
        "import os; from cgdit.prop_models.diffusion_backbone import resolve_checkpoint; print(resolve_checkpoint(os.environ['RL_SOURCE_PATH']))")
    cp -f "${rl_model_path}/hparams.yaml" "${model_dir}/hparams.yaml"
    cp -f "${source_checkpoint}" "${model_dir}/$(basename "${source_checkpoint}")"

    if [[ -s "${policy_manifest}" ]]; then
        {
            echo
            echo "resume_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
            echo "status=running"
        } >> "${policy_manifest}"
    else
        {
            echo "policy=${policy_name}"
            echo "task=${task_name}"
            echo "status=running"
            echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
            echo "source_rl_model=${rl_model_path}"
            echo "source_checkpoint=${source_checkpoint}"
            echo "frozen_model_copy=${model_dir}"
            echo "target=${target_assignment}"
            sha256sum "${source_checkpoint}" "${rl_model_path}/hparams.yaml"
        } > "${policy_manifest}"
    fi

    if generation_complete "${model_dir}" "${abinitio_label}"; then
        echo "[SKIP] ${policy_name} ab initio generation already complete"
    else
        echo "[RUN] ${policy_name} ab initio generation at $(date '+%Y-%m-%d %H:%M:%S %z')"
        python -m scripts.cli.generation.generate \
            --model_path "${model_dir}" \
            --batch_size "${BATCH_SIZE}" \
            --num_batches_to_samples "${NUM_BATCHES}" \
            --seed "${SEED}" \
            --label "${abinitio_label}" \
            --evaluation_target "${target_assignment}" \
            --guidance_scale 1.0 \
            --ab_initio \
            --use_empirical_prior
    fi

    if structural_metrics_complete "${model_dir}" "${abinitio_label}"; then
        echo "[SKIP] ${policy_name} ab initio structural metrics already complete"
    else
        echo "[RUN] ${policy_name} ab initio structural metrics"
        python -m scripts.cli.evaluation.evaluate_metrics \
            --root_path "${model_dir}" \
            --tasks gen \
            --label "${abinitio_label}" \
            --gt_file "${GT_FILE}" \
            --num_workers 1 \
            --calc_prop false
    fi

    if generation_complete "${model_dir}" "${template_label}"; then
        echo "[SKIP] ${policy_name} template generation already complete"
    else
        echo "[RUN] ${policy_name} template generation at $(date '+%Y-%m-%d %H:%M:%S %z')"
        python -m scripts.cli.generation.generate \
            --model_path "${model_dir}" \
            --batch_size "${BATCH_SIZE}" \
            --num_batches_to_samples "${NUM_BATCHES}" \
            --seed "${SEED}" \
            --label "${template_label}" \
            --evaluation_target "${target_assignment}" \
            --guidance_scale 1.0
    fi

    if structural_metrics_complete "${model_dir}" "${template_label}"; then
        echo "[SKIP] ${policy_name} template structural metrics already complete"
    else
        echo "[RUN] ${policy_name} template structural metrics"
        python -m scripts.cli.evaluation.evaluate_metrics \
            --root_path "${model_dir}" \
            --tasks gen \
            --label "${template_label}" \
            --gt_file "${GT_FILE}" \
            --num_workers 1 \
            --calc_prop false
    fi

    if property_metrics_complete "${model_dir}" "${abinitio_label}" && \
       property_metrics_complete "${model_dir}" "${template_label}"; then
        echo "[SKIP] ${policy_name} property evaluation already complete"
    else
        echo "[RUN] ${policy_name} Seed42 property evaluation"
        python -m scripts.cli.evaluation.evaluate_generated_properties \
            --root_path "${model_dir}" \
            --fe_run "${FE_RUN}" \
            --bg_run "${BG_RUN}" \
            --eh_run "${EH_RUN}" \
            --predictor_label seed42 \
            --output_dir "${policy_root}/property_summary_seed42" \
            --batch_size 64 \
            --num_workers 4 \
            --device cuda
    fi

    for label in "${abinitio_label}" "${template_label}"; do
        test "$(find "${model_dir}/generated_structures" -type f -name "eval_gen_${label}.pt" -size +0c | wc -l)" -eq 1
        test -s "${model_dir}/evaluations/structural_metrics/eval_metrics_gen_${label}.json"
        test -s "${model_dir}/evaluations/property_metrics/eval_property_metrics_gen_${label}_predictor_seed42.json"
        test -s "${model_dir}/evaluations/property_predictions/eval_properties_gen_${label}_predictor_seed42.csv"
    done

    {
        echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "status=complete"
    } >> "${policy_manifest}"
    echo "[OK] ${policy_name} generation and evaluation completed"
}

run_policy fe "${FE_RL_MODEL_PATH}" formation_energy_per_atom \
    "${FE_TARGET_VALUE}" "${FE_TARGET_LABEL}"
run_policy bg "${BG_RL_MODEL_PATH}" band_gap \
    "${BG_TARGET_VALUE}" "${BG_TARGET_LABEL}"

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "Both FE and BG RL generation and evaluation tasks completed successfully."
echo "Results: ${OUTPUT_ROOT}"
