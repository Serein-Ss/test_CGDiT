#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the launcher}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set by the launcher}"
LOG_ROOT="${LOG_ROOT:?LOG_ROOT must be set by the launcher}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:?SOURCE_DATA_ROOT must be set}"
BASE_DIFFUSION_MODEL="${BASE_DIFFUSION_MODEL:?BASE_DIFFUSION_MODEL must be set}"
GLOBAL_TC_BENCHMARK_ROOT="${GLOBAL_TC_BENCHMARK_ROOT:?GLOBAL_TC_BENCHMARK_ROOT must be set}"
SEEDS_TEXT="${SEEDS:-42 123 3407}"
TC_THRESHOLD_K="${TC_THRESHOLD_K:-300}"
MIN_GATE_RECALL="${MIN_GATE_RECALL:-0.80}"
DATA_ROOT="${OUTPUT_ROOT}/data"
MANIFEST="${OUTPUT_ROOT}/run_manifest.txt"

read -r -a SEED_LIST <<< "${SEEDS_TEXT}"
if (( ${#SEED_LIST[@]} == 0 )); then
    echo "SEEDS must contain at least one integer." >&2
    exit 2
fi
for seed in "${SEED_LIST[@]}"; do
    if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed: ${seed}" >&2
        exit 2
    fi
done

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

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
    echo "run_id=${RUN_ID:-unknown}"
    echo "status=running"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "experiment=high_low_tc_gate_with_regression_experts"
    echo "seeds=${SEEDS_TEXT}"
    echo "tc_threshold_k=${TC_THRESHOLD_K}"
    echo "min_gate_validation_recall=${MIN_GATE_RECALL}"
    echo "source_data_root=${SOURCE_DATA_ROOT}"
    echo "base_diffusion_model=${BASE_DIFFUSION_MODEL}"
    echo "global_tc_benchmark_root=${GLOBAL_TC_BENCHMARK_ROOT}"
    echo "gate_loss=class_weighted_bce"
    echo "expert_loss=huber"
    echo "expert_target_normalization=global_magndata_train_statistics"
    echo "primary_route=soft"
} > "${MANIFEST}"

python -m scripts.cli.data.prepare_magndata_tc_moe \
    --source-root "${SOURCE_DATA_ROOT}" \
    --output-root "${DATA_ROOT}" \
    --threshold-k "${TC_THRESHOLD_K}" \
    > "${LOG_ROOT}/prepare_data.log" 2>&1

read_manifest_value() {
    python -c 'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); value=data; [value := value[key] for key in sys.argv[2].split(".")]; print(value)' \
        "${DATA_ROOT}/moe_data_manifest.json" "$1"
}

POS_WEIGHT="$(read_manifest_value gate_pos_weight)"

resolve_checkpoint() {
    python -c 'from cgdit.prop_models.diffusion_backbone import resolve_checkpoint; import sys; print(resolve_checkpoint(sys.argv[1]))' "$1"
}

run_gate() {
    local seed="$1"
    local run_dir="${OUTPUT_ROOT}/seed_${seed}/gate"
    local run_log="${LOG_ROOT}/gate_seed_${seed}.log"
    mkdir -p "${run_dir}"

    if [[ -s "${run_dir}/test_probs.npy" && -s "${run_dir}/test_targets.npy" ]] && \
       compgen -G "${run_dir}/*.ckpt" >/dev/null; then
        echo "[SKIP] gate seed=${seed}: trained test outputs already exist"
    else
        echo "[RUN] gate seed=${seed} at $(date '+%Y-%m-%d %H:%M:%S %z')"
        python cgdit/run.py \
            data=magndata \
            model=property_predictors/diffusion_cspnet/tc_classifier \
            optim=magndata_tc \
            train=magndata_tc_classifier \
            "data.root_path=${DATA_ROOT}/gate" \
            data.prop=high_tc \
            'data.prop_list=[high_tc]' \
            "model.pretrained_model_path=${BASE_DIFFUSION_MODEL}" \
            "model.pos_weight=${POS_WEIGHT}" \
            "train.random_seed=${seed}" \
            "expname=magndata_tc_gate_seed${seed}" \
            logging.wandb.mode=offline \
            logging.val_check_interval=1 \
            "hydra.run.dir=${run_dir}" \
            > "${run_log}" 2>&1
        test -s "${run_dir}/test_probs.npy"
        test -s "${run_dir}/test_targets.npy"
        compgen -G "${run_dir}/*.ckpt" >/dev/null
        echo "[OK] gate seed=${seed} trained at $(date '+%Y-%m-%d %H:%M:%S %z')"
    fi

    if [[ -s "${run_dir}/val_probs.npy" && -s "${run_dir}/val_targets.npy" ]]; then
        echo "[SKIP] gate validation prediction seed=${seed}"
    else
        echo "[RUN] gate validation prediction seed=${seed}"
        python -m scripts.cli.evaluation.predict_tc_gate \
            --run-dir "${run_dir}" \
            --output-dir "${run_dir}" \
            --device cuda \
            >> "${run_log}" 2>&1
        test -s "${run_dir}/val_probs.npy"
        test -s "${run_dir}/val_targets.npy"
        echo "[OK] gate validation prediction seed=${seed}"
    fi
}

run_expert() {
    local branch="$1"
    local seed="$2"
    local run_dir="${OUTPUT_ROOT}/seed_${seed}/${branch}_expert"
    local run_log="${LOG_ROOT}/${branch}_expert_seed_${seed}.log"
    local global_run="${GLOBAL_TC_BENCHMARK_ROOT}/diffusion_base_pretrained/seed_${seed}"
    local global_checkpoint
    global_checkpoint="$(resolve_checkpoint "${global_run}")"
    mkdir -p "${run_dir}"

    if [[ -s "${run_dir}/test_preds.npy" && -s "${run_dir}/test_targets.npy" ]] && \
       compgen -G "${run_dir}/*.ckpt" >/dev/null; then
        echo "[SKIP] ${branch}_expert seed=${seed}: complete outputs already exist"
        return
    fi

    echo "[RUN] ${branch}_expert seed=${seed} at $(date '+%Y-%m-%d %H:%M:%S %z')"
    FINETUNE_CHECKPOINT="${global_checkpoint}" python cgdit/run.py \
        data=magndata \
        model=property_predictors/diffusion_cspnet/tc_regressor \
        optim=magndata_tc \
        train=magndata_tc \
        "data.root_path=${DATA_ROOT}/${branch}" \
        data.prop=tc \
        'data.prop_list=[tc]' \
        model.pretrained_model_path=null \
        model.loss_type=huber \
        model.huber_delta=1.0 \
        '+train.finetune_from_ckpt=${oc.env:FINETUNE_CHECKPOINT}' \
        +train.require_full_finetune_load=true \
        "train.random_seed=${seed}" \
        "expname=magndata_tc_${branch}_expert_seed${seed}" \
        logging.wandb.mode=offline \
        logging.val_check_interval=1 \
        "hydra.run.dir=${run_dir}" \
        > "${run_log}" 2>&1
    test -s "${run_dir}/test_preds.npy"
    test -s "${run_dir}/test_targets.npy"
    test -s "${run_dir}/pretrained_load_report.json"
    compgen -G "${run_dir}/*.ckpt" >/dev/null
    echo "[OK] ${branch}_expert seed=${seed} at $(date '+%Y-%m-%d %H:%M:%S %z')"
}

for seed in "${SEED_LIST[@]}"; do
    run_gate "${seed}"
    run_expert low "${seed}"
    run_expert high "${seed}"
done

python -m scripts.cli.evaluation.summarize_magndata_tc_moe \
    --moe-root "${OUTPUT_ROOT}" \
    --global-benchmark-root "${GLOBAL_TC_BENCHMARK_ROOT}" \
    --seeds "${SEED_LIST[@]}" \
    --tc-threshold-k "${TC_THRESHOLD_K}" \
    --min-gate-recall "${MIN_GATE_RECALL}" \
    > "${LOG_ROOT}/moe_summary.log" 2>&1

for result in \
    gate_metrics_per_seed.csv \
    gate_metrics_aggregate.csv \
    regression_metrics_per_seed.csv \
    regression_metrics_aggregate.csv \
    routing_differences.csv \
    test_predictions.csv \
    moe_summary.json; do
    test -s "${OUTPUT_ROOT}/${result}"
done

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "All Magndata-Tc gated MoE runs completed successfully."
echo "Metrics: ${OUTPUT_ROOT}/regression_metrics_aggregate.csv"
