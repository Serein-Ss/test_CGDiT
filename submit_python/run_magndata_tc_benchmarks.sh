#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the launcher}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set by the launcher}"
LOG_ROOT="${LOG_ROOT:?LOG_ROOT must be set by the launcher}"
BASE_PRETRAINED_DIFFUSION_MODEL="${BASE_PRETRAINED_DIFFUSION_MODEL:?BASE_PRETRAINED_DIFFUSION_MODEL must be set}"
JOINT_PRETRAINED_DIFFUSION_MODEL="${JOINT_PRETRAINED_DIFFUSION_MODEL:?JOINT_PRETRAINED_DIFFUSION_MODEL must be set}"
REUSE_BENCHMARK_ROOT="${REUSE_BENCHMARK_ROOT:-}"
SEEDS_TEXT="${SEEDS:-42 123 3407}"
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
    echo "comparison=m3gnet_scratch,diffusion_scratch,diffusion_base_pretrained,diffusion_joint_pretrained"
    echo "seeds=${SEEDS_TEXT}"
    echo "split_manifest=${PROJECT_ROOT}/data/magndata/split_manifest.json"
    echo "base_pretrained_diffusion_model=${BASE_PRETRAINED_DIFFUSION_MODEL}"
    echo "joint_pretrained_diffusion_model=${JOINT_PRETRAINED_DIFFUSION_MODEL}"
    echo "reused_benchmark_root=${REUSE_BENCHMARK_ROOT:-none}"
    echo "target=tc"
    echo "target_unit=K"
    echo "primary_metric=test_mae_k"
} > "${MANIFEST}"

reuse_arm() {
    local target_method="$1"
    local source_method="$2"
    local seed="$3"
    local source_dir="${REUSE_BENCHMARK_ROOT}/${source_method}/seed_${seed}"
    local target_dir="${OUTPUT_ROOT}/${target_method}/seed_${seed}"

    if [[ -z "${REUSE_BENCHMARK_ROOT}" || -d "${target_dir}" ]]; then
        return
    fi
    test -s "${source_dir}/test_preds.npy"
    test -s "${source_dir}/test_targets.npy"
    compgen -G "${source_dir}/*.ckpt" >/dev/null
    mkdir -p "$(dirname "${target_dir}")"
    cp -a "${source_dir}" "${target_dir}"
    {
        echo "source_run_dir=${source_dir}"
        echo "reused_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    } > "${target_dir}/reused_from.txt"
    echo "[REUSE] ${source_method} seed=${seed} -> ${target_method}"
}

run_arm() {
    local method="$1"
    local seed="$2"
    local run_dir="${OUTPUT_ROOT}/${method}/seed_${seed}"
    local run_log="${LOG_ROOT}/${method}_seed_${seed}.log"
    mkdir -p "${run_dir}"

    if [[ -s "${run_dir}/test_preds.npy" && -s "${run_dir}/test_targets.npy" ]] && \
       compgen -G "${run_dir}/*.ckpt" >/dev/null; then
        echo "[SKIP] ${method} seed=${seed}: complete outputs already exist"
        return
    fi

    echo "[RUN] ${method} seed=${seed} at $(date '+%Y-%m-%d %H:%M:%S %z')"
    common_args=(
        data=magndata
        optim=magndata_tc
        train=magndata_tc
        "train.random_seed=${seed}"
        "expname=magndata_tc_${method}_seed${seed}"
        logging.wandb.mode=offline
        logging.val_check_interval=1
        "hydra.run.dir=${run_dir}"
    )

    if [[ "${method}" == "m3gnet_scratch" ]]; then
        python cgdit/run.py model=m3gnet "${common_args[@]}" > "${run_log}" 2>&1
    elif [[ "${method}" == "diffusion_scratch" ]]; then
        python cgdit/run.py model=prop_models/tc \
            model.pretrained_model_path=null \
            "${common_args[@]}" > "${run_log}" 2>&1
    elif [[ "${method}" == "diffusion_base_pretrained" ]]; then
        python cgdit/run.py model=prop_models/tc \
            "model.pretrained_model_path=${BASE_PRETRAINED_DIFFUSION_MODEL}" \
            "${common_args[@]}" > "${run_log}" 2>&1
    elif [[ "${method}" == "diffusion_joint_pretrained" ]]; then
        python cgdit/run.py model=prop_models/tc \
            "model.pretrained_model_path=${JOINT_PRETRAINED_DIFFUSION_MODEL}" \
            "${common_args[@]}" > "${run_log}" 2>&1
    else
        echo "Unknown benchmark method: ${method}" >&2
        return 2
    fi

    test -s "${run_dir}/test_preds.npy"
    test -s "${run_dir}/test_targets.npy"
    test -s "${run_dir}/hparams.yaml"
    compgen -G "${run_dir}/*.ckpt" >/dev/null
    if [[ "${method}" == diffusion_* ]]; then
        test -s "${run_dir}/pretrained_load_report.json"
    fi
    echo "[OK] ${method} seed=${seed} at $(date '+%Y-%m-%d %H:%M:%S %z')"
}

for seed in "${SEED_LIST[@]}"; do
    reuse_arm m3gnet_scratch m3gnet_scratch "${seed}"
    reuse_arm diffusion_scratch diffusion_scratch "${seed}"
    reuse_arm diffusion_joint_pretrained diffusion_pretrained "${seed}"

    run_arm m3gnet_scratch "${seed}"
    run_arm diffusion_scratch "${seed}"
    run_arm diffusion_base_pretrained "${seed}"
    run_arm diffusion_joint_pretrained "${seed}"
done

python -m scripts.cli.evaluation.summarize_magndata_tc \
    --benchmark-root "${OUTPUT_ROOT}" \
    --seeds "${SEED_LIST[@]}" \
    > "${LOG_ROOT}/benchmark_summary.log" 2>&1

test -s "${OUTPUT_ROOT}/per_run_metrics.csv"
test -s "${OUTPUT_ROOT}/aggregate_metrics.csv"
test -s "${OUTPUT_ROOT}/paired_differences.csv"
test -s "${OUTPUT_ROOT}/benchmark_summary.json"

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "All Magndata-Tc benchmark arms completed successfully."
echo "Metrics: ${OUTPUT_ROOT}/aggregate_metrics.csv"
