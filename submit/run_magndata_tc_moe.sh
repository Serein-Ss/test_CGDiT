#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORKER="submit_python/run_magndata_tc_moe.sh"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/magndata_tc_moe/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/magndata_tc_moe/${RUN_ID}}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-${PROJECT_ROOT}/data/magndata}"
SEEDS="${SEEDS:-42 123 3407}"
GPU_ID="${GPU_ID:-0}"
TC_THRESHOLD_K="${TC_THRESHOLD_K:-300}"
MIN_GATE_RECALL="${MIN_GATE_RECALL:-0.80}"
BASE_DIFFUSION_MODEL="${BASE_DIFFUSION_MODEL:-${PROJECT_ROOT}/output/singlerun/2026-06-27/00-32-50-mp20_base}"
GLOBAL_TC_BENCHMARK_ROOT="${GLOBAL_TC_BENCHMARK_ROOT:-${PROJECT_ROOT}/output/magndata_tc_benchmark/20260824-113321}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the cgdit Conda environment before launching." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
bash -n "${WORKER}"
for split in train val test; do
    test -s "${SOURCE_DATA_ROOT}/${split}.csv"
done
test -s "${BASE_DIFFUSION_MODEL}/hparams.yaml"
compgen -G "${BASE_DIFFUSION_MODEL}/*.ckpt" >/dev/null
for seed in ${SEEDS}; do
    global_run="${GLOBAL_TC_BENCHMARK_ROOT}/diffusion_base_pretrained/seed_${seed}"
    test -s "${global_run}/hparams.yaml"
    test -s "${global_run}/test_preds.npy"
    test -s "${global_run}/test_targets.npy"
    compgen -G "${global_run}/*.ckpt" >/dev/null
done

python -m pytest -q tests/test_magndata_tc_moe.py \
    > "${LOG_ROOT}/pytest_preflight.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT}" \
RUN_ID="${RUN_ID}" \
SEEDS="${SEEDS}" \
TC_THRESHOLD_K="${TC_THRESHOLD_K}" \
MIN_GATE_RECALL="${MIN_GATE_RECALL}" \
BASE_DIFFUSION_MODEL="${BASE_DIFFUSION_MODEL}" \
GLOBAL_TC_BENCHMARK_ROOT="${GLOBAL_TC_BENCHMARK_ROOT}" \
nohup bash "${WORKER}" > "${LOG_ROOT}/magndata_tc_moe.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${LOG_ROOT}/master.pid"

echo "Started Magndata-Tc gated MoE PID: ${pid}"
echo "Run ID: ${RUN_ID}"
echo "Seeds: ${SEEDS}"
echo "Tc threshold: ${TC_THRESHOLD_K} K"
echo "Results: ${OUTPUT_ROOT}"
echo "Main log: ${LOG_ROOT}/magndata_tc_moe.log"
echo "Follow with: tail -f ${LOG_ROOT}/magndata_tc_moe.log"
echo "Check with: bash submit/check_magndata_tc_moe.sh ${RUN_ID}"
