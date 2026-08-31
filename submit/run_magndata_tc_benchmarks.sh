#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORKER="submit_python/run_magndata_tc_benchmarks.sh"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/magndata_tc_benchmark/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/property_predictor_training/magndata_tc/${RUN_ID}}"
TEST_LOG_ROOT="${TEST_LOG_ROOT:-${PROJECT_ROOT}/logs/tests/property_predictors/magndata_tc/${RUN_ID}}"
SEEDS="${SEEDS:-42 123 3407}"
GPU_ID="${GPU_ID:-0}"
BASE_PRETRAINED_DIFFUSION_MODEL="${BASE_PRETRAINED_DIFFUSION_MODEL:-${PROJECT_ROOT}/output/singlerun/2026-06-27/00-32-50-mp20_base}"
JOINT_PRETRAINED_DIFFUSION_MODEL="${JOINT_PRETRAINED_DIFFUSION_MODEL:-${PROJECT_ROOT}/output/singlerun/2026-08-05/13-12-29-mp20_fe_bg_eh}"
REUSE_BENCHMARK_ROOT="${REUSE_BENCHMARK_ROOT:-}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the cgdit Conda environment before launching." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TEST_LOG_ROOT}"
bash -n "${WORKER}"
test -s data/magndata/train.csv
test -s data/magndata/val.csv
test -s data/magndata/test.csv
test -s data/magndata/split_manifest.json
for pretrained_model in \
    "${BASE_PRETRAINED_DIFFUSION_MODEL}" \
    "${JOINT_PRETRAINED_DIFFUSION_MODEL}"; do
    test -s "${pretrained_model}/hparams.yaml"
    compgen -G "${pretrained_model}/*.ckpt" >/dev/null
done
if [[ -n "${REUSE_BENCHMARK_ROOT}" ]]; then
    test -d "${REUSE_BENCHMARK_ROOT}"
fi

python -m pytest -q tests/test_magndata_tc_pipeline.py \
    > "${TEST_LOG_ROOT}/pytest_preflight.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
RUN_ID="${RUN_ID}" \
SEEDS="${SEEDS}" \
BASE_PRETRAINED_DIFFUSION_MODEL="${BASE_PRETRAINED_DIFFUSION_MODEL}" \
JOINT_PRETRAINED_DIFFUSION_MODEL="${JOINT_PRETRAINED_DIFFUSION_MODEL}" \
REUSE_BENCHMARK_ROOT="${REUSE_BENCHMARK_ROOT}" \
nohup bash "${WORKER}" > "${LOG_ROOT}/magndata_tc_benchmark.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${LOG_ROOT}/master.pid"

echo "Started Magndata-Tc benchmark PID: ${pid}"
echo "Run ID: ${RUN_ID}"
echo "Seeds: ${SEEDS}"
if [[ -n "${REUSE_BENCHMARK_ROOT}" ]]; then
    echo "Reusing completed arms from: ${REUSE_BENCHMARK_ROOT}"
fi
echo "Results: ${OUTPUT_ROOT}"
echo "Main log: ${LOG_ROOT}/magndata_tc_benchmark.log"
echo "Follow with: tail -f ${LOG_ROOT}/magndata_tc_benchmark.log"
echo "Check with: bash submit/check_magndata_tc_benchmarks.sh ${RUN_ID}"
