#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="/root/private_data/rszhong/workspace/test_CGDiT"
cd "${PROJECT_ROOT}"

WORKER="submit_python/run_all_mp20_predictors.sh"
LOG_FILE="logs/other/orchestration/property_predictor_training/mp20_seed42_main.log"

mkdir -p "$(dirname "${LOG_FILE}")"
bash -n "${WORKER}"

nohup bash "${WORKER}" > "${LOG_FILE}" 2>&1 < /dev/null &
pid=$!

echo "Started PID: ${pid}"
echo "Main log: ${PROJECT_ROOT}/${LOG_FILE}"
echo "Follow with: tail -f ${PROJECT_ROOT}/${LOG_FILE}"
