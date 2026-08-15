#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="/root/private_data/rszhong/workspace/test_CGDiT"
cd "${PROJECT_ROOT}"

WORKER="submit_python/run_all_generation_evaluation.sh"
LOG_FILE="logs/all_generation_evaluation_main.log"

mkdir -p logs
bash -n "${WORKER}"

nohup bash "${WORKER}" > "${LOG_FILE}" 2>&1 < /dev/null &
pid=$!

echo "Started PID: ${pid}"
echo "Main log: ${PROJECT_ROOT}/${LOG_FILE}"
echo "Follow with: tail -f ${PROJECT_ROOT}/${LOG_FILE}"
