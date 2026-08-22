#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "${PROJECT_ROOT}"

WORKER="submit_python/run_seed42_property_evaluation.sh"
LOG_FILE="logs/seed42_property_evaluation_main.log"

mkdir -p logs
bash -n "${WORKER}"

nohup bash "${WORKER}" > "${LOG_FILE}" 2>&1 < /dev/null &
pid=$!

echo "Started PID: ${pid}"
echo "Main log: ${PROJECT_ROOT}/${LOG_FILE}"
echo "Follow with: tail -f ${PROJECT_ROOT}/${LOG_FILE}"
