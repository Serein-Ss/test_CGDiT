#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORKER="submit_python/run_rl_smoke_conda.sh"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
LOG_DIR="${PROJECT_ROOT}/logs/remote_rl/${RUN_ID}"
LOG_FILE="${LOG_DIR}/zrs-gen-rl.log"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the existing Conda environment before running this script." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
bash -n "${WORKER}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable in the active Conda environment'; print('gpu=', torch.cuda.get_device_name(0))"

export PROJECT_ROOT RUN_ID
nohup bash "${WORKER}" > "${LOG_FILE}" 2>&1 < /dev/null &
pid=$!

echo "Started zrs-gen-rl PID: ${pid}"
echo "Conda environment: ${CONDA_PREFIX}"
echo "Run ID: ${RUN_ID}"
echo "Main log: ${LOG_FILE}"
echo "Follow with: tail -f ${LOG_FILE}"
