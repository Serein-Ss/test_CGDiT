#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set by the launcher}"
LOG_ROOT="${LOG_ROOT:?LOG_ROOT must be set by the launcher}"
UPDATES="${UPDATES:-8}"
ARM_WORKER="submit_python/run_rl_fe_pilot_arm_conda.sh"
SEQUENCE_RESULTS="${OUTPUT_ROOT}/train_results"
SEQUENCE_MANIFEST="${SEQUENCE_RESULTS}/sequence_manifest.txt"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the existing Conda environment before running this worker." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${SEQUENCE_RESULTS}" "${LOG_ROOT}"
bash -n "${ARM_WORKER}"

record_failure() {
    local status=$?
    {
        echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "status=failed"
        echo "exit_code=${status}"
    } >> "${SEQUENCE_MANIFEST}"
    exit "${status}"
}
trap record_failure ERR

{
    echo "run_id=${RUN_ID:-unknown}"
    echo "execution_mode=single_gpu_sequential"
    echo "status=running"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "arm_order=P0,G0,P2,G2"
} > "${SEQUENCE_MANIFEST}"

for arm in P0 G0 P2 G2; do
    echo "Starting ${arm} at $(date '+%Y-%m-%d %H:%M:%S %z')"
    bash "${ARM_WORKER}" "${arm}" > "${LOG_ROOT}/${arm}.log" 2>&1
    echo "Completed ${arm} at $(date '+%Y-%m-%d %H:%M:%S %z')"
done

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${SEQUENCE_MANIFEST}"

echo "zrs-gen-rl FE pilot sequence completed."
echo "Results: ${OUTPUT_ROOT}"
