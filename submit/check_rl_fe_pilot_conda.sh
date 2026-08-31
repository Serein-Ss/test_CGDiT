#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ID="${1:?Usage: bash submit/check_rl_fe_pilot_conda.sh RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/reinforcement_learning/local_runs/${RUN_ID}"
MASTER_PID_FILE="${LOG_ROOT}/master.pid"
LAUNCH_MANIFEST="${LOG_ROOT}/launch_manifest.txt"

if [[ ! -f "${MASTER_PID_FILE}" || ! -f "${LAUNCH_MANIFEST}" ]]; then
    echo "Run metadata not found under: ${LOG_ROOT}" >&2
    exit 1
fi

pid=$(<"${MASTER_PID_FILE}")
output_root=$(awk -F= '$1 == "output_root" {print substr($0, index($0, "=") + 1)}' "${LAUNCH_MANIFEST}")
sequence_manifest="${output_root}/train_results/sequence_manifest.txt"
master_status="starting"
if [[ -f "${sequence_manifest}" ]]; then
    recorded=$(awk -F= '$1 == "status" {value=$2} END {print value}' "${sequence_manifest}")
    [[ -n "${recorded}" ]] && master_status="${recorded}"
fi
if [[ "${master_status}" != "complete" && "${master_status}" != "failed" ]]     && ! kill -0 "${pid}" 2>/dev/null
then
    master_status="stopped"
fi

echo "Master PID: ${pid}"
echo "Master status: ${master_status}"
printf '%-4s %-12s %s\n' ARM STATUS LOG
ARM_RESULTS=(
    "P0:ppo_fe_seed42"
    "G0:grpo_fe_seed42"
    "P2:ppo_fe_seed42_pirl"
    "G2:grpo_fe_seed42_pirl"
)
for item in "${ARM_RESULTS[@]}"; do
    arm="${item%%:*}"
    result_name="${item#*:}"
    manifest="${output_root}/${result_name}/train_results/arm_manifest.txt"
    status="pending"
    if [[ -f "${manifest}" ]]; then
        recorded=$(awk -F= '$1 == "status" {value=$2} END {print value}' "${manifest}")
        [[ -n "${recorded}" ]] && status="${recorded}"
    elif [[ "${master_status}" == "failed" || "${master_status}" == "stopped" ]]; then
        status="not_started"
    fi
    printf '%-4s %-12s %s\n' "${arm}" "${status}" "${LOG_ROOT}/${arm}.log"
done
