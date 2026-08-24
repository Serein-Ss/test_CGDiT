#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_ROOT}"
JOB_ID="${1:-}"

if [[ -z "${JOB_ID}" ]]; then
    JOB_ID="$(awk -F= '$1 == "job_id" {print $2}' output/rl_diagnostics/grpo_pipo_probe32_latest.txt)"
fi

echo "Job ID: ${JOB_ID}"
squeue -j "${JOB_ID}" -o "%.18i %.9P %.20j %.8T %.10M %.6D %R" || true
for task in 0 1; do
    manifest="output/rl_diagnostics/${JOB_ID}-seed42-grpo-pipo-probe32/task_${task}/manifest.txt"
    log="output/rl_diagnostics/${JOB_ID}-seed42-grpo-pipo-probe32/task_${task}/training.log"
    status="pending"
    if [[ -f "${manifest}" ]]; then
        status="$(awk -F= '$1 == "status" {value=$2} END {print value}' "${manifest}")"
    fi
    updates=0
    if [[ -f "${log}" ]]; then
        updates="$(grep -c '^{"step":' "${log}" || true)"
    fi
    printf 'task=%s status=%s updates=%s/200 log=%s\n' "${task}" "${status}" "${updates}" "${log}"
done
