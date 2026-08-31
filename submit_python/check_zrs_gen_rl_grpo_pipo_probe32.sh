#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_ROOT}"
MANIFEST="${1:-${PROJECT_ROOT}/output/reinforcement_learning/2026-08-23/21-47-35-grpo-pipo-probe32/submission_manifest.txt}"
test -s "${MANIFEST}"

JOB_ID="$(awk -F= '$1 == "job_id" {print $2}' "${MANIFEST}")"
TRAIN_ROOT="$(awk -F= '$1 == "train_root" {print substr($0, index($0, "=") + 1)}' "${MANIFEST}")"
LOG_ROOT="$(awk -F= '$1 == "log_root" {print substr($0, index($0, "=") + 1)}' "${MANIFEST}")"
if [[ -z "${TRAIN_ROOT}" ]]; then
    TRAIN_ROOT="$(dirname "${MANIFEST}")"
fi
if [[ -z "${LOG_ROOT}" ]]; then
    run_dir="$(basename "${TRAIN_ROOT}")"
    run_date="$(basename "$(dirname "${TRAIN_ROOT}")" | tr -d '-')"
    run_time="${run_dir%%-grpo-pipo-probe32}"
    LOG_ROOT="${PROJECT_ROOT}/logs/reinforcement_learning/slurm/${run_date}-$(tr -d '-' <<< "${run_time}")-grpo-pipo-probe32"
fi

echo "Manifest: ${MANIFEST}"
echo "Job ID: ${JOB_ID}"
squeue -j "${JOB_ID}" -o "%.18i %.9P %.20j %.8T %.10M %.6D %R" || true
for task in 0 1; do
    manifest="${TRAIN_ROOT}/diagnostics/task_${task}/manifest.txt"
    log="${LOG_ROOT}/task_${task}/training.log"
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
