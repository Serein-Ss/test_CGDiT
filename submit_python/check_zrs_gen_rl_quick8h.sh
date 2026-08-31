#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MANIFEST="${1:-${PROJECT_ROOT}/output/test/reinforcement_learning/2026-08-23/15-40-35-quick8h-seed42/submission_manifest.txt}"
test -s "${MANIFEST}"

job_ids=()
while IFS='=' read -r key value; do
    case "${key}" in
        training_array_job|generation_array_job|analysis_job) job_ids+=("${value}") ;;
    esac
done < "${MANIFEST}"

joined="$(IFS=,; echo "${job_ids[*]}")"
echo "Manifest: ${MANIFEST}"
squeue -j "${joined}" -o '%.18i %.20j %.10T %.10M %.24R' || true
echo
sacct -j "${joined}" --format=JobID,JobName%20,State,ExitCode,Elapsed -n -X || true
