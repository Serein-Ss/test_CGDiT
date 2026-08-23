#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MANIFEST="${1:-${PROJECT_ROOT}/output/rl_diagnostics/seed42_pipeline_latest.txt}"
test -s "${MANIFEST}"

job_ids=()
while IFS='=' read -r key value; do
    case "${key}" in
        preflight_job|training_array_job|baseline_eval_job|generation_array_job|analysis_job)
            job_ids+=("${value}")
            ;;
    esac
done < "${MANIFEST}"

echo "Manifest: ${MANIFEST}"
if [[ ${#job_ids[@]} -eq 0 ]]; then
    echo "No job IDs found." >&2
    exit 1
fi

joined="$(IFS=,; echo "${job_ids[*]}")"
squeue -j "${joined}" -o '%.18i %.20j %.10T %.10M %.24R' || true
echo
sacct -j "${joined}" --format=JobID,JobName%20,State,ExitCode,Elapsed -n -X || true
