#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ID="${1:?Usage: bash submit/pack_rl_fe_pilot_results.sh RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/reinforcement_learning/local_runs/${RUN_ID}"
LAUNCH_MANIFEST="${LOG_ROOT}/launch_manifest.txt"

if [[ ! -f "${LAUNCH_MANIFEST}" ]]; then
    echo "Launch manifest not found: ${LAUNCH_MANIFEST}" >&2
    exit 1
fi
OUTPUT_ROOT=$(awk -F= '$1 == "output_root" {print substr($0, index($0, "=") + 1)}' "${LAUNCH_MANIFEST}")
if [[ -z "${OUTPUT_ROOT}" || "${OUTPUT_ROOT}" != "${PROJECT_ROOT}/"* ]]; then
    echo "Invalid output_root in ${LAUNCH_MANIFEST}" >&2
    exit 1
fi

for result_name in     ppo_fe_seed42     ppo_fe_seed42_pirl     grpo_fe_seed42     grpo_fe_seed42_pirl
do
    manifest="${OUTPUT_ROOT}/${result_name}/train_results/arm_manifest.txt"
    if [[ ! -f "${manifest}" ]] || ! grep -q '^status=complete$' "${manifest}"; then
        echo "Result is not complete: ${result_name}" >&2
        exit 1
    fi
done

mkdir -p "${PROJECT_ROOT}/tmp"
archive="${PROJECT_ROOT}/tmp/zrs-gen-rl-fe-pilot-${RUN_ID}.tar.gz"
output_relative="${OUTPUT_ROOT#${PROJECT_ROOT}/}"
log_relative="${LOG_ROOT#${PROJECT_ROOT}/}"
tar -czf "${archive}" -C "${PROJECT_ROOT}" "${output_relative}" "${log_relative}"

echo "Archive: ${archive}"
