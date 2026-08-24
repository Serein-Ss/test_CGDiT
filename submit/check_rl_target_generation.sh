#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
    RUN_ID=$(find "${PROJECT_ROOT}/logs/rl_generation" -mindepth 1 -maxdepth 1 -type d \
        -printf '%f\n' 2>/dev/null | sort | tail -n 1)
fi
if [[ -z "${RUN_ID}" ]]; then
    echo "No RL generation run was found." >&2
    exit 1
fi

OUTPUT_ROOT="${PROJECT_ROOT}/output/rl_generation/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/rl_generation/${RUN_ID}"

echo "========== RL generation ${RUN_ID} =========="
pgrep -af 'run_rl_target_generation|scripts.cli.generation.generate|scripts.cli.evaluation' \
    || echo "No active RL generation process"
echo
cat "${OUTPUT_ROOT}/run_manifest.txt" 2>/dev/null || true
echo
for policy_name in fe bg; do
    echo "---------- ${policy_name^^} policy ----------"
    cat "${OUTPUT_ROOT}/${policy_name}/policy_manifest.txt" 2>/dev/null || \
        echo "Policy has not started"
done
echo
printf "Generated structures: "
find "${OUTPUT_ROOT}" -type f \
    \( -name 'eval_gen_abinitio_empirical_rl_*_n4096_seed42.pt' \
       -o -name 'eval_gen_template_rl_*_n4096_seed42.pt' \) \
    -size +0c 2>/dev/null | wc -l
printf "Structural metrics:   "
find "${OUTPUT_ROOT}" -type f -name 'eval_metrics_gen_*rl*_n4096_seed42.json' \
    -size +0c 2>/dev/null | wc -l
printf "Property metrics:     "
find "${OUTPUT_ROOT}" -type f \
    -name 'eval_property_metrics_gen_*rl*_n4096_seed42_predictor_seed42.json' \
    -size +0c 2>/dev/null | wc -l
echo "Expected for each line: 4 (FE/BG x ab initio/template)"
echo
tail -n 30 "${LOG_ROOT}/rl_target_generation.log" 2>/dev/null || true
