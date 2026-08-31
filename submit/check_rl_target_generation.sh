#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
    RUN_ID=$(find "${PROJECT_ROOT}/logs/structure_generation/reinforcement_learning" -mindepth 1 -maxdepth 1 -type d \
        -printf '%f\n' 2>/dev/null | sort | tail -n 1)
fi
if [[ -z "${RUN_ID}" ]]; then
    echo "No RL generation run was found." >&2
    exit 1
fi

source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
OUTPUT_ROOT="$(rl_run_root "${PROJECT_ROOT}" reinforcement_learning rl-target-generation "${RUN_ID}")"
LOG_ROOT="${PROJECT_ROOT}/logs/structure_generation/reinforcement_learning/${RUN_ID}"

echo "========== RL generation ${RUN_ID} =========="
pgrep -af 'run_rl_target_generation|scripts.cli.generation.generate|scripts.cli.evaluation' \
    || echo "No active RL generation process"
echo
cat "${OUTPUT_ROOT}/run_manifest.txt" 2>/dev/null || true
echo
for policy_root in "${OUTPUT_ROOT}"/*; do
    [[ -f "${policy_root}/policy_manifest.txt" ]] || continue
    echo "---------- ${policy_root##*/} ----------"
    cat "${policy_root}/policy_manifest.txt"
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
