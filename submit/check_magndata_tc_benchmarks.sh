#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
    RUN_ID=$(find "${PROJECT_ROOT}/logs/property_predictor_training/magndata_tc" -mindepth 1 -maxdepth 1 -type d \
        -printf '%f\n' 2>/dev/null | sort | tail -n 1)
fi
if [[ -z "${RUN_ID}" ]]; then
    echo "No Magndata-Tc run was found." >&2
    exit 1
fi

OUTPUT_ROOT="${PROJECT_ROOT}/output/magndata_tc_benchmark/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/property_predictor_training/magndata_tc/${RUN_ID}"

echo "========== Magndata-Tc ${RUN_ID} =========="
pgrep -af 'run_magndata_tc_benchmarks|cgdit/run.py.*magndata' || echo "No active benchmark process"
echo
cat "${OUTPUT_ROOT}/run_manifest.txt" 2>/dev/null || true
echo

complete=0
seeds_text=$(awk -F= '$1 == "seeds" {sub(/^seeds=/, ""); print; exit}' \
    "${OUTPUT_ROOT}/run_manifest.txt" 2>/dev/null || true)
read -r -a seed_list <<< "${seeds_text:-42 123 3407}"
for method in \
    m3gnet_scratch \
    diffusion_scratch \
    diffusion_base_pretrained \
    diffusion_joint_pretrained; do
    for seed in "${seed_list[@]}"; do
        run_dir="${OUTPUT_ROOT}/${method}/seed_${seed}"
        if [[ -s "${run_dir}/test_preds.npy" && -s "${run_dir}/test_targets.npy" ]] && \
           compgen -G "${run_dir}/*.ckpt" >/dev/null; then
            status=complete
            complete=$((complete + 1))
        else
            status=pending
        fi
        echo "${method} seed=${seed}: ${status}"
    done
done
echo "Complete arms: ${complete}/$((4 * ${#seed_list[@]}))"
echo
tail -n 30 "${LOG_ROOT}/magndata_tc_benchmark.log" 2>/dev/null || true
