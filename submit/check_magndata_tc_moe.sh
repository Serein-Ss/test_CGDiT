#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
    RUN_ID=$(find "${PROJECT_ROOT}/logs/magndata_tc_moe" -mindepth 1 -maxdepth 1 -type d \
        -printf '%f\n' 2>/dev/null | sort | tail -n 1)
fi
if [[ -z "${RUN_ID}" ]]; then
    echo "No Magndata-Tc gated MoE run was found." >&2
    exit 1
fi

OUTPUT_ROOT="${PROJECT_ROOT}/output/magndata_tc_moe/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/magndata_tc_moe/${RUN_ID}"

echo "========== Magndata-Tc gated MoE ${RUN_ID} =========="
pgrep -af 'run_magndata_tc_moe|cgdit/run.py.*magndata_tc_.*expert|cgdit/run.py.*magndata_tc_gate|predict_tc_gate' \
    || echo "No active gated MoE process"
echo
cat "${OUTPUT_ROOT}/run_manifest.txt" 2>/dev/null || true
echo

seeds_text=$(awk -F= '$1 == "seeds" {sub(/^seeds=/, ""); print; exit}' \
    "${OUTPUT_ROOT}/run_manifest.txt" 2>/dev/null || true)
read -r -a seed_list <<< "${seeds_text:-42 123 3407}"
complete=0
for seed in "${seed_list[@]}"; do
    gate_dir="${OUTPUT_ROOT}/seed_${seed}/gate"
    if [[ -s "${gate_dir}/test_probs.npy" && -s "${gate_dir}/val_probs.npy" ]] && \
       compgen -G "${gate_dir}/*.ckpt" >/dev/null; then
        gate_status=complete
        complete=$((complete + 1))
    else
        gate_status=pending
    fi
    echo "gate seed=${seed}: ${gate_status}"

    for branch in low high; do
        expert_dir="${OUTPUT_ROOT}/seed_${seed}/${branch}_expert"
        if [[ -s "${expert_dir}/test_preds.npy" && -s "${expert_dir}/test_targets.npy" ]] && \
           compgen -G "${expert_dir}/*.ckpt" >/dev/null; then
            expert_status=complete
            complete=$((complete + 1))
        else
            expert_status=pending
        fi
        echo "${branch}_expert seed=${seed}: ${expert_status}"
    done
done
echo "Complete arms: ${complete}/$((3 * ${#seed_list[@]}))"

if [[ -s "${OUTPUT_ROOT}/regression_metrics_aggregate.csv" ]]; then
    echo "Aggregate metrics: complete"
else
    echo "Aggregate metrics: pending"
fi
echo
tail -n 35 "${LOG_ROOT}/magndata_tc_moe.log" 2>/dev/null || true
