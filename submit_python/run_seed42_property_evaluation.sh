#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_ROOT}"

FE_RUN="output/singlerun/2026-08-16/01-27-48-mp20_predictor_fe"
BG_RUN="output/singlerun/2026-08-16/15-18-32-mp20_predictor_bg"
EH_RUN="output/singlerun/2026-08-17/05-54-35-mp20_predictor_eh"

BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"

if [[ "${DEVICE}" != "cpu" ]] &&
   [[ "${ALLOW_CONCURRENT_GPU:-0}" != "1" ]] &&
   pgrep -f 'cgdit/run.py.*mp20_predictor' >/dev/null; then
    echo "A property predictor is currently training." >&2
    echo "Wait for it to finish, use DEVICE=cpu, or explicitly set ALLOW_CONCURRENT_GPU=1." >&2
    exit 1
fi

for run_dir in "${FE_RUN}" "${BG_RUN}" "${EH_RUN}"; do
    [[ -d "${run_dir}" ]] || { echo "Missing predictor run: ${run_dir}" >&2; exit 1; }
    [[ -s "${run_dir}/test_preds.npy" ]] || { echo "Missing test_preds.npy: ${run_dir}" >&2; exit 1; }
    [[ -s "${run_dir}/test_targets.npy" ]] || { echo "Missing test_targets.npy: ${run_dir}" >&2; exit 1; }
    [[ "$(find "${run_dir}" -maxdepth 1 -type f -name '*.ckpt' | wc -l)" -eq 1 ]] || {
        echo "Expected exactly one checkpoint: ${run_dir}" >&2
        exit 1
    }
done

uv run --locked python -m pytest \
    tests/test_property_predictor.py \
    tests/test_generated_property_evaluation.py -q

uv run --locked python -m scripts.cli.evaluation.evaluate_generated_properties \
    --root_path output/singlerun \
    --fe_run "${FE_RUN}" \
    --bg_run "${BG_RUN}" \
    --eh_run "${EH_RUN}" \
    --output_dir assets/model_results/source_data/property_evaluation_seed42 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}"
