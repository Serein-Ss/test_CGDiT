#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORKER="submit_python/run_rl_target_generation.sh"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
RUN_DATE="${RUN_DATE:-${RUN_ID:0:4}-${RUN_ID:4:2}-${RUN_ID:6:2}}"
RUN_TIME="${RUN_TIME:-${RUN_ID:9:2}-${RUN_ID:11:2}-${RUN_ID:13:2}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/reinforcement_learning/${RUN_DATE}/${RUN_TIME}-rl-target-generation}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/structure_generation/reinforcement_learning/${RUN_ID}}"
TEST_LOG_ROOT="${TEST_LOG_ROOT:-${PROJECT_ROOT}/logs/tests/structure_generation/reinforcement_learning/${RUN_ID}}"
GPU_ID="${GPU_ID:-0}"

FE_RL_MODEL_PATH="${FE_RL_MODEL_PATH:-${PROJECT_ROOT}/output/reinforcement_learning/2026-08-23/21-47-35-grpo-pipo-probe32/grpo_fe_seed42_pipo/model}"
BG_RL_MODEL_PATH="${BG_RL_MODEL_PATH:-${PROJECT_ROOT}/output/reinforcement_learning/2026-08-23/21-47-35-grpo-pipo-probe32/grpo_bg_seed42_pipo/model}"
FE_TARGET_VALUE="${FE_TARGET_VALUE:--1.5}"
BG_TARGET_VALUE="${BG_TARGET_VALUE:-2.0}"
FE_TARGET_LABEL="${FE_TARGET_LABEL:-fe_m1p5}"
BG_TARGET_LABEL="${BG_TARGET_LABEL:-bg_2}"

FE_RUN="${FE_RUN:-${PROJECT_ROOT}/output/singlerun/2026-08-16/01-27-48-mp20_predictor_fe}"
BG_RUN="${BG_RUN:-${PROJECT_ROOT}/output/singlerun/2026-08-16/15-18-32-mp20_predictor_bg}"
EH_RUN="${EH_RUN:-${PROJECT_ROOT}/output/singlerun/2026-08-17/05-54-35-mp20_predictor_eh}"
GT_FILE="${GT_FILE:-${PROJECT_ROOT}/data/mp_20/test.csv}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the cgdit Conda environment before launching." >&2
    exit 1
fi
for target_label in "${FE_TARGET_LABEL}" "${BG_TARGET_LABEL}"; do
    if ! [[ "${target_label}" =~ ^[A-Za-z0-9_]+$ ]]; then
        echo "FE_TARGET_LABEL and BG_TARGET_LABEL may contain only letters, numbers, and underscores." >&2
        exit 2
    fi
done
python -c "import sys; float(sys.argv[1]); float(sys.argv[2])" \
    "${FE_TARGET_VALUE}" "${BG_TARGET_VALUE}" >/dev/null

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TEST_LOG_ROOT}"
bash -n "${WORKER}"
for rl_model_path in "${FE_RL_MODEL_PATH}" "${BG_RL_MODEL_PATH}"; do
    test -s "${rl_model_path}/hparams.yaml"
    compgen -G "${rl_model_path}/*.ckpt" >/dev/null
done
test -s "${FE_RUN}/hparams.yaml"
test -s "${BG_RUN}/hparams.yaml"
test -s "${EH_RUN}/hparams.yaml"
test -s "${GT_FILE}"

python -m pytest -q \
    tests/test_rl_target_generation.py \
    tests/test_generated_property_evaluation.py \
    > "${TEST_LOG_ROOT}/pytest_preflight.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
RUN_ID="${RUN_ID}" \
FE_RL_MODEL_PATH="${FE_RL_MODEL_PATH}" \
BG_RL_MODEL_PATH="${BG_RL_MODEL_PATH}" \
FE_TARGET_VALUE="${FE_TARGET_VALUE}" \
BG_TARGET_VALUE="${BG_TARGET_VALUE}" \
FE_TARGET_LABEL="${FE_TARGET_LABEL}" \
BG_TARGET_LABEL="${BG_TARGET_LABEL}" \
FE_RUN="${FE_RUN}" \
BG_RUN="${BG_RUN}" \
EH_RUN="${EH_RUN}" \
GT_FILE="${GT_FILE}" \
nohup bash "${WORKER}" >> "${LOG_ROOT}/rl_target_generation.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${LOG_ROOT}/master.pid"

echo "Started RL target generation PID: ${pid}"
echo "Run ID: ${RUN_ID}"
echo "FE target: formation_energy_per_atom=${FE_TARGET_VALUE}"
echo "BG target: band_gap=${BG_TARGET_VALUE}"
echo "Results: ${OUTPUT_ROOT}"
echo "Main log: ${LOG_ROOT}/rl_target_generation.log"
echo "Follow with: tail -f ${LOG_ROOT}/rl_target_generation.log"
echo "Check with: bash submit/check_rl_target_generation.sh ${RUN_ID}"
