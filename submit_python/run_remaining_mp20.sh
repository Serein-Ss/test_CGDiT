#!/bin/bash

set -Eeuo pipefail

export PROJECT_ROOT="/root/private_data/rszhong/workspace/test_CGDiT"
export HYDRA_JOBS="${PROJECT_ROOT}/output"
export WABDB_DIR="${PROJECT_ROOT}/wandb"
export WANDB_MODE=offline

cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs/generative_model_training/mp20_remaining"
mkdir -p "${LOG_DIR}" "${WABDB_DIR}"

if pgrep -f "${PROJECT_ROOT}/cgdit/run.py" >/dev/null; then
    echo "A CGDiT training process is already running:" >&2
    pgrep -af "${PROJECT_ROOT}/cgdit/run.py" >&2
    exit 1
fi

EXPNAMES=(
    "mp20_fe_bg_eh"
    "mp20_fe_bg"
    "mp20_fe_eh"
    "mp20_bg_eh"
)

MODELS=(
    "experiments/exp_mp20_fe_bg_eh"
    "experiments/exp_mp20_fe_bg"
    "experiments/exp_mp20_fe_eh"
    "experiments/exp_mp20_bg_eh"
)

TOTAL=${#EXPNAMES[@]}

echo "========================================================"
echo "CGDiT mp_20 remaining experiments"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "HYDRA_JOBS: ${HYDRA_JOBS}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

for i in "${!EXPNAMES[@]}"; do
    expname="${EXPNAMES[$i]}"
    model="${MODELS[$i]}"
    log_file="${LOG_DIR}/${expname}.log"

    echo
    echo "--------------------------------------------------------"
    echo "Experiment $((i + 1))/${TOTAL}: ${expname}"
    echo "Model config: conf/model/${model}.yaml"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------------------"

    python "${PROJECT_ROOT}/cgdit/run.py" \
        data=mp_20 \
        model="${model}" \
        expname="${expname}" \
        logging.wandb.mode=offline \
        2>&1 | tee "${log_file}"

    echo "Completed: ${expname} at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo
echo "All ${TOTAL} remaining experiments completed at $(date '+%Y-%m-%d %H:%M:%S')"
