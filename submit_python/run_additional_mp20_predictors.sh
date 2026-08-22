#!/usr/bin/env bash
# Train three additional random-seed replicas for each MP-20 property predictor.

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PROJECT_ROOT
export HYDRA_JOBS="${HYDRA_JOBS:-${PROJECT_ROOT}/output}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${PROJECT_ROOT}"

MAX_EPOCHS="${MAX_EPOCHS:-300}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

SEEDS=(123 2026 3407)
PROPERTIES=(formation_energy_per_atom band_gap e_above_hull)
MODEL_NAMES=(fe bg eh)

LOG_DIR="${PROJECT_ROOT}/logs/mp20_predictors_multiseed"
mkdir -p "${LOG_DIR}" "${WANDB_DIR}"

# Share the lock with the original predictor-training script so two training
# jobs cannot accidentally compete for the same GPU.
if command -v flock >/dev/null 2>&1; then
    exec 9>"${PROJECT_ROOT}/logs/mp20_predictors/run.lock"
    if ! flock -n 9; then
        echo "Another MP-20 predictor training run is already active." >&2
        exit 1
    fi
fi

echo "============================================================"
echo "Additional MP-20 property predictor training"
echo "Seeds=${SEEDS[*]}"
echo "Epochs=${MAX_EPOCHS}"
echo "Batch sizes train/eval=${TRAIN_BATCH_SIZE}/${EVAL_BATCH_SIZE}"
echo "Start time=$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

uv run --locked python --version
uv run --locked python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
uv run --locked python -m pytest tests/test_property_predictor.py -q

for seed in "${SEEDS[@]}"; do
    for i in "${!PROPERTIES[@]}"; do
        prop="${PROPERTIES[$i]}"
        model_name="${MODEL_NAMES[$i]}"
        expname="mp20_predictor_${model_name}_seed${seed}"
        log_file="${LOG_DIR}/${expname}.log"

        existing_dir="$(find "${HYDRA_JOBS}/singlerun" -mindepth 2 -maxdepth 2 \
            -type d -name "*-${expname}" 2>/dev/null | sort | tail -n 1 || true)"
        if [[ -n "${existing_dir}" ]] &&
           find "${existing_dir}" -maxdepth 1 -type f -name '*.ckpt' -print -quit | grep -q . &&
           [[ -s "${existing_dir}/test_preds.npy" ]] &&
           [[ -s "${existing_dir}/test_targets.npy" ]]; then
            echo "[SKIP] Completed ${expname}: ${existing_dir}"
            continue
        fi

        echo
        echo "------------------------------------------------------------"
        echo "Training ${expname}: ${prop}, seed=${seed}"
        echo "Start time=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------------------------"

        uv run --locked python cgdit/run.py \
            data=mp_20_surrogate \
            model=m3gnet \
            "data.prop=${prop}" \
            "data.train_max_epochs=${MAX_EPOCHS}" \
            "data.datamodule.batch_size.train=${TRAIN_BATCH_SIZE}" \
            "data.datamodule.batch_size.val=${EVAL_BATCH_SIZE}" \
            "data.datamodule.batch_size.test=${EVAL_BATCH_SIZE}" \
            "train.random_seed=${seed}" \
            "expname=${expname}" \
            logging.wandb.mode=offline \
            2>&1 | tee "${log_file}"

        echo "Completed ${expname} at $(date '+%Y-%m-%d %H:%M:%S')"
    done
done

echo
echo "All additional MP-20 predictor trainings completed."
echo "Outputs: ${HYDRA_JOBS}/singlerun/YYYY-MM-DD/*-mp20_predictor_*_seed*"
echo "Logs: ${LOG_DIR}"
