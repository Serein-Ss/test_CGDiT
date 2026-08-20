#!/usr/bin/env bash
# Train independent M3GNet predictors for MP-20 formation energy, band gap,
# and energy above hull. Existing MP-20 graph caches are shared by all tasks.

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PROJECT_ROOT
export HYDRA_JOBS="${HYDRA_JOBS:-${PROJECT_ROOT}/output}"
export WABDB_DIR="${WABDB_DIR:-${PROJECT_ROOT}/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${PROJECT_ROOT}"

MAX_EPOCHS="${MAX_EPOCHS:-300}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
FAST_DEV_RUN="${FAST_DEV_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

LOG_DIR="${PROJECT_ROOT}/logs/mp20_predictors"
mkdir -p "${LOG_DIR}" "${WABDB_DIR}"

if command -v flock >/dev/null 2>&1; then
    exec 9>"${LOG_DIR}/run.lock"
    if ! flock -n 9; then
        echo "Another MP-20 predictor training run is already active." >&2
        exit 1
    fi
fi

PROPERTIES=(
    formation_energy_per_atom
    band_gap
    e_above_hull
)

EXPNAMES=(
    mp20_predictor_fe
    mp20_predictor_bg
    mp20_predictor_eh
)

echo "============================================================"
echo "MP-20 property predictor training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "Git commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Epochs=${MAX_EPOCHS}"
echo "Batch sizes train/eval=${TRAIN_BATCH_SIZE}/${EVAL_BATCH_SIZE}"
echo "Start time=$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python --version
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

for required_file in \
    data/mp_20/train.csv \
    data/mp_20/val.csv \
    data/mp_20/test.csv \
    data/mp_20/train_sym.pt \
    data/mp_20/val_sym.pt \
    data/mp_20/test_sym.pt; do
    [[ -s "${required_file}" ]] || { echo "Missing required file: ${required_file}" >&2; exit 1; }
done

python -c "import torch; paths=['data/mp_20/train_sym.pt','data/mp_20/val_sym.pt','data/mp_20/test_sym.pt']; props=['formation_energy_per_atom','band_gap','e_above_hull']; [(lambda d,p: (_ for _ in ()).throw(RuntimeError(f'{p} lacks required properties')) if any(k not in d[0] for k in props) else print(p, len(d), 'properties_ok'))(torch.load(p,map_location='cpu',weights_only=False),p) for p in paths]"

python -m pytest tests/test_property_predictor.py -q

for i in "${!PROPERTIES[@]}"; do
    prop="${PROPERTIES[$i]}"
    expname="${EXPNAMES[$i]}"
    log_file="${LOG_DIR}/${expname}.log"

    if [[ "${SKIP_EXISTING}" == "1" ]]; then
        existing_ckpt="$(find "${HYDRA_JOBS}/singlerun" -type f -path "*-${expname}/*.ckpt" -print -quit 2>/dev/null || true)"
        if [[ -n "${existing_ckpt}" ]]; then
            echo "[SKIP] Existing checkpoint for ${expname}: ${existing_ckpt}"
            continue
        fi
    fi

    cmd=(
        python cgdit/run.py
        data=mp_20_surrogate
        model=m3gnet
        "data.prop=${prop}"
        "data.train_max_epochs=${MAX_EPOCHS}"
        "data.datamodule.batch_size.train=${TRAIN_BATCH_SIZE}"
        "data.datamodule.batch_size.val=${EVAL_BATCH_SIZE}"
        "data.datamodule.batch_size.test=${EVAL_BATCH_SIZE}"
        "expname=${expname}"
        logging.wandb.mode=offline
    )

    if [[ "${FAST_DEV_RUN}" == "1" ]]; then
        cmd+=(train.pl_trainer.fast_dev_run=true)
    fi

    echo
    echo "------------------------------------------------------------"
    echo "Training ${expname}: ${prop}"
    echo "Start time=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    "${cmd[@]}" 2>&1 | tee "${log_file}"

    echo "Completed ${expname} at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo
echo "All requested MP-20 predictor trainings completed."
echo "Outputs: ${HYDRA_JOBS}/singlerun/YYYY-MM-DD/*-mp20_predictor_*"
echo "Logs: ${LOG_DIR}"
