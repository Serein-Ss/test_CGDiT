#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SEQUENCE_WORKER="submit_python/run_rl_fe_pilot_sequence_conda.sh"
ARM_WORKER="submit_python/run_rl_fe_pilot_arm_conda.sh"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
RUN_DATE="${RUN_DATE:-${RUN_ID:0:4}-${RUN_ID:4:2}-${RUN_ID:6:2}}"
RUN_TIME="${RUN_TIME:-${RUN_ID:9:2}-${RUN_ID:11:2}-${RUN_ID:13:2}}"
RUN_NAME="${RUN_TIME}-pilot-zrs-gen-rl-fe"
OUTPUT_ROOT="${PROJECT_ROOT}/output/rl_finetune/${RUN_DATE}/${RUN_NAME}"
LOG_ROOT="${PROJECT_ROOT}/logs/remote_rl/${RUN_ID}"
MASTER_PID_FILE="${LOG_ROOT}/master.pid"
LAUNCH_MANIFEST="${LOG_ROOT}/launch_manifest.txt"
UPDATES="${UPDATES:-8}"
GPU_ID="${GPU_ID:-0}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the existing Conda environment before running this script." >&2
    exit 1
fi
if ! [[ "${UPDATES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "UPDATES must be a positive integer." >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
bash -n "${SEQUENCE_WORKER}"
bash -n "${ARM_WORKER}"
bash -n submit/run_rl_fe_pilot_sequential_conda.sh
bash -n submit/check_rl_fe_pilot_conda.sh
bash -n submit/pack_rl_fe_pilot_results.sh

gpu_count=$(python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.device_count())")
if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]] || (( GPU_ID >= gpu_count )); then
    echo "Invalid GPU_ID=${GPU_ID}; visible GPU count is ${gpu_count}." >&2
    exit 2
fi
python -c "import torch; print('gpu=', torch.cuda.get_device_name(${GPU_ID}))"

echo "Running RL preflight tests..."
python -m pytest -q \
    tests/test_rl_objectives.py \
    tests/test_rl_training_cli.py \
    tests/test_rl_reproducibility_cli.py \
    > "${LOG_ROOT}/pytest_preflight.log" 2>&1

echo "Checking PPO and GRPO dependencies..."
python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/ppo_fe_pilot.yaml \
    --preflight-only \
    > "${LOG_ROOT}/ppo_preflight.log" 2>&1
python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/grpo_fe_pilot.yaml \
    --preflight-only \
    > "${LOG_ROOT}/grpo_preflight.log" 2>&1

echo "Checking full-trajectory reproducibility in two fresh processes..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m \
    scripts.cli.training.check_rl_reproducibility \
    --train-config conf/rl/ppo_fe_pilot.yaml \
    > "${LOG_ROOT}/reproducibility_1.json"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m \
    scripts.cli.training.check_rl_reproducibility \
    --train-config conf/rl/ppo_fe_pilot.yaml \
    > "${LOG_ROOT}/reproducibility_2.json"
if ! cmp -s "${LOG_ROOT}/reproducibility_1.json" "${LOG_ROOT}/reproducibility_2.json"; then
    echo "Seeded full trajectories are not reproducible; training was not started." >&2
    diff -u "${LOG_ROOT}/reproducibility_1.json" "${LOG_ROOT}/reproducibility_2.json" >&2 || true
    exit 3
fi
echo "Reproducibility check passed."

{
    echo "run_id=${RUN_ID}"
    echo "status=launched"
    echo "execution_mode=single_gpu_sequential"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "project_root=${PROJECT_ROOT}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "log_root=${LOG_ROOT}"
    echo "conda_prefix=${CONDA_PREFIX}"
    echo "updates=${UPDATES}"
    echo "rollouts_per_arm=$((UPDATES * 16))"
    echo "gpu_id=${GPU_ID}"
    echo "arm_order=P0,G0,P2,G2"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
} > "${LAUNCH_MANIFEST}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PROJECT_ROOT="${PROJECT_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" RUN_ID="${RUN_ID}" UPDATES="${UPDATES}" ASSIGNED_GPU="${GPU_ID}" nohup bash "${SEQUENCE_WORKER}" > "${LOG_ROOT}/zrs-gen-rl.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${MASTER_PID_FILE}"

echo "Started single-GPU sequential zrs-gen-rl PID: ${pid}"
echo "Run ID: ${RUN_ID}"
echo "Order: P0 -> G0 -> P2 -> G2"
echo "Results: ${OUTPUT_ROOT}"
echo "Main log: ${LOG_ROOT}/zrs-gen-rl.log"
echo "Check: bash submit/check_rl_fe_pilot_conda.sh ${RUN_ID}"
