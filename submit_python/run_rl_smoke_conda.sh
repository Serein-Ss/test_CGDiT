#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
OUTPUT_ROOT="${PROJECT_ROOT}/output/remote_rl_smoke/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/remote_rl/${RUN_ID}"
MANIFEST="${OUTPUT_ROOT}/run_manifest.txt"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "The worker requires an active Conda environment." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

export PROJECT_ROOT
export WANDB_MODE=offline
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

{
    echo "run_id=${RUN_ID}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "project_root=${PROJECT_ROOT}"
    echo "conda_prefix=${CONDA_PREFIX}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    python -c "import torch, pytorch_lightning; print('python_torch=' + torch.__version__); print('cuda_runtime=' + str(torch.version.cuda)); print('lightning=' + pytorch_lightning.__version__); print('gpu=' + torch.cuda.get_device_name(0))"
    sha256sum \
        cgdit/pl_modules/diffusion.py \
        cgdit/generation/general.py \
        scripts/cli/training/train_crystal_rl.py \
        conf/rl/ppo_fe.yaml \
        conf/rl/grpo_fe.yaml \
        conf/rl/reward_closed_loop_mp20.yaml \
        conf/rl/reward_models_mp20.yaml
} > "${MANIFEST}"

python -m pytest -q \
    tests/test_diffusion_sampling.py \
    tests/test_property_predictor.py \
    tests/test_rl_multifidelity.py \
    tests/test_rl_objectives.py \
    tests/test_rl_property_reward.py \
    tests/test_rl_rewards.py \
    tests/test_rl_symmetry_quotient.py \
    tests/test_rl_training_cli.py \
    tests/test_rl_transition_logprob.py \
    2>&1 | tee "${LOG_ROOT}/pytest_rl.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/ppo_fe.yaml \
    --preflight-only \
    2>&1 | tee "${LOG_ROOT}/ppo_fe_preflight.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/grpo_fe.yaml \
    --preflight-only \
    2>&1 | tee "${LOG_ROOT}/grpo_fe_preflight.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/ppo_fe.yaml \
    --no-pirl \
    --output-root "${OUTPUT_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/ppo_fe.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/grpo_fe.yaml \
    --no-pirl \
    --output-root "${OUTPUT_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/grpo_fe.log"

{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "zrs-gen-rl smoke completed."
echo "Results: ${OUTPUT_ROOT}"
echo "Logs: ${LOG_ROOT}"
