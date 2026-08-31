#!/usr/bin/env bash

set -Eeuo pipefail

ARM="${1:?Usage: run_rl_fe_pilot_arm_conda.sh P0|P2|G0|G2}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set by the parallel launcher}"
LOG_ROOT="${LOG_ROOT:?LOG_ROOT must be set by the parallel launcher}"
UPDATES="${UPDATES:-8}"
ASSIGNED_GPU="${ASSIGNED_GPU:-unknown}"

case "${ARM}" in
    P0)
        CONFIG="conf/rl/experiments/ppo_fe_pilot.yaml"
        RUN_NAME="ppo_fe_seed42"
        PIRL_ARGS=(--no-pirl)
        ;;
    P2)
        CONFIG="conf/rl/experiments/ppo_fe_pilot.yaml"
        RUN_NAME="ppo_fe_seed42_pirl"
        PIRL_ARGS=(--pirl)
        ;;
    G0)
        CONFIG="conf/rl/experiments/grpo_fe_pilot.yaml"
        RUN_NAME="grpo_fe_seed42"
        PIRL_ARGS=(--no-pirl)
        ;;
    G2)
        CONFIG="conf/rl/experiments/grpo_fe_pilot.yaml"
        RUN_NAME="grpo_fe_seed42_pirl"
        PIRL_ARGS=(--pirl)
        ;;
    *)
        echo "Unknown arm: ${ARM}. Expected P0, P2, G0 or G2." >&2
        exit 2
        ;;
esac

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the existing Conda environment before running this worker." >&2
    exit 1
fi
if ! [[ "${UPDATES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "UPDATES must be a positive integer." >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
RESULT_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
RESULTS_DIR="${RESULT_ROOT}/train_results"
MODEL_DIR="${RESULT_ROOT}/model"
METRICS="${RESULTS_DIR}/metrics.jsonl"
FINAL_EVALUATION="${RESULTS_DIR}/final_paired_evaluation.json"
MANIFEST="${RESULTS_DIR}/arm_manifest.txt"
mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}"

export WANDB_MODE=offline
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

record_failure() {
    local status=$?
    {
        echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "status=failed"
        echo "exit_code=${status}"
    } >> "${MANIFEST}"
    exit "${status}"
}
trap record_failure ERR

{
    echo "arm=${ARM}"
    echo "run_id=${RUN_ID:-unknown}"
    echo "status=running"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "assigned_gpu=${ASSIGNED_GPU}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "conda_prefix=${CONDA_PREFIX}"
    echo "config=${CONFIG}"
    echo "updates=${UPDATES}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    python -c "import torch; assert torch.cuda.is_available(); print('visible_gpu=' + torch.cuda.get_device_name(0))"
    sha256sum \
        cgdit/rl/objectives.py \
        cgdit/rl/paired_probe.py \
        cgdit/rl/policy_improvement.py \
        scripts/cli/training/train_crystal_rl.py \
        scripts/cli/training/check_rl_reproducibility.py \
        "${CONFIG}" \
        conf/rl/components/rewards/contracts/mp20.yaml \
        conf/rl/components/rewards/registries/mp20.yaml
} > "${MANIFEST}"

python -m scripts.cli.training.train_crystal_rl     --train-config "${CONFIG}"     "${PIRL_ARGS[@]}"     --updates "${UPDATES}"     --run-kind train     --output-root "${OUTPUT_ROOT}"

python - "${METRICS}" "${MODEL_DIR}" "${UPDATES}" "${ARM}" "${CONFIG}" "${FINAL_EVALUATION}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

metrics_path = Path(sys.argv[1])
model_dir = Path(sys.argv[2])
expected_updates = int(sys.argv[3])
arm = sys.argv[4]
config = OmegaConf.load(sys.argv[5])
final_evaluation_path = Path(sys.argv[6])
expect_pirl = arm in {"P2", "G2"}
expected_policy_epochs = int(config.policy_epochs)

records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
expected_steps = list(range(expected_updates))
if [record["step"] for record in records] != expected_steps:
    raise RuntimeError(f"Unexpected training steps in {metrics_path}")
ratio_changed = False
for record in records:
    if record["pirl"] is not expect_pirl:
        raise RuntimeError(f"Unexpected PIRL flag in {metrics_path}")
    if expect_pirl and record["decision"] is None:
        raise RuntimeError(f"Missing PIRL decision in {metrics_path}")
    for key in ("approx_kl", "clip_fraction", "advantage", "policy_epochs"):
        if key not in record:
            raise RuntimeError(f"Missing {key} in {metrics_path}")
    epochs = record["policy_epochs"]
    if len(epochs) != expected_policy_epochs:
        raise RuntimeError(
            f"Expected {expected_policy_epochs} policy epochs in {metrics_path}"
        )
    ratio_changed = ratio_changed or any(
        abs(float(epoch["ratio_mean"]) - 1.0) > 1.0e-8
        for epoch in epochs[1:]
    )

if expected_policy_epochs > 1 and not ratio_changed:
    raise RuntimeError(
        f"Policy ratio never changed after the first epoch in {metrics_path}"
    )

checkpoints = sorted(model_dir.glob("*.ckpt"))
if len(checkpoints) != expected_updates:
    raise RuntimeError(
        f"Expected {expected_updates} checkpoints in {model_dir}, found {len(checkpoints)}"
    )
if not final_evaluation_path.is_file():
    raise RuntimeError(f"Missing final paired evaluation: {final_evaluation_path}")
final_evaluation = json.loads(final_evaluation_path.read_text())
for key in ("baseline", "final", "decision"):
    if key not in final_evaluation:
        raise RuntimeError(f"Missing {key} in {final_evaluation_path}")
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "zrs-gen-rl ${ARM} completed."
echo "Results: ${RESULT_ROOT}"
