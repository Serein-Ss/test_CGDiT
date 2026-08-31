#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/reinforcement_learning/slurm/zrs-gen-rl-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/reinforcement_learning/slurm/zrs-gen-rl-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="$(rl_resolve_run_id)"
RUN_ROOT="$(rl_run_root "${PROJECT_ROOT}" test/reinforcement_learning gate3b-scale "${RUN_ID}")"
DIAGNOSTICS_ROOT="${RUN_ROOT}/diagnostics"
LOG_ROOT="${PROJECT_ROOT}/logs/tests/reinforcement_learning/${RUN_ID}-gate3b-scale"
MANIFEST="${DIAGNOSTICS_ROOT}/manifest.txt"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
export PROJECT_ROOT
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${PROJECT_ROOT}"
mkdir -p "${DIAGNOSTICS_ROOT}" "${LOG_ROOT}" logs/tests/reinforcement_learning/slurm
test -x "${PYTHON}"

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
    echo "job_id=${SLURM_JOB_ID}"
    echo "job_name=${SLURM_JOB_NAME}"
    echo "status=running"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "node=${SLURMD_NODENAME}"
    echo "partition=${SLURM_JOB_PARTITION}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "python=${PYTHON}"
    srun "${PYTHON}" -c "import torch; print('gpu=' + torch.cuda.get_device_name(0)); print('torch=' + torch.__version__); print('cuda=' + str(torch.version.cuda))"
    sha256sum \
        scripts/cli/training/train_crystal_rl.py \
        scripts/cli/training/check_rl_reproducibility.py \
        scripts/cli/training/diagnose_rl_update_scale.py \
        conf/rl/experiments/ppo_fe_pilot.yaml \
        conf/rl/experiments/grpo_fe_pilot.yaml \
        conf/rl/components/rewards/contracts/mp20.yaml \
        conf/rl/components/rewards/registries/mp20.yaml
} > "${MANIFEST}"

srun "${PYTHON}" -m pytest -q \
    tests/test_rl_objectives.py \
    tests/test_rl_training_cli.py \
    tests/test_rl_reproducibility_cli.py \
    tests/test_rl_update_scale_diagnostic.py \
    > "${LOG_ROOT}/pytest.log" 2>&1

srun "${PYTHON}" -m \
    scripts.cli.training.check_rl_reproducibility \
    --train-config conf/rl/experiments/ppo_fe_pilot.yaml \
    > "${DIAGNOSTICS_ROOT}/reproducibility_1.json"
srun "${PYTHON}" -m \
    scripts.cli.training.check_rl_reproducibility \
    --train-config conf/rl/experiments/ppo_fe_pilot.yaml \
    > "${DIAGNOSTICS_ROOT}/reproducibility_2.json"
if ! cmp -s \
    "${DIAGNOSTICS_ROOT}/reproducibility_1.json" \
    "${DIAGNOSTICS_ROOT}/reproducibility_2.json"
then
    echo "Seeded full trajectories differ across fresh processes." >&2
    exit 3
fi

srun "${PYTHON}" -m \
    scripts.cli.training.diagnose_rl_update_scale \
    --train-config conf/rl/experiments/ppo_fe_pilot.yaml \
    --output "${DIAGNOSTICS_ROOT}/ppo_scale_scan.json"

srun "${PYTHON}" -m \
    scripts.cli.training.diagnose_rl_update_scale \
    --train-config conf/rl/experiments/grpo_fe_pilot.yaml \
    --output "${DIAGNOSTICS_ROOT}/grpo_scale_scan.json"

"${PYTHON}" - \
    "${DIAGNOSTICS_ROOT}/ppo_scale_scan.json" \
    "${DIAGNOSTICS_ROOT}/grpo_scale_scan.json" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

expected_scales = [0.05, 0.1, 0.25, 0.5, 1.0]
for path_value, algorithm in zip(sys.argv[1:], ("ppo", "grpo")):
    path = Path(path_value)
    report = json.loads(path.read_text())
    if report["algorithm"] != algorithm:
        raise RuntimeError(f"Unexpected algorithm in {path}")
    scales = [item["scale"] for item in report["scale_evaluations"]]
    if scales != expected_scales:
        raise RuntimeError(f"Unexpected scales in {path}: {scales}")
    epochs = report["policy_epochs"]
    if len(epochs) != 4:
        raise RuntimeError(f"Expected four policy epochs in {path}")
    if not any(
        abs(float(epoch["ratio_mean"]) - 1.0) > 1.0e-8
        for epoch in epochs[1:]
    ):
        raise RuntimeError(f"Policy ratio did not change in {path}")
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "Gate 3b scale diagnosis completed: ${RUN_ROOT}"
