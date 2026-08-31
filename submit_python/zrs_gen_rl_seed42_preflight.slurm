#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/tests/reinforcement_learning/slurm/zrs-gen-rl-seed42-preflight-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/tests/reinforcement_learning/slurm/zrs-gen-rl-seed42-preflight-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="$(rl_resolve_run_id)"
AUDIT_ROOT="$(rl_run_root "${PROJECT_ROOT}" test/reinforcement_learning seed42-preflight "${RUN_ID}")"
RUN_ROOT="${AUDIT_ROOT}/runs"
LOG_ROOT="${PROJECT_ROOT}/logs/tests/reinforcement_learning/${RUN_ID}-seed42-preflight"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${PROJECT_ROOT}"
mkdir -p "${AUDIT_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}" logs/tests/reinforcement_learning/slurm
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
    echo "seed=42"
    echo "scope=seed42_300step_full_denoising"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum \
        cgdit/rl/objectives.py \
        cgdit/pl_modules/diffusion.py \
        cgdit/rl/paired_probe.py \
        cgdit/rl/pipo.py \
        cgdit/rl/policy_improvement.py \
        cgdit/rl/property_reward.py \
        scripts/cli/training/train_crystal_rl.py \
        conf/rl/components/rewards/contracts/mp20_seed42_pilot.yaml
} > "${MANIFEST}"

srun "${PYTHON}" -c "import torch; print('gpu=' + torch.cuda.get_device_name(0)); print('torch=' + torch.__version__); print('cuda=' + str(torch.version.cuda))" \
    > "${LOG_ROOT}/gpu.log" 2>&1

srun "${PYTHON}" -m pytest -q \
    tests/test_rl_pipo.py \
    tests/test_rl_objectives.py \
    tests/test_rl_training_cli.py \
    tests/test_rl_property_reward.py \
    tests/test_generated_property_evaluation.py \
    tests/test_generation_conditioning.py \
    > "${LOG_ROOT}/pytest.log" 2>&1

srun "${PYTHON}" -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/experiments/grpo_fe_seed42_300step.yaml \
    --pirl \
    --no-pipo \
    --no-audit-policy-updates \
    --updates 1 \
    --run-kind train \
    --output-root "${RUN_ROOT}" \
    --no-final-paired-evaluation \
    > "${LOG_ROOT}/pirl_gpu_smoke.log" 2>&1

"${PYTHON}" - "${RUN_ROOT}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
pirl_path = root / "grpo_fe_seed42_pirl" / "train_results" / "metrics.jsonl"
pirl = [json.loads(line) for line in pirl_path.read_text().splitlines() if line.strip()]
if len(pirl) != 1:
    raise RuntimeError("CrystalPIRL 5x16 GPU capacity probe did not finish one update")
audit = pirl[0]["decision"]["safety_audit"]
if not audit["formal_complete"] or set(audit["evaluated"]) != {"validity", "stability"}:
    raise RuntimeError("Pilot safety contract was not completely evaluated")
contract = pirl[0]["sampling_contract"]
if contract["training_prompts_per_update"] != 5 or contract["training_trajectories_per_prompt"] != 16:
    raise RuntimeError(f"Unexpected training capacity contract: {contract}")
if contract["probe_prompts"] != 5 or contract["probe_trajectories_per_prompt"] != 1:
    raise RuntimeError(f"Unexpected probe capacity contract: {contract}")
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "seed=42 tests and 5x16 GPU capacity probe complete: ${AUDIT_ROOT}"
