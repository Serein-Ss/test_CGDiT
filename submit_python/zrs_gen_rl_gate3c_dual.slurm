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
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
RUN_STAMP="$(date '+%Y-%m-%d/%H-%M-%S')"
RUN_ROOT="${PROJECT_ROOT}/output/rl_finetune/${RUN_STAMP}-test-zrs-gen-rl-gate3c"
AUDIT_ROOT="${PROJECT_ROOT}/output/rl_diagnostics/${SLURM_JOB_ID}-gate3c"
MANIFEST="${AUDIT_ROOT}/manifest.txt"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}" "${AUDIT_ROOT}" logs/rl
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
    echo "run_root=${RUN_ROOT}"
    srun "${PYTHON}" -c "import torch; print('gpu=' + torch.cuda.get_device_name(0)); print('torch=' + torch.__version__); print('cuda=' + str(torch.version.cuda))"
    sha256sum \
        cgdit/rl/policy_improvement.py \
        scripts/cli/training/train_crystal_rl.py \
        conf/rl/reward_closed_loop_mp20.yaml \
        conf/rl/ppo_fe_pilot.yaml \
        conf/rl/grpo_fe_pilot.yaml
} > "${MANIFEST}"

srun "${PYTHON}" -m pytest -q \
    tests/test_rl_objectives.py \
    tests/test_rl_training_cli.py \
    > "${AUDIT_ROOT}/pytest.log" 2>&1

for algorithm in ppo grpo; do
    srun "${PYTHON}" -m scripts.cli.training.train_crystal_rl \
        --train-config "conf/rl/${algorithm}_fe_pilot.yaml" \
        --pirl \
        --updates 2 \
        --run-kind test \
        --output-root "${RUN_ROOT}" \
        --no-final-paired-evaluation \
        > "${AUDIT_ROOT}/${algorithm}.log" 2>&1
done

"${PYTHON}" - "${RUN_ROOT}" "${AUDIT_ROOT}/gate3c_summary.json" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
summary = {}
for algorithm in ("ppo", "grpo"):
    metrics = (
        run_root
        / f"{algorithm}_fe_seed42_pirl"
        / "test_results"
        / "metrics.jsonl"
    )
    records = [
        json.loads(line)
        for line in metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 2:
        raise RuntimeError(f"Expected two {algorithm} updates, got {len(records)}")
    for record in records:
        decision = record["decision"]
        if decision["action"] not in {"accept", "reject"}:
            raise RuntimeError("Unverified attenuation reached checkpoint output")
        initial = decision["initial"]
        if "local" not in initial or "absolute" not in initial:
            raise RuntimeError("Dual-baseline evidence is missing")
        if initial["action"] == "attenuate" and decision["recheck"] is None:
            raise RuntimeError("Attenuated candidate was not re-evaluated")
        safety = decision["safety_audit"]
        if safety["evaluated"] != ["validity"]:
            raise RuntimeError("Unexpected online safety metric declaration")
        if "stability" not in safety["missing"]:
            raise RuntimeError("Missing stability constraint was not disclosed")
        if safety["formal_complete"]:
            raise RuntimeError("Partial online safety gate was marked formal-complete")
        if safety["verification_status"] != "diagnostic_only":
            raise RuntimeError("Partial gate was not labeled diagnostic-only")
        if decision["verified_checkpoint_update"]:
            raise RuntimeError("Diagnostic checkpoint was marked verified")
        if not Path(record["checkpoint"]).is_file():
            raise RuntimeError("Recorded checkpoint does not exist")
    summary[algorithm] = {
        "actions": [record["decision"]["action"] for record in records],
        "initial_actions": [
            record["decision"]["initial"]["action"] for record in records
        ],
        "rechecks": [
            record["decision"]["recheck"] is not None for record in records
        ],
    }
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "Gate 3c dual-baseline validation completed: ${RUN_ROOT}"
echo "Audit: ${AUDIT_ROOT}"
