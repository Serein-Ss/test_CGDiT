#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/tests/reinforcement_learning/slurm/zrs-gen-rl-pipo-microbatch-smoke-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/tests/reinforcement_learning/slurm/zrs-gen-rl-pipo-microbatch-smoke-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/submit_python/rl_output_layout.sh"
RUN_ID="$(rl_resolve_run_id)"
OUTPUT_ROOT="$(rl_run_root "${PROJECT_ROOT}" test/reinforcement_learning pipo-microbatch-smoke "${RUN_ID}")"
DIAGNOSTICS_ROOT="${OUTPUT_ROOT}/diagnostics"
LOG_ROOT="${PROJECT_ROOT}/logs/tests/reinforcement_learning/${RUN_ID}-pipo-microbatch-smoke"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${DIAGNOSTICS_ROOT}" "${LOG_ROOT}" logs/tests/reinforcement_learning/slurm

srun "${PYTHON}" -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/experiments/grpo_fe_seed42_pipo_probe32.yaml \
    --pipo \
    --no-pirl \
    --no-audit-policy-updates \
    --updates 1 \
    --checkpoint-every 1 \
    --no-final-paired-evaluation \
    --output-root "${OUTPUT_ROOT}" \
    > "${LOG_ROOT}/training.log" 2>&1

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "grpo_fe_seed42_pipo/train_results/metrics.jsonl"
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if len(records) != 1:
    raise RuntimeError(f"Expected one smoke update, got {len(records)}")
if any(epoch["microbatches"] != 4 for epoch in records[0]["policy_epochs"]):
    raise RuntimeError("Expected four prompt microbatches per policy epoch")
print(json.dumps({
    "status": "ready",
    "gradient_norm": records[0]["gradient_norm"],
    "microbatches": records[0]["policy_epochs"][0]["microbatches"],
    "checkpoint": records[0]["checkpoint"],
}))
PY_VERIFY
