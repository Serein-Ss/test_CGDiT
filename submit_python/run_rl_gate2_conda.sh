#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
RUN_DATE="${RUN_DATE:-${RUN_ID:0:4}-${RUN_ID:4:2}-${RUN_ID:6:2}}"
RUN_TIME="${RUN_TIME:-${RUN_ID:9:2}-${RUN_ID:11:2}-${RUN_ID:13:2}}"
RUN_NAME="${RUN_TIME}-test-zrs-gen-rl-gate2"
OUTPUT_ROOT="${PROJECT_ROOT}/output/rl_finetune/${RUN_DATE}/${RUN_NAME}"
LOG_ROOT="${PROJECT_ROOT}/logs/remote_rl/${RUN_ID}"
RESULTS_ROOT="${OUTPUT_ROOT}/test_results"
MANIFEST="${RESULTS_ROOT}/run_manifest.txt"
PIRL_ROOT="${OUTPUT_ROOT}/pirl_resume"
STEP_200_ROOT="${OUTPUT_ROOT}/trajectory_200"
FULL_STEP_ROOT="${OUTPUT_ROOT}/trajectory_full"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "The worker requires an active Conda environment." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

export PROJECT_ROOT
export WANDB_MODE=offline
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

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
    echo "run_id=${RUN_ID}"
    echo "test_kind=rl_gate2"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "project_root=${PROJECT_ROOT}"
    echo "conda_prefix=${CONDA_PREFIX}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    python -c "import torch, pytorch_lightning; print('python_torch=' + torch.__version__); print('cuda_runtime=' + str(torch.version.cuda)); print('lightning=' + pytorch_lightning.__version__); print('gpu=' + torch.cuda.get_device_name(0))"
    sha256sum \
        cgdit/pl_modules/diffusion.py \
        cgdit/rl/paired_probe.py \
        cgdit/rl/policy_improvement.py \
        scripts/cli/training/train_crystal_rl.py \
        conf/rl/ppo_fe.yaml \
        conf/rl/grpo_fe.yaml \
        conf/rl/reward_closed_loop_mp20.yaml \
        conf/rl/reward_models_mp20.yaml
} > "${MANIFEST}"

python -m pytest -q \
    tests/test_diffusion_sampling.py \
    tests/test_output_paths.py \
    tests/test_property_predictor.py \
    tests/test_rl_multifidelity.py \
    tests/test_rl_objectives.py \
    tests/test_rl_property_reward.py \
    tests/test_rl_rewards.py \
    tests/test_rl_symmetry_quotient.py \
    tests/test_rl_training_cli.py \
    tests/test_rl_transition_logprob.py \
    2>&1 | tee "${LOG_ROOT}/pytest_rl_gate2.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/ppo_fe.yaml \
    --pirl \
    --preflight-only \
    2>&1 | tee "${LOG_ROOT}/ppo_pirl_preflight.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/grpo_fe.yaml \
    --pirl \
    --preflight-only \
    2>&1 | tee "${LOG_ROOT}/grpo_pirl_preflight.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/ppo_fe.yaml \
    --pirl \
    --run-kind test \
    --output-root "${PIRL_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/ppo_pirl_step0.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/grpo_fe.yaml \
    --pirl \
    --run-kind test \
    --output-root "${PIRL_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/grpo_pirl_step0.log"

PPO_RESUME="${PIRL_ROOT}/ppo_fe_seed42_pirl/model/epoch=0-step=0.ckpt"
GRPO_RESUME="${PIRL_ROOT}/grpo_fe_seed42_pirl/model/epoch=0-step=0.ckpt"
[[ -s "${PPO_RESUME}" ]] || { echo "Missing PPO PIRL checkpoint: ${PPO_RESUME}" >&2; exit 1; }
[[ -s "${GRPO_RESUME}" ]] || { echo "Missing GRPO PIRL checkpoint: ${GRPO_RESUME}" >&2; exit 1; }

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/ppo_fe.yaml \
    --pirl \
    --run-kind test \
    --resume "${PPO_RESUME}" \
    --output-root "${PIRL_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/ppo_pirl_resume_step1.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/grpo_fe.yaml \
    --pirl \
    --run-kind test \
    --resume "${GRPO_RESUME}" \
    --output-root "${PIRL_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/grpo_pirl_resume_step1.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/grpo_fe.yaml \
    --no-pirl \
    --run-kind test \
    --num-prompts 1 \
    --group-size 2 \
    --short-diff-ratio 0.2 \
    --output-root "${STEP_200_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/grpo_trajectory_200.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/grpo_fe.yaml \
    --no-pirl \
    --run-kind test \
    --num-prompts 1 \
    --group-size 2 \
    --short-diff-ratio 1.0 \
    --output-root "${FULL_STEP_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/grpo_trajectory_full.log"

python - "${OUTPUT_ROOT}" "${RUN_ID}" <<'PY_GATE2'
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1]).resolve()
project_root = Path.cwd().resolve()


def relative(path):
    return str(path.resolve().relative_to(project_root))


def load_metrics(path, expected_steps, expect_pirl):
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if [record["step"] for record in records] != expected_steps:
        raise RuntimeError(f"Unexpected steps in {path}: {[r['step'] for r in records]}")
    for record in records:
        if record["pirl"] is not expect_pirl:
            raise RuntimeError(f"Unexpected PIRL flag in {path}")
        if expect_pirl and record["decision"] is None:
            raise RuntimeError(f"Missing PIRL decision in {path}")
        if not math.isfinite(record["gradient_norm"]) or record["gradient_norm"] <= 0:
            raise RuntimeError(f"Invalid gradient norm in {path}")
    return records


sections = {
    "pirl_resume": {
        "ppo_fe_seed42_pirl": ([0, 1], True),
        "grpo_fe_seed42_pirl": ([0, 1], True),
    },
    "trajectory_200": {"grpo_fe_seed42": ([0], False)},
    "trajectory_full": {"grpo_fe_seed42": ([0], False)},
}
artifacts = {}
for section, algorithms in sections.items():
    artifacts[section] = {}
    for algorithm, (steps, expect_pirl) in algorithms.items():
        algorithm_root = run_root / section / algorithm
        checkpoints = sorted((algorithm_root / "model").glob("*.ckpt"))
        if len(checkpoints) != len(steps):
            raise RuntimeError(f"Unexpected checkpoint count in {algorithm_root}")
        metrics = algorithm_root / "test_results/metrics.jsonl"
        load_metrics(metrics, steps, expect_pirl)
        artifacts[section][algorithm] = {
            "checkpoints": [relative(path) for path in checkpoints],
            "model_hparams": relative(algorithm_root / "model/hparams.yaml"),
            "test_metrics": relative(metrics),
        }

manifest = {
    "run_type": "test",
    "test_kind": "rl_gate2",
    "original_run_id": sys.argv[2],
    "run_directory": relative(run_root),
    "artifacts": artifacts,
    "environment_manifest": relative(run_root / "test_results/run_manifest.txt"),
}
(run_root / "test_results/source_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY_GATE2

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

echo "zrs-gen-rl Gate 2 completed."
echo "Results: ${OUTPUT_ROOT}"
echo "Logs: ${LOG_ROOT}"
