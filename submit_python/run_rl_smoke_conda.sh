#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
RUN_DATE="${RUN_DATE:-${RUN_ID:0:4}-${RUN_ID:4:2}-${RUN_ID:6:2}}"
RUN_TIME="${RUN_TIME:-${RUN_ID:9:2}-${RUN_ID:11:2}-${RUN_ID:13:2}}"
RUN_NAME="${RUN_TIME}-test-zrs-gen-rl-smoke"
OUTPUT_ROOT="${PROJECT_ROOT}/output/test/reinforcement_learning/${RUN_DATE}/${RUN_NAME}"
LOG_ROOT="${PROJECT_ROOT}/logs/reinforcement_learning/local_runs/${RUN_ID}"
TEST_LOG_ROOT="${PROJECT_ROOT}/logs/tests/reinforcement_learning/${RUN_ID}"
RESULTS_ROOT="${OUTPUT_ROOT}/test_results"
MANIFEST="${RESULTS_ROOT}/run_manifest.txt"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "The worker requires an active Conda environment." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}" "${TEST_LOG_ROOT}"

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
        conf/rl/experiments/ppo_fe.yaml \
        conf/rl/experiments/grpo_fe.yaml \
        conf/rl/components/rewards/contracts/mp20.yaml \
        conf/rl/components/rewards/registries/mp20.yaml
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
    2>&1 | tee "${TEST_LOG_ROOT}/pytest_rl.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/experiments/ppo_fe.yaml \
    --preflight-only \
    2>&1 | tee "${TEST_LOG_ROOT}/ppo_fe_preflight.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/experiments/grpo_fe.yaml \
    --preflight-only \
    2>&1 | tee "${TEST_LOG_ROOT}/grpo_fe_preflight.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/experiments/ppo_fe.yaml \
    --no-pirl \
    --run-kind test \
    --output-root "${OUTPUT_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/ppo_fe.log"

python -m scripts.cli.training.train_crystal_rl \
    --train-config conf/rl/experiments/grpo_fe.yaml \
    --no-pirl \
    --run-kind test \
    --output-root "${OUTPUT_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/grpo_fe.log"

{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
} >> "${MANIFEST}"

python - "${OUTPUT_ROOT}" "${RUN_ID}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1]).resolve()
project_root = Path.cwd().resolve()

def relative(path):
    return str(path.resolve().relative_to(project_root))

algorithms = {}
for algorithm_root in sorted(
    path for path in run_root.iterdir() if (path / "model").is_dir()
):
    checkpoints = sorted((algorithm_root / "model").glob("*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one RL checkpoint in {algorithm_root / 'model'}")
    algorithms[algorithm_root.name] = {
        "checkpoint": relative(checkpoints[0]),
        "model_hparams": relative(algorithm_root / "model/hparams.yaml"),
        "test_metrics": relative(algorithm_root / "test_results/metrics.jsonl"),
    }
manifest = {
    "run_type": "test",
    "test_kind": "one_update_rl_smoke",
    "original_run_id": sys.argv[2],
    "run_directory": relative(run_root),
    "algorithms": algorithms,
    "environment_manifest": relative(run_root / "test_results/run_manifest.txt"),
}
(run_root / "test_results/source_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

echo "zrs-gen-rl smoke completed."
echo "Results: ${OUTPUT_ROOT}"
echo "Logs: ${LOG_ROOT}"
