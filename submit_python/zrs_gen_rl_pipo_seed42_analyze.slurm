#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/metric_evaluation/reinforcement_learning/slurm/zrs-gen-rl-pipo-seed42-analysis-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/metric_evaluation/reinforcement_learning/slurm/zrs-gen-rl-pipo-seed42-analysis-%j.err

set -Eeuo pipefail

: "${TRAIN_FE_BG_JOB:?TRAIN_FE_BG_JOB is required}"
: "${TRAIN_JOINT_JOB:?TRAIN_JOINT_JOB is required}"
: "${GENERATION_JOB:?GENERATION_JOB is required}"

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
FE_BG_ROOT="${FE_BG_ROOT:-${PROJECT_ROOT}/output/reinforcement_learning/2026-08-24/20-57-12-grpo-pipo-probe32-resume60}"
JOINT_ROOT="${JOINT_ROOT:-${PROJECT_ROOT}/output/reinforcement_learning/2026-08-24/20-57-12-grpo-pipo-joint-max60}"
ANALYSIS_ROOT="${FE_BG_ROOT}/analysis"
SOURCE_DIR="${ANALYSIS_ROOT}/source_data"
ASSET_DIR="${PROJECT_ROOT}/assets/crystalpirl_seed42_pipo_200step"
MANIFEST="${ANALYSIS_ROOT}/analysis_manifest.txt"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${PROJECT_ROOT}"
mkdir -p "${ANALYSIS_ROOT}" logs/metric_evaluation/reinforcement_learning/slurm
test -x "${PYTHON}"
test ! -e "${SOURCE_DIR}"
test ! -e "${ASSET_DIR}"

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
    echo "method=GRPO+PIPO"
    echo "method_is_crystalpirl=false"
    echo "training_fe_bg_job=${TRAIN_FE_BG_JOB}"
    echo "training_joint_job=${TRAIN_JOINT_JOB}"
    echo "generation_job=${GENERATION_JOB}"
    echo "source_dir=${SOURCE_DIR}"
    echo "asset_dir=${ASSET_DIR}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum \
        scripts/result_figures/crystalpirl_pipo_seed42/build_source_data.py \
        scripts/result_figures/crystalpirl_pipo_seed42/audit_conclusions.py \
        scripts/result_figures/crystalpirl_pipo_seed42/plot_figures.py
} > "${MANIFEST}"

"${PYTHON}" scripts/result_figures/crystalpirl_pipo_seed42/build_source_data.py \
    --fe-bg-root "${FE_BG_ROOT}" \
    --joint-root "${JOINT_ROOT}" \
    --output-dir "${SOURCE_DIR}"

"${PYTHON}" scripts/result_figures/crystalpirl_pipo_seed42/audit_conclusions.py \
    --source-dir "${SOURCE_DIR}"

"${PYTHON}" scripts/result_figures/crystalpirl_pipo_seed42/plot_figures.py \
    --source-dir "${SOURCE_DIR}" \
    --asset-dir "${ASSET_DIR}"

"${PYTHON}" - "${ASSET_DIR}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

asset_dir = Path(sys.argv[1])
expected = {
    "fig1_crystalpirl_method.png",
    "fig2_pipo_training_dynamics.png",
    "fig3_fe_bg_control.png",
    "fig4_quality_search_space.png",
}
observed = {path.name for path in asset_dir.glob("*.png")}
if observed != expected:
    raise RuntimeError(f"Unexpected PNG set: {sorted(observed)}")
for path in asset_dir.iterdir():
    if path.suffix.lower() in {".pdf", ".svg", ".tif", ".tiff"}:
        raise RuntimeError(f"PNG-only contract violated by {path}")
audit = json.loads(
    (asset_dir / "source_data/conclusion_audit.json").read_text()
)
if not audit["plot_authorized"]:
    raise RuntimeError("Copied audit does not authorize plotting")
if any(path.stat().st_size == 0 for path in asset_dir.glob("*.png")):
    raise RuntimeError("An exported figure is empty")
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
    echo "source_data=${SOURCE_DIR}"
    echo "assets=${ASSET_DIR}"
} >> "${MANIFEST}"

echo "seed=42 GRPO+PIPO Source Data audit and PNG figures complete: ${ASSET_DIR}"
