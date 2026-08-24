#!/usr/bin/env bash
#SBATCH --job-name=zrs-gen-rl
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-crystalpirl-seed42-complete-analysis-%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT/logs/rl/zrs-gen-rl-crystalpirl-seed42-complete-analysis-%j.err

set -Eeuo pipefail

: "${PIPO_FE_BG_JOB:?PIPO_FE_BG_JOB is required}"
: "${PIPO_JOINT_JOB:?PIPO_JOINT_JOB is required}"
: "${PIPO_GENERATION_JOB:?PIPO_GENERATION_JOB is required}"
: "${CRYSTALPIRL_TRAIN_JOB:?CRYSTALPIRL_TRAIN_JOB is required}"
: "${CRYSTALPIRL_GENERATION_JOB:?CRYSTALPIRL_GENERATION_JOB is required}"

PROJECT_ROOT="/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PIPO_FE_BG_ROOT="${PROJECT_ROOT}/output/rl_finetune/seed42_grpo_pipo_probe32_resume60/${PIPO_FE_BG_JOB}"
PIPO_JOINT_ROOT="${PROJECT_ROOT}/output/rl_finetune/seed42_grpo_pipo_joint_max60/${PIPO_JOINT_JOB}"
CRYSTALPIRL_ROOT="${PROJECT_ROOT}/output/rl_finetune/seed42_crystalpirl_block10_200step/${CRYSTALPIRL_TRAIN_JOB}"
ANALYSIS_ROOT="${PROJECT_ROOT}/output/rl_analysis/seed42_crystalpirl_complete/${SLURM_JOB_ID}"
SOURCE_DIR="${ANALYSIS_ROOT}/source_data"
ASSET_DIR="${PROJECT_ROOT}/assets/crystalpirl_seed42_complete"
MANIFEST="${ANALYSIS_ROOT}/analysis_manifest.txt"
VALIDATOR="/share/home/xlzou/WORKSPACE/rszhong/.codex/skills/nature-figure/scripts/validate_figure.py"
PLOT_SOURCE="scripts/result_figures/crystalpirl_seed42_complete/plot_figures.py"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "${PROJECT_ROOT}"
mkdir -p "${ANALYSIS_ROOT}" logs/rl
test -x "${PYTHON}"
test -f "${VALIDATOR}"
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
    echo "main_method=GRPO+CrystalPIRL_block10"
    echo "comparator=GRPO+PIPO"
    echo "pipo_fe_bg_job=${PIPO_FE_BG_JOB}"
    echo "pipo_joint_job=${PIPO_JOINT_JOB}"
    echo "pipo_generation_job=${PIPO_GENERATION_JOB}"
    echo "crystalpirl_train_job=${CRYSTALPIRL_TRAIN_JOB}"
    echo "crystalpirl_generation_job=${CRYSTALPIRL_GENERATION_JOB}"
    echo "source_dir=${SOURCE_DIR}"
    echo "asset_dir=${ASSET_DIR}"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    sha256sum \
        scripts/result_figures/crystalpirl_seed42_complete/build_source_data.py \
        scripts/result_figures/crystalpirl_seed42_complete/audit_conclusions.py \
        "${PLOT_SOURCE}"
} > "${MANIFEST}"

"${PYTHON}" "${VALIDATOR}" "${PLOT_SOURCE}" --json \
    > "${ANALYSIS_ROOT}/nature_figure_source_preflight.json"

"${PYTHON}" - "${ANALYSIS_ROOT}/nature_figure_source_preflight.json" <<'PY_PREFLIGHT'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
failures = {
    item["check_id"]
    for item in report["findings"]
    if item["level"] == "FAIL"
}
warnings = {
    item["check_id"]
    for item in report["findings"]
    if item["level"] == "WARN"
}
if failures != {"EXPORT-VECTOR"}:
    raise RuntimeError(f"Unexpected Nature figure source failures: {sorted(failures)}")
if warnings - {"EXPORT-RASTER"}:
    raise RuntimeError(f"Unexpected Nature figure source warnings: {sorted(warnings)}")
PY_PREFLIGHT

"${PYTHON}" scripts/result_figures/crystalpirl_seed42_complete/build_source_data.py \
    --pipo-fe-bg-root "${PIPO_FE_BG_ROOT}" \
    --pipo-joint-root "${PIPO_JOINT_ROOT}" \
    --crystalpirl-root "${CRYSTALPIRL_ROOT}" \
    --output-dir "${SOURCE_DIR}"

"${PYTHON}" scripts/result_figures/crystalpirl_seed42_complete/audit_conclusions.py \
    --source-dir "${SOURCE_DIR}"

"${PYTHON}" "${PLOT_SOURCE}" \
    --source-dir "${SOURCE_DIR}" \
    --asset-dir "${ASSET_DIR}"

"${PYTHON}" - "${ASSET_DIR}" <<'PY_VERIFY'
import json
import sys
from pathlib import Path

from PIL import Image

asset_dir = Path(sys.argv[1])
expected = {
    "fig1_crystalpirl_method.png",
    "fig2_verified_policy_improvement.png",
    "fig3_fe_bg_control.png",
    "fig4_quality_search_space.png",
}
observed = {path.name for path in asset_dir.glob("*.png")}
if observed != expected:
    raise RuntimeError(f"Unexpected PNG set: {sorted(observed)}")
for path in asset_dir.iterdir():
    if path.suffix.lower() in {".pdf", ".svg", ".tif", ".tiff"}:
        raise RuntimeError(f"PNG-only contract violated by {path}")
for path in asset_dir.glob("*.png"):
    if path.stat().st_size == 0:
        raise RuntimeError(f"Empty figure: {path}")
    with Image.open(path) as image:
        if image.width < 3000 or image.height < 1500:
            raise RuntimeError(f"Insufficient rendered resolution for {path}: {image.size}")
audit = json.loads((asset_dir / "source_data/conclusion_audit.json").read_text())
if not audit["plot_authorized"]:
    raise RuntimeError("Copied conclusion audit does not authorize plotting")
qa = json.loads((asset_dir / "figure_qa.json").read_text())
if qa["seed"] != 42 or "CrystalPIRL" not in qa["main_method"]:
    raise RuntimeError("Figure QA method/seed contract is incorrect")
PY_VERIFY

trap - ERR
{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "status=complete"
    echo "source_data=${SOURCE_DIR}"
    echo "assets=${ASSET_DIR}"
    echo "png_only_exemption=vector_and_tiff_deferred_by_user_request"
    echo "rendered_visual_audit=pending_manual_panel_inspection"
} >> "${MANIFEST}"

echo "Complete seed=42 CrystalPIRL Source Data and PNG figures created: ${ASSET_DIR}"
