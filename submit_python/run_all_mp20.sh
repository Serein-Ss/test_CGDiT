#!/bin/bash
# =============================================================================
# 一键运行 mp_20 全部 8 个实验（顺序执行）
# 用法：bash run_all_mp20.sh
# 中断恢复：bash run_all_mp20.sh  （会跳过已有 checkpoint 的实验）
# =============================================================================

set -e  # 任意一个实验失败则立即停止

# ── 路径配置（脚本在 submit_python/ 下，PROJECT_ROOT 自动定位到上一层）────────
export PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── 日志目录 ──────────────────────────────────────────────────────────────────
LOG_DIR="${PROJECT_ROOT}/logs/mp20_runs"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "  CGDiT mp_20 全量实验"
echo "  PROJECT_ROOT = ${PROJECT_ROOT}"
echo "  日志目录     = ${LOG_DIR}"
echo "  开始时间     = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ── 实验列表（名称 : model配置名）────────────────────────────────────────────
declare -a EXPNAMES=(
    "mp20_base"
    "mp20_fe"
    "mp20_bg"
    "mp20_eh"
    "mp20_fe_bg"
    "mp20_fe_eh"
    "mp20_bg_eh"
    "mp20_fe_bg_eh"
)

declare -a MODELS=(
    "exp_mp20_base"
    "exp_mp20_fe"
    "exp_mp20_bg"
    "exp_mp20_eh"
    "exp_mp20_fe_bg"
    "exp_mp20_fe_eh"
    "exp_mp20_bg_eh"
    "exp_mp20_fe_bg_eh"
)

TOTAL=${#EXPNAMES[@]}

for i in "${!EXPNAMES[@]}"; do
    EXPNAME="${EXPNAMES[$i]}"
    MODEL="${MODELS[$i]}"
    IDX=$((i + 1))

    echo ""
    echo "--------------------------------------------------------"
    echo "  实验 ${IDX}/${TOTAL}: ${EXPNAME}"
    echo "  模型配置: conf/model/${MODEL}.yaml  (修复后相对路径)"
    echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------------------"

    LOG_FILE="${LOG_DIR}/${EXPNAME}.log"

    # 执行训练，stdout+stderr 同时输出到终端和日志文件
    python "${PROJECT_ROOT}/cgdit/run.py" \
        data=mp_20 \
        model="${MODEL}" \
        expname="${EXPNAME}" \
        2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "  [完成] ${EXPNAME}  耗时见日志: ${LOG_FILE}"
    echo "  完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "========================================================"
echo "  全部 ${TOTAL} 个实验完成！"
echo "  结束时间 = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
