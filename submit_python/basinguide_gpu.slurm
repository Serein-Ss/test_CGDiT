#!/bin/bash
#SBATCH --job-name=bg3-generate
#SBATCH --account=xlzougrp
#SBATCH --partition=rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

set -eo pipefail
source /share/home/xlzou/Anaconda3/etc/profile.d/conda.sh
conda activate cgdit
set -u
export PROJECT_ROOT=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$PROJECT_ROOT"
: "${CAMPAIGN:?}"
if [[ "${ACTION:-generate}" == generate ]]; then
    python -m scripts.basinguide_diagnostic generate --campaign "$CAMPAIGN"
else
    python -m scripts.basinguide_diagnostic mlip --campaign "$CAMPAIGN" \
        --checkpoint "$CAMPAIGN/models/checkpoint_sevennet_omni_i12.pth"
fi
