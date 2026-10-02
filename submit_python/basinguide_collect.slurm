#!/bin/bash
#SBATCH --job-name=bg3-summary
#SBATCH --account=xlzougrp
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -eo pipefail
source /share/home/xlzou/Anaconda3/etc/profile.d/conda.sh
conda activate cgdit
set -u
export PROJECT_ROOT=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
python -m scripts.basinguide_diagnostic collect --campaign "$CAMPAIGN"
