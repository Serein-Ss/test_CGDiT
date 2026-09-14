#!/bin/bash
#SBATCH --job-name=bg3-dft
#SBATCH --account=xlzougrp
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-499

set -eo pipefail
source /share/home/xlzou/Anaconda3/etc/profile.d/conda.sh
conda activate cgdit
module load py/vasp/intel-ips2018-u4/vasp6.3.1/unchange
set -u
export PROJECT_ROOT=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_CGDiT
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$PROJECT_ROOT"
: "${CAMPAIGN:?}"
python -m scripts.basinguide_diagnostic dft --campaign "$CAMPAIGN" \
    --index "$SLURM_ARRAY_TASK_ID"
