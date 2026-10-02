#!/bin/bash
set -eo pipefail
: "${BASINGUIDE_ROOT:?}"
# Configure site-specific module/conda commands in site_env.sh.
source "$BASINGUIDE_ROOT/site_env.sh"
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$BASINGUIDE_ROOT"
if [[ "${BASINGUIDE_ACTION:-dft}" == collect ]]; then
    "${PYTHON_BIN:-python}" run_dft.py --collect
else
    "${PYTHON_BIN:-python}" run_dft.py --index "$SLURM_ARRAY_TASK_ID"
fi
