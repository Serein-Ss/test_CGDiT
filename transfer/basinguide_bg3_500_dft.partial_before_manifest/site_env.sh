#!/bin/bash
# EDIT for the destination server. Python should run in its cgdit environment.
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate cgdit
# module load your-compiler your-mpi your-vasp
# export PYTHON_BIN=/path/to/envs/cgdit/bin/python
# Default launch: mpirun -np $SLURM_NTASKS vasp_std
# If required by your site: export VASP_COMMAND="srun vasp_std"
export PYTHON_BIN="${PYTHON_BIN:-python}"
