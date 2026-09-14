#!/bin/bash
set -eo pipefail
BASINGUIDE_ROOT="$(cd "$(dirname "$0")" && pwd)"
export BASINGUIDE_ROOT
source "$BASINGUIDE_ROOT/site_env.sh"
set -u
: "${PARTITION:?Set PARTITION to the destination CPU partition}"
if [[ -e "$BASINGUIDE_ROOT/submitted_jobs.txt" ]]; then
    echo 'Submission record exists; inspect it before submitting again.' >&2
    exit 1
fi
"${PYTHON_BIN:-python}" "$BASINGUIDE_ROOT/run_dft.py" --check
mkdir -p "$BASINGUIDE_ROOT/logs"
account_options=()
if [[ -n "${ACCOUNT:-}" ]]; then account_options=(--account="$ACCOUNT"); fi
array="${ARRAY:-0-499}"
if [[ -n "${MAX_CONCURRENT:-}" ]]; then array="$array%$MAX_CONCURRENT"; fi
job=$(sbatch --parsable "${account_options[@]}" --partition="$PARTITION" \
    --nodes=1 --ntasks="${NTASKS:-20}" --cpus-per-task=1 --mem="${MEMORY:-64G}" \
    --time="${WALLTIME:-7-00:00:00}" --array="$array" --job-name=bg3-dft \
    --output="$BASINGUIDE_ROOT/logs/dft-%A_%a.out" \
    --error="$BASINGUIDE_ROOT/logs/dft-%A_%a.err" --export=ALL \
    "$BASINGUIDE_ROOT/job.slurm")
job="${job%%;*}"
printf 'dft_array=%s\n' "$job" | tee "$BASINGUIDE_ROOT/submitted_jobs.txt"
summary=$(sbatch --parsable "${account_options[@]}" --partition="$PARTITION" \
    --nodes=1 --ntasks=1 --mem=4G --time=01:00:00 --job-name=bg3-summary \
    --dependency="afterany:$job" --export=ALL,BASINGUIDE_ACTION=collect \
    --output="$BASINGUIDE_ROOT/logs/summary-%j.out" \
    --error="$BASINGUIDE_ROOT/logs/summary-%j.err" "$BASINGUIDE_ROOT/job.slurm")
printf 'summary=%s\n' "$summary" | tee -a "$BASINGUIDE_ROOT/submitted_jobs.txt"
