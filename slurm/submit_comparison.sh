#!/bin/bash
# Runs every config in a list as its own job, one after another.
#
#   bash slurm/submit_comparison.sh                      # slurm/comparison.txt
#   bash slurm/submit_comparison.sh my_list.txt
#
# Only the first job is submitted here; each job queues the next one when it
# starts (see slurm/run_one.sbatch). To stop the chain, scancel both the
# running job and the pending one.
set -eo pipefail
cd "$(dirname "$0")/.."

LIST="${1:-slurm/comparison.txt}"
mapfile -t CONFIGS < <(sed 's/\r$//' "$LIST" | grep -v '^[[:space:]]*#' | grep -v '^[[:space:]]*$')
if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "No configs in ${LIST}" >&2
    exit 1
fi

mkdir -p logs
export CONFIG="${CONFIGS[0]}"
export QUEUE="${CONFIGS[*]:1}"
export RUN_ROOT="${RUN_ROOT:-outputs/comparison}"

echo "Submitting ${#CONFIGS[@]} configs in order: ${CONFIGS[*]}"
sbatch --export=ALL --job-name="$(basename "$CONFIG" .yaml)" slurm/run_one.sbatch
