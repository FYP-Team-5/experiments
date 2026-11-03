#!/bin/bash
# Submits one experiment config as its own job.
#
#   bash slurm/submit.sh cmp_grading_base
#   bash slurm/submit.sh configs/experiments/cmp_grading_base.yaml
#
# The CS480G3 account allows 2 submitted jobs (1 running), so submit at most
# two configs, and the next one after one of them finishes (check with myqueue).
# Logs: logs/<config name>_<job id>.out and .err
set -eo pipefail
cd "$(dirname "$0")/.."

if [ $# -ne 1 ]; then
    echo "Usage: bash slurm/submit.sh <config name or path>" >&2
    exit 1
fi
CONFIG="$1"
[ -f "$CONFIG" ] || CONFIG="configs/experiments/${1%.yaml}.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "No such config: $1" >&2
    exit 1
fi

mkdir -p logs
sbatch --job-name="$(basename "$CONFIG" .yaml)" slurm/run_one.sbatch "$CONFIG"
