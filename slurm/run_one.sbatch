#!/bin/bash
#SBATCH --job-name=gfv_one
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=project
#SBATCH --account=cs480
#SBATCH --qos=cs480qos

# Runs one experiment config as its own job. Submit it with slurm/submit.sh:
#
#   bash slurm/submit.sh cmp_grading_base
#
# Results go to $RUN_ROOT/<config name>/ (default outputs/comparison). The
# folder is fixed, so resubmitting a config resumes it.

source slurm/env_setup.sh

CONFIG="${1:-${CONFIG:-}}"
if [ -z "$CONFIG" ]; then
    echo "Usage: sbatch slurm/run_one.sbatch configs/experiments/<name>.yaml" >&2
    exit 1
fi
RUN_ROOT="${RUN_ROOT:-outputs/comparison}"
NAME="$(basename "$CONFIG" .yaml)"

echo "Running ${CONFIG} -> ${RUN_ROOT}/${NAME}"
PYTHONPATH=. python3 scripts/run_stage.py --config "$CONFIG" --output_dir "${RUN_ROOT}/${NAME}"
