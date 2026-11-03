#!/bin/bash
#SBATCH --job-name=gfv_batch_run
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=project
#SBATCH --account=cs480
#SBATCH --qos=cs480qos

# Activate cluster environment and HuggingFace cache paths
source slurm/env_setup.sh

# Override with: sbatch --export=ALL,EXPERIMENT_LIST_FILE=slurm/comparison.txt slurm/run_stage.sbatch
EXPERIMENT_LIST_FILE="${EXPERIMENT_LIST_FILE:-slurm/experiments.txt}"

echo "Submitting batch run reading from: ${EXPERIMENT_LIST_FILE}"

PYTHONPATH=. python3 scripts/run_batch.py --file "${EXPERIMENT_LIST_FILE}"