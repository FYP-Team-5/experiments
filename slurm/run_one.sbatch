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

# Runs one experiment config as its own job.
#
#   CONFIG    config to run (required)
#   QUEUE     space-separated configs still to run. The next one is submitted
#             as soon as this job starts, to begin when this job ends, even if
#             it fails. At most two jobs are then queued at once, which is the
#             account limit. slurm/submit_comparison.sh sets this up.
#   RUN_ROOT  results go to $RUN_ROOT/<config name>/ (default outputs/comparison).
#             The folder is fixed, so resubmitting a config resumes it.
#
# Submit a single config by hand:
#   CONFIG=configs/experiments/cmp_grading_qlora.yaml sbatch --export=ALL slurm/run_one.sbatch

source slurm/env_setup.sh

if [ -z "${CONFIG:-}" ]; then
    echo "Set CONFIG=configs/experiments/<name>.yaml" >&2
    exit 1
fi
RUN_ROOT="${RUN_ROOT:-outputs/comparison}"
NAME="$(basename "$CONFIG" .yaml)"

if [ -n "${QUEUE:-}" ]; then
    read -r NEXT REST <<< "$QUEUE"
    if CONFIG="$NEXT" QUEUE="$REST" RUN_ROOT="$RUN_ROOT" sbatch --export=ALL \
        --job-name="$(basename "$NEXT" .yaml)" --dependency="afterany:${SLURM_JOB_ID}" slurm/run_one.sbatch; then
        echo "Queued ${NEXT} to start after job ${SLURM_JOB_ID}"
    else
        echo "Could not queue ${NEXT}; submit the rest of the queue by hand: ${QUEUE}" >&2
    fi
fi

echo "Running ${CONFIG} -> ${RUN_ROOT}/${NAME}"
PYTHONPATH=. python3 scripts/run_stage.py --config "$CONFIG" --output_dir "${RUN_ROOT}/${NAME}"
