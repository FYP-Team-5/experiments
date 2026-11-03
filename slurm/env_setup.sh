#!/bin/bash

# Activate python virtual environment or conda environment
# conda activate gfv_env

# Set Hugging Face cache location to scratch directory (absolute path; without the
# leading / the cache would be created inside the repo). Export HF_HOME before
# sourcing this file to use another cache, e.g. one that already has the models.
export HF_HOME="${HF_HOME:-/common/scratch/projectgrps/CS480/CS480G3/huggingface_cache/}"

# Set python path to root directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export PYTHONUNBUFFERED=1