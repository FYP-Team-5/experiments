#!/bin/bash

# Activate python virtual environment or conda environment
# conda activate gfv_env

# Set Hugging Face cache location to scratch directory
export HF_HOME="common/scratch/projectgrps/CS480/CS480G3/huggingface_cache/"

# Set python path to root directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export PYTHONUNBUFFERED=1