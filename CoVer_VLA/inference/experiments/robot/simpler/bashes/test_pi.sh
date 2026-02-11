#!/bin/bash

# NOTE: Activate the 'cover' virtual environment first:
#   source /home/xilunz/rebuttal/vla-clip/.venv_cover/bin/activate
# And set environment variables:
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
# Add inference root to PYTHONPATH (go up 4 levels: bashes -> simpler -> robot -> experiments -> inference)
INFERENCE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PYTHONPATH="$INFERENCE_ROOT:$PYTHONPATH"
export PRISMATIC_DATA_ROOT=.

CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name simpler_widowx \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 100 \
    --use_verifier True \
    --policy_batch_inference_size 5 \
    --lang_rephrase_num 8

CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name simpler_ood \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 100 \
    --use_verifier True \
    --policy_batch_inference_size 5 \
    --lang_rephrase_num 8

wait