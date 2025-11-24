#!/bin/bash

# RoboMonkey OpenVLA-Mini Consistency Check Evaluation Script
# Uses Lipschitz continuity-based consistency scoring with adaptive softmax temperature

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_openpi_presentation.py \
    --task_suite_name simpler_redbull_on_plate \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 10 \
    --use_verifier False \
    --policy_batch_inference_size 1 \
    --lang_rephrase_num 1
