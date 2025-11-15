#!/bin/bash

# RoboMonkey OpenVLA-Mini Consistency Check Evaluation Script
# Uses Lipschitz continuity-based consistency scoring with adaptive softmax temperature

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.


CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_openpi_teaser.py \
    --task_suite_name simpler_zucchini_on_towel \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 150 \
    --use_verifier True \
    --policy_batch_inference_size 5 \
    --lang_rephrase_num 8 &

CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi_teaser.py \
    --task_suite_name simpler_tennis_ball_in_basket \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 150 \
    --use_verifier True \
    --policy_batch_inference_size 5 \
    --lang_rephrase_num 8 &


CUDA_VISIBLE_DEVICES=2 python ../run_simpler_eval_with_openpi_teaser.py \
    --task_suite_name simpler_redbull_on_plate \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 150 \
    --use_verifier True \
    --policy_batch_inference_size 5 \
    --lang_rephrase_num 8 &
 

wait