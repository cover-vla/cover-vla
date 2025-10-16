#!/bin/bash

# RoboMonkey OpenVLA-Mini Consistency Check Evaluation Script
# Uses Lipschitz continuity-based consistency scoring with adaptive softmax temperature

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.


CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi_robomonkey.py \
    --task_suite_name simpler_widowx \
    --lang_transform_type no_transform \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 300 \
    --augmented_samples 16


CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi_robomonkey.py \
    --task_suite_name simpler_ood \
    --lang_transform_type no_transform \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 300 \
    --augmented_samples 16



CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi_robomonkey.py \
    --task_suite_name simpler_widowx \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 300 \
    --augmented_samples 16


CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi_robomonkey.py \
    --task_suite_name simpler_ood \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 300 \
    --augmented_samples 16


# ## rephrased finetuned openpi


# CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi_robomonkey.py \
#     --task_suite_name simpler_widowx \
#     --lang_transform_type no_transform \
#     --pretrained_checkpoint juexzz/INTACT-pi0-finetune-rephrase-bridge \
#     --num_trials_per_task 300 \
#     --augmented_samples 41


# CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi_robomonkey.py \
#     --task_suite_name simpler_ood \
#     --lang_transform_type no_transform \
#     --pretrained_checkpoint juexzz/INTACT-pi0-finetune-rephrase-bridge \
#     --num_trials_per_task 300 \
#     --augmented_samples 41


# CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi_robomonkey.py \
#     --task_suite_name simpler_widowx \
#     --lang_transform_type rephrase \
#     --pretrained_checkpoint juexzz/INTACT-pi0-finetune-rephrase-bridge \
#     --num_trials_per_task 300 \
#     --augmented_samples 41


# CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi_robomonkey.py \
#     --task_suite_name simpler_ood \
#     --lang_transform_type rephrase \
#     --pretrained_checkpoint juexzz/INTACT-pi0-finetune-rephrase-bridge \
#     --num_trials_per_task 300 \
#     --augmented_samples 41

wait