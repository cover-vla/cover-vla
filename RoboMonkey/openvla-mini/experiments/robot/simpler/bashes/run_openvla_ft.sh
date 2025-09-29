#!/bin/bash

# RoboMonkey OpenVLA-Mini Consistency Check Evaluation Script
# Uses Lipschitz continuity-based consistency scoring with adaptive softmax temperature

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.

echo "Running evaluation with OpenVLA-FT..."


# CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openvla_ft.py \
#     --task_suite_name simpler_widowx \
#     --lang_transform_type rephrase \
#     --use_vla_clip_trajectory_scorer False \
#     --clip_select_action_num_candidates 1 \
#     --batch_temperature 0 \
#     --num_trials_per_task 300 \
#     --use_consistency_check False \
#     --save_image_name reward_img_openvla_ft.jpg \
#     --batch_server_url http://localhost:3700 \

CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openvla_ft.py \
    --task_suite_name simpler_widowx \
    --lang_transform_type no_transform \
    --use_vla_clip_trajectory_scorer False \
    --clip_select_action_num_candidates 1 \
    --batch_temperature 0 \
    --num_trials_per_task 300 \
    --use_consistency_check False \
    --save_image_name reward_img_openvla_ft.jpg \
    --batch_server_url http://localhost:3700 \

# CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openvla_ft.py \
#     --task_suite_name simpler_ood \
#     --lang_transform_type rephrase \
#     --use_vla_clip_trajectory_scorer False \
#     --clip_select_action_num_candidates 1 \
#     --batch_temperature 0 \
#     --num_trials_per_task 300 \
#     --use_consistency_check False \
#     --save_image_name reward_img_openvla_ft.jpg \
#     --batch_server_url http://localhost:3700 \

CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openvla_ft.py \
    --task_suite_name simpler_ood \
    --lang_transform_type no_transform \
    --use_vla_clip_trajectory_scorer False \
    --clip_select_action_num_candidates 1 \
    --batch_temperature 0 \
    --num_trials_per_task 300 \
    --use_consistency_check False \
    --save_image_name reward_img_openvla_ft.jpg \
    --batch_server_url http://localhost:3700 \


echo "OpenVLA-FT evaluation completed!"

