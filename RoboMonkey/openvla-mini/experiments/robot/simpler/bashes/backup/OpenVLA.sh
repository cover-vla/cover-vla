#!/bin/bash

# RoboMonkey OpenVLA-Mini Consistency Check Evaluation Script
# Uses Lipschitz continuity-based consistency scoring with adaptive softmax temperature

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.


# CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_verifier_consistancy_check.py \
#     --task_suite_name simpler_widowx \
#     --use_vla_clip_trajectory_scorer False \
#     --vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_23.pt \
#     --vla_clip_history_length 10 \
#     --vla_clip_score_threshold 0.05 \
#     --lang_transform_type rephrase \
#     --clip_select_action_num_candidates 1 \
#     --batch_temperature 0 \
#     --num_trials_per_task 300 \
#     --use_consistency_check False \
#     --consistency_temperature_scale 10.0 \
#     --clip_select_action_strategy highest_score \
#     --consistency_top_k 5 \
#     --save_image_name openvla_reward_img.jpg \
#     --batch_server_url http://localhost:3200

CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_verifier_consistancy_check.py \
    --task_suite_name simpler_widowx \
    --use_vla_clip_trajectory_scorer False \
    --vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_23.pt \
    --vla_clip_history_length 10 \
    --vla_clip_score_threshold 0.05 \
    --lang_transform_type no_transform \
    --clip_select_action_num_candidates 1 \
    --batch_temperature 0 \
    --num_trials_per_task 300 \
    --use_consistency_check False \
    --consistency_temperature_scale 10.0 \
    --clip_select_action_strategy highest_score \
    --consistency_top_k 5 \
    --save_image_name openvla_reward_img.jpg \
    --batch_server_url http://localhost:3200

# CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_verifier_consistancy_check.py \
#     --task_suite_name simpler_ood \
#     --use_vla_clip_trajectory_scorer False \
#     --vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_23.pt \
#     --vla_clip_history_length 10 \
#     --vla_clip_score_threshold 0.05 \
#     --lang_transform_type rephrase \
#     --clip_select_action_num_candidates 1 \
#     --batch_temperature 0 \
#     --num_trials_per_task 300 \
#     --use_consistency_check False \
#     --consistency_temperature_scale 10.0 \
#     --clip_select_action_strategy highest_score \
#     --consistency_top_k 5 \
#     --save_image_name openvla_reward_img.jpg \
#     --batch_server_url http://localhost:3200

CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_verifier_consistancy_check.py \
    --task_suite_name simpler_ood \
    --use_vla_clip_trajectory_scorer False \
    --vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_23.pt \
    --vla_clip_history_length 10 \
    --vla_clip_score_threshold 0.05 \
    --lang_transform_type no_transform \
    --clip_select_action_num_candidates 1 \
    --batch_temperature 0 \
    --num_trials_per_task 300 \
    --use_consistency_check False \
    --consistency_temperature_scale 10.0 \
    --clip_select_action_strategy highest_score \
    --consistency_top_k 5 \
    --save_image_name openvla_reward_img.jpg \
    --batch_server_url http://localhost:3200
