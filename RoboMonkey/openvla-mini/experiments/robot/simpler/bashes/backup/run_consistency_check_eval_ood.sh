#!/bin/bash

# RoboMonkey OpenVLA-Mini Consistency Check Evaluation Script
# Uses Lipschitz continuity-based consistency scoring with adaptive softmax temperature

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.

echo "Running evaluation with Lipschitz-based consistency check..."

# Run with consistency check enabled
# Key parameters:
# - use_consistency_check: Enables consistency check and uses score to adjust softmax temperature
# - clip_select_action_strategy: softmax_sample for probabilistic action selection
# - consistency_temperature_scale: Higher values = more temperature variation based on consistency

for i in {1,3,5,9,17}; do
CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_with_verifier_consistancy_check.py \
    --task_suite_name simpler_ood \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_23.pt \
    --vla_clip_history_length 10 \
    --vla_clip_score_threshold 0.05 \
    --clip_select_action_num_candidates $i \
    --batch_temperature 1.0 \
    --num_trials_per_task 30 \
    --use_consistency_check True \
    --consistency_temperature_scale 5.0 \
    --clip_select_action_strategy softmax_sample \
    --consistency_top_k 5 \
    --save_image_name consistency_reward_img_ood.jpg \
    --batch_server_url http://localhost:3200
done
echo "Lipschitz consistency check evaluation completed!"

for i in {3,5,9,17}; do
CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_with_verifier_consistancy_check.py \
    --task_suite_name simpler_ood \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_23.pt \
    --vla_clip_history_length 10 \
    --vla_clip_score_threshold 0.05 \
    --clip_select_action_num_candidates $i \
    --batch_temperature 1.0 \
    --num_trials_per_task 30 \
    --use_consistency_check False \
    --consistency_temperature_scale 5.0 \
    --clip_select_action_strategy highest_score \
    --consistency_top_k 5 \
    --save_image_name consistency_reward_img_ood.jpg \
    --batch_server_url http://localhost:3200
done