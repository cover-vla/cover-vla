#!/bin/bash

# RoboMonkey OpenVLA-Mini Parallel Evaluation Script
# This script runs multiple evaluations in parallel with different parameters

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.

# Run for num_candidates=1 with no_transform
echo "Running: num_candidates=1, transform=no_transform"
CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_with_verifier.py \
--task_suite_name simpler_ood \
--use_vla_clip_trajectory_scorer True \
--vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_20.pt \
--vla_clip_history_length 10 \
--vla_clip_score_threshold 0.05 \
--clip_select_action_num_candidates 1 \
--lang_transform_type no_transform \
--num_trials_per_task 30

echo "Running: num_candidates=1, transform=rephrase"
CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_with_verifier.py \
--task_suite_name simpler_ood \
--use_vla_clip_trajectory_scorer True \
--vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_20.pt \
--vla_clip_history_length 10 \
--vla_clip_score_threshold 0.05 \
--clip_select_action_num_candidates 1 \
--lang_transform_type rephrase \
--num_trials_per_task 30

echo "Running: num_candidates=2, transform=rephrase"
CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_with_verifier.py \
--task_suite_name simpler_ood \
--use_vla_clip_trajectory_scorer True \
--vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_20.pt \
--vla_clip_history_length 10 \
--vla_clip_score_threshold 0.05 \
--clip_select_action_num_candidates 2 \
--lang_transform_type rephrase \
--num_trials_per_task 30

echo "Running: num_candidates=4, transform=rephrase"
CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_with_verifier.py \
--task_suite_name simpler_ood \
--use_vla_clip_trajectory_scorer True \
--vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_20.pt \
--vla_clip_history_length 10 \
--vla_clip_score_threshold 0.05 \
--clip_select_action_num_candidates 4 \
--lang_transform_type rephrase \
--num_trials_per_task 30

echo "Running: num_candidates=8, transform=rephrase"
CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_with_verifier.py \
--task_suite_name simpler_ood \
--use_vla_clip_trajectory_scorer True \
--vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_20.pt \
--vla_clip_history_length 10 \
--vla_clip_score_threshold 0.05 \
--clip_select_action_num_candidates 8 \
--lang_transform_type rephrase \
--num_trials_per_task 30

CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_with_verifier.py \
--task_suite_name simpler_ood \
--use_vla_clip_trajectory_scorer True \
--vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_20.pt \
--vla_clip_history_length 10 \
--vla_clip_score_threshold 0.05 \
--clip_select_action_num_candidates 10 \
--lang_transform_type rephrase \
--num_trials_per_task 30

echo "Running: num_candidates=16, transform=rephrase"
CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_with_verifier.py \
--task_suite_name simpler_ood \
--use_vla_clip_trajectory_scorer True \
--vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_20.pt \
--vla_clip_history_length 10 \
--vla_clip_score_threshold 0.05 \
--clip_select_action_num_candidates 16 \
--lang_transform_type rephrase \
--num_trials_per_task 30


echo "Running: num_candidates=32, transform=rephrase"
CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_with_verifier.py \
--task_suite_name simpler_ood \
--use_vla_clip_trajectory_scorer True \
--vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_20.pt \
--vla_clip_history_length 10 \
--vla_clip_score_threshold 0.05 \
--clip_select_action_num_candidates 32 \
--lang_transform_type rephrase \
--num_trials_per_task 30