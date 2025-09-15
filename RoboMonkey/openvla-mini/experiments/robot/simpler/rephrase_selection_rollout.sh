#!/bin/bash

# RoboMonkey OpenVLA-Mini Parallel Evaluation Script
# This script runs multiple evaluations in parallel with different parameters

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.

for k in {0..18}; do

CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_rephrase_selection.py \
--task_suite_name simpler_widowx \
--use_vla_clip_trajectory_scorer True \
--vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_23.pt \
--vla_clip_history_length 10 \
--vla_clip_score_threshold 0.05 \
--clip_select_action_num_candidates 1 \
--lang_transform_type rephrase \
--num_trials_per_task 30 \
--batch_temperature 1 \
--instruction_index $k
done

for k in {0..18}; do

CUDA_VISIBLE_DEVICES=0 python run_simpler_eval_rephrase_selection.py \
--task_suite_name simpler_ood \
--use_vla_clip_trajectory_scorer True \
--vla_clip_traj_model_path /root/vla-clip/bridge_verifier/bridge_rephrases_epoch_23.pt \
--vla_clip_history_length 10 \
--vla_clip_score_threshold 0.05 \
--clip_select_action_num_candidates 1 \
--lang_transform_type rephrase \
--num_trials_per_task 30 \
--batch_temperature 1 \
--instruction_index $k
done

