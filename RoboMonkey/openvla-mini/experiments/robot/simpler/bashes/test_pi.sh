#!/bin/bash

# RoboMonkey OpenVLA-Mini Consistency Check Evaluation Script
# Uses Lipschitz continuity-based consistency scoring with adaptive softmax temperature

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.

(
for i in {4,2,1}; do

CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name simpler_widowx \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 150 \
    --use_verifier True \
    --policy_batch_inference_size 5 \
    --lang_rephrase_num $i
done
) &
(
for j in {1,3,5}; do

CUDA_VISIBLE_DEVICES=2 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name simpler_widowx \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 150 \
    --use_verifier True \
    --policy_batch_inference_size $j \
    --lang_rephrase_num 8

done
) &
(

for i in {1,2,4}; do

CUDA_VISIBLE_DEVICES=3 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name simpler_ood \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 150 \
    --use_verifier True \
    --policy_batch_inference_size 5 \
    --lang_rephrase_num $i
done

for j in {1,3,5}; do

CUDA_VISIBLE_DEVICES=3 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name simpler_ood \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 150 \
    --use_verifier True \
    --policy_batch_inference_size $j \
    --lang_rephrase_num 8

done
) &