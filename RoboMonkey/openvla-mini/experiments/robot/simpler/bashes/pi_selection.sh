#!/bin/bash

# RoboMonkey OpenVLA-Mini Consistency Check Evaluation Script
# Uses Lipschitz continuity-based consistency scoring with adaptive softmax temperature

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.
(
for instruction_index in {0..2}
do
CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name  simpler_spoon_on_towel\
    --lang_transform_type rephrase \
    --instruction_index $instruction_index \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 50 \
    --use_verifier False \
    --policy_batch_inference_size 1 \
    --lang_rephrase_num 1 &

CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name  simpler_put_eggplant_in_basket\
    --lang_transform_type rephrase \
    --instruction_index $instruction_index \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 50 \
    --use_verifier False \
    --policy_batch_inference_size 1 \
    --lang_rephrase_num 1 &
wait
done
) &
(
for instruction_index in {3..5}
do
CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name  simpler_spoon_on_towel\
    --lang_transform_type rephrase \
    --instruction_index $instruction_index \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 50 \
    --use_verifier False \
    --policy_batch_inference_size 1 \
    --lang_rephrase_num 1 &

CUDA_VISIBLE_DEVICES=1 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name  simpler_put_eggplant_in_basket\
    --lang_transform_type rephrase \
    --instruction_index $instruction_index \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 50 \
    --use_verifier False \
    --policy_batch_inference_size 1 \
    --lang_rephrase_num 1 &
wait
done
) &

(
for instruction_index in {6..8}
do
CUDA_VISIBLE_DEVICES=2 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name  simpler_spoon_on_towel\
    --lang_transform_type rephrase \
    --instruction_index $instruction_index \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 50 \
    --use_verifier False \
    --policy_batch_inference_size 1 \
    --lang_rephrase_num 1 &

CUDA_VISIBLE_DEVICES=3 python ../run_simpler_eval_with_openpi.py \
    --task_suite_name  simpler_put_eggplant_in_basket\
    --lang_transform_type rephrase \
    --instruction_index $instruction_index \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 50 \
    --use_verifier False \
    --policy_batch_inference_size 1 \
    --lang_rephrase_num 1 &
wait
done
) &