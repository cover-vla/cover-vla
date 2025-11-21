#!/bin/bash

# RoboMonkey OpenVLA-Mini Consistency Check Evaluation Script
# Uses Lipschitz continuity-based consistency scoring with adaptive softmax temperature

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export PRISMATIC_DATA_ROOT=. && export PYTHONPATH=.


for batch_size in 100; do
    CUDA_VISIBLE_DEVICES=0 python ../run_simpler_eval_with_openpi_latency.py \
        --latency_only_mode True \
        --task_suite_name simpler_zucchini_on_towel \
        --lang_transform_type no_transform \
        --latency_test_steps 30 \
        --latency_test_instruction "put the zucchini on the towel" \
        --lang_rephrase_num 1 \
        --policy_batch_inference_size $batch_size \
        --num_trials_per_task 10
done


