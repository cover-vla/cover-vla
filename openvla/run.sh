#!/bin/bash

# Define the language transforms to iterate over
# LANG_TRANSFORMS=("no_transform" "random_shuffle" "synonym" "antonym" "negation" "verb_noun_shuffle" "out_set")
LANG_TRANSFORMS=("no_transform")

# Iterate over each transform
for transform in "${LANG_TRANSFORMS[@]}"; do
    echo "Running evaluation with lang_transform: $transform"
    
    CUDA_VISIBLE_DEVICES=7 python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --center_crop True \
        --lang_transform "$transform"
        
    echo "Completed evaluation with lang_transform: $transform"
    echo "----------------------------------------"
done

# # Launch LIBERO-Object evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
#   --task_suite_name libero_object \
#   --center_crop True

# # Launch LIBERO-Goal evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
#   --task_suite_name libero_goal \
#   --center_crop True

# # Launch LIBERO-10 (LIBERO-Long) evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
#   --task_suite_name libero_10 \
#   --center_crop True