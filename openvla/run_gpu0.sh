#!/bin/bash

# loop over beta values
for beta in 0.1 0.5 1 3; do
    for alignment_text in original transformed; do
        CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --center_crop True \
        --clip_action_iter 5 \
        --language_transformation_type rephrase \
        --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_final.pt \
        --beta $beta \
        --alignment_text $alignment_text
    done
done


