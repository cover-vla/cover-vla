#!/bin/bash

# CUDA_VISIBLE_DEVICES=5 python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --center_crop True \
#   --language_transformation True \
#   --language_transformation_type synonym \
#   --clip_filtered_actions True \
#   --clip_action_iter 10

# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --center_crop True \
#   --clip_action_iter 5


CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --clip_action_iter 5 \
  --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_action_encoder_only_augmented_dataset_final.pt

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --language_transformation True \
  --language_transformation_type synonym \
  --clip_action_iter 5 \
  --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_action_encoder_only_augmented_dataset_final.pt

