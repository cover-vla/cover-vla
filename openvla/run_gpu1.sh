#!/bin/bash


# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --center_crop True \
#   --clip_action_iter 5 \
#   --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_action_encoder_only_augmented_dataset_final.pt



# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --center_crop True \
#   --clip_action_iter 1 \
#   --language_transformation_type out_set \
#   --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_final.pt

# loop over beta values
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
--model_family openvla \
--pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
--task_suite_name libero_spatial \
--center_crop True \
--clip_action_iter 1 \
--language_transformation_type rephrase \
--clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_final.pt \

wait
# loop over beta values
for beta in 0.05 0.01; do
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
--model_family openvla \
--pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
--task_suite_name libero_spatial \
--center_crop True \
--clip_action_iter 5 \
--language_transformation_type rephrase \
--clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_final.pt \
--beta $beta
done


# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --center_crop True \
#   --language_transformation True \
#   --language_transformation_type synonym \
#   --clip_action_iter 5 \
#   --clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_action_encoder_only_augmented_dataset_final.pt

