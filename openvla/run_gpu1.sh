#!/bin/bash


CUDA_VISIBLE_DEVICES="" python experiments/robot/libero/run_libero_eval.py \
--model_family openvla \
--pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
--task_suite_name libero_spatial \
--center_crop True \
--clip_action_iter 5 \
--language_transformation_type rephrase \
--clip_model_path /home/xilun/vla-clip/clip_verifier/bash/model_checkpoints/spatial_clip_final.pt \
--beta 10 \
--alignment_text original \
--use_gradient_optimization True \
