# #!/bin/bash

CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/spatial_transformer_h10_neg0_padded_exp_final_best.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --lang_transform_type no_transform \
    --clip_select_action_num_candidates 1 \
    --clip_select_action_strategy softmax_sample \
    --use_original_task_description True

CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/spatial_transformer_h10_neg0_padded_exp_final_best.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --lang_transform_type no_transform \
    --clip_select_action_num_candidates 10 \
    --clip_select_action_strategy softmax_sample \
    --use_original_task_description False


CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/spatial_transformer_h10_neg0_padded_exp_final_best.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 1 \
    --clip_select_action_strategy softmax_sample \
    --use_original_task_description False