# #!/bin/bash

CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_1300.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 1 \
    --clip_select_action_strategy highest_score \
    --use_original_task_description True &


CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_1300.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 5 \
    --clip_select_action_strategy highest_score \
    --use_original_task_description True &


CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_1300.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 10 \
    --clip_select_action_strategy highest_score \
    --use_original_task_description True &


CUDA_VISIBLE_DEVICES=4 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_1300.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 15 \
    --clip_select_action_strategy highest_score \
    --use_original_task_description True &


CUDA_VISIBLE_DEVICES=5 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_1300.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 20 \
    --clip_select_action_strategy highest_score \
    --use_original_task_description True &  


CUDA_VISIBLE_DEVICES=6 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_1300.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 25 \
    --clip_select_action_strategy highest_score \
    --use_original_task_description True &

wait
