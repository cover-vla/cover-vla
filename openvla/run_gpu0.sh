# #!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --use_vla_dino_trajectory_scorer True \
#   --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_500.pt \
#   --vla_clip_history_length 10 \
#   --vla_clip_score_threshold 15 \
#   --clip_select_action_num_candidates 1 \
#   --lang_transform_type no_transform &


CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_dino_trajectory_scorer True \
  --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_500.pt \
  --vla_clip_history_length 10 \
  --vla_clip_score_threshold 15 \
  --clip_select_action_num_candidates 1 \
  --lang_transform_type rephrase &


CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_dino_trajectory_scorer True \
  --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_500.pt \
  --vla_clip_history_length 10 \
  --vla_clip_score_threshold 15 \
  --clip_select_action_num_candidates 5 \
  --lang_transform_type rephrase &


# CUDA_VISIBLE_DEVICES=5 python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --use_vla_dino_trajectory_scorer True \
#   --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_500.pt \
#   --vla_clip_history_length 10 \
#   --vla_clip_score_threshold 15 \
#   --clip_select_action_num_candidates 10 \
#   --lang_transform_type rephrase &


CUDA_VISIBLE_DEVICES=4 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_dino_trajectory_scorer True \
  --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_500.pt \
  --vla_clip_history_length 10 \
  --vla_clip_score_threshold 15 \
  --clip_select_action_num_candidates 15 \
  --lang_transform_type rephrase &

CUDA_VISIBLE_DEVICES=5 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_dino_trajectory_scorer True \
  --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_500.pt \
  --vla_clip_history_length 10 \
  --vla_clip_score_threshold 15 \
  --clip_select_action_num_candidates 25 \
  --lang_transform_type rephrase \

