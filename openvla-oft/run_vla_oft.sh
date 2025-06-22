# CUDA_VISIBLE_DEVICES=6 python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --use_vla_dino_trajectory_scorer True \
#   --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_oft_epoch_500.pt \
#   --vla_clip_history_length 8 \
#   --vla_clip_score_threshold 15 \
#   --clip_select_action_num_candidates 1 \
#   --lang_transform_type no_transform &

# CUDA_VISIBLE_DEVICES=7 python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --use_vla_dino_trajectory_scorer True \
#   --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_oft_epoch_500.pt \
#   --vla_clip_history_length 8 \
#   --vla_clip_score_threshold 15 \
#   --clip_select_action_num_candidates 1 \
#   --lang_transform_type rephrase &

# wait 

CUDA_VISIBLE_DEVICES=6 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_dino_trajectory_scorer True \
  --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_oft_epoch_500.pt \
  --vla_clip_history_length 8 \
  --vla_clip_score_threshold 15 \
  --clip_select_action_num_candidates 5 \
  --lang_transform_type rephrase &

CUDA_VISIBLE_DEVICES=7 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_dino_trajectory_scorer True \
  --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_oft_epoch_500.pt \
  --vla_clip_history_length 8 \
  --vla_clip_score_threshold 15 \
  --clip_select_action_num_candidates 10 \
  --lang_transform_type rephrase &

wait 

CUDA_VISIBLE_DEVICES=6 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_dino_trajectory_scorer True \
  --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_oft_epoch_500.pt \
  --vla_clip_history_length 8 \
  --vla_clip_score_threshold 15 \
  --clip_select_action_num_candidates 15 \
  --lang_transform_type rephrase &

CUDA_VISIBLE_DEVICES=7 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_dino_trajectory_scorer True \
  --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_oft_epoch_500.pt \
  --vla_clip_history_length 8 \
  --vla_clip_score_threshold 15 \
  --clip_select_action_num_candidates 25 \
  --lang_transform_type rephrase &

wait 