# #!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_oracle.py \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --use_oracle_scorer True \
#   --vla_clip_history_length 10 \
#   --clip_select_action_num_candidates 15 \
#   --lang_transform_type rephrase


echo "🚀 Starting all experiment groups..."

# --- Group 1: Run candidates 1 and 5 in series (in the background) ---
(
  echo "Starting serial group: candidate 1 -> 5"

  # Run for num_candidates=1
  CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_vla_dino_trajectory_scorer True \
    --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_all_diverse_final_best.pt \
    --vla_clip_history_length 10 \
    --vla_clip_score_threshold 0.5 \
    --clip_select_action_num_candidates 1 \
    --lang_transform_type no_transform
  
  # Run for num_candidates=1
  CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_vla_dino_trajectory_scorer True \
    --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_all_diverse_final_best.pt \
    --vla_clip_history_length 10 \
    --vla_clip_score_threshold 0.5 \
    --clip_select_action_num_candidates 1 \
    --lang_transform_type rephrase

  # Run for num_candidates=5 (only if the first command succeeds)
  CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_vla_dino_trajectory_scorer True \
    --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_all_diverse_final_best.pt \
    --vla_clip_history_length 10 \
    --vla_clip_score_threshold 0.5 \
    --clip_select_action_num_candidates 10 \
    --lang_transform_type rephrase
    
  echo "✅ Finished serial group."
) &


# --- Group 2: Run candidate 10 in parallel (in the background) ---
(
  CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_vla_dino_trajectory_scorer True \
    --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_all_diverse_final_best.pt \
    --vla_clip_history_length 10 \
    --vla_clip_score_threshold 0.5 \
    --clip_select_action_num_candidates 5 \
    --lang_transform_type rephrase

  CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_vla_dino_trajectory_scorer True \
    --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_all_diverse_final_best.pt \
    --vla_clip_history_length 10 \
    --vla_clip_score_threshold 0.5 \
    --clip_select_action_num_candidates 20 \
    --lang_transform_type rephrase
  echo "✅ Finished parallel job: candidate 10."
) &


# --- Group 3: Run candidate 25 in parallel (in the background) ---
(
  echo "Starting parallel job: candidate 25"
  CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_vla_dino_trajectory_scorer True \
    --vla_clip_traj_model_path ../clip_verifier/bash/trajectory_checkpoints/libero_spatial_all_diverse_final_best.pt \
    --vla_clip_history_length 10 \
    --vla_clip_score_threshold 0.5 \
    --clip_select_action_num_candidates 30 \
    --lang_transform_type rephrase
  echo "✅ Finished parallel job: candidate 25."
) &

