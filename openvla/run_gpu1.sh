# #!/bin/bash

CUDA_VISIBLE_DEVICES=4 python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/spatial_transformer_h10_neg2_padded_epoch_450.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --lang_transform no_transform \
    --clip_select_action_num_candidates 5 \
    --clip_select_action_strategy softmax_sample



