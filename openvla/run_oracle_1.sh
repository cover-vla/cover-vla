(
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval_oracle.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 1 \
    --clip_select_action_strategy highest_score \
    --use_oracle_scorer True

CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval_oracle.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 5 \
    --clip_select_action_strategy highest_score \
    --use_oracle_scorer True
) &


CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval_oracle.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 10 \
    --clip_select_action_strategy highest_score \
    --use_oracle_scorer True &

CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval_oracle.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 20 \
    --clip_select_action_strategy highest_score \
    --use_oracle_scorer True &


CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_oracle.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --lang_transform_type rephrase \
    --clip_select_action_num_candidates 30 \
    --clip_select_action_strategy highest_score \
    --use_oracle_scorer True &

wait