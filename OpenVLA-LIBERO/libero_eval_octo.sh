# Launch LIBERO-Spatial evals
# CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --center_crop True

# Launch LIBERO-Object evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
#   --task_suite_name libero_object \
#   --center_crop True

# # Launch LIBERO-Goal evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
#   --task_suite_name libero_goal \
#   --center_crop True

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family octo \
  --pretrained_checkpoint octo-base \
  --task_suite_name libero_10 \
  --center_crop True \
  --num_trials_per_task 1 \
  --transform_type antonym \
