
# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type random_shuffle \
  --num_trials_per_task 1 \
  --task_seed_list [40] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type interpolation \
  --num_trials_per_task 1 \
  --task_seed_list [40] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type no_transform \
  --num_trials_per_task 1 \
  --task_seed_list [40] \
  
