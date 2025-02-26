
# Launch LIBERO-10 (LIBERO-Long) evals Done 
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type synonym \
  --task_seed_list [40,41,42,43,44] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type negation \
  --task_seed_list [40,41,42,43,44] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type in_set \
  --task_seed_list [40,41,42,43,44] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type random_shuffle \
  --task_seed_list [40,41,42,43,44] \
