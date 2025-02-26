# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type antonym \
  --task_seed_list [40,41,42,43,44] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type no_transform \
  --task_seed_list [40,41,42,43,44] \


# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type verb_noun_shuffle \
  --task_seed_list [40,41,42,43,44] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type out_set \
  --task_seed_list [40,41,42,43,44] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type interpolation \
  --task_seed_list [40,41,42,43,44] \

# Launch LIBERO-10 (LIBERO-Long) evals 
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type synonym \
  --task_seed_list [40,41,42,43,44] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type negation \
  --task_seed_list [40,41,42,43,44] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type in_set \
  --task_seed_list [40,41,42,43,44] \

# Launch LIBERO-10 (LIBERO-Long) evals
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --transform_type random_shuffle \
  --task_seed_list [40,41,42,43,44] \
