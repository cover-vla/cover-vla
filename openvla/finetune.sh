# export CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
# --vla_path "openvla/openvla-7b" \
# --data_root_dir "./../dataset/modified_libero_rlds" \
# --dataset_name libero_spatial_no_noops \
# --run_root_dir "./../run/openvla/libero_spatial/" \
# --adapter_tmp_dir "./../run/openvla/libero_spatial/adapter" \
# --lora_rank 32 \
# --batch_size 4 \
# --grad_accumulation_steps 1 \
# --learning_rate 5e-4 \
# --image_aug True \
# --wandb_project openvla \
# --wandb_entity libero_spatial_lora \
# --save_steps 5000

CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vla_path "openvla/openvla-7b" \
--data_root_dir "./modified_libero_rlds" \
--dataset_name libero_spatial_no_noops \
--run_root_dir "./run/libero_spatial/" \
--adapter_tmp_dir "./run/libero_spatial/adapter" \
--lora_rank 32 \
--batch_size 6 \
--grad_accumulation_steps 1 \
--learning_rate 5e-4 \
--image_aug True \
--wandb_project openvla \
--wandb_entity willhan327 \
--save_steps 5000 \
--loss_type "both_clip"


# CUDA_VISIBLE_DEVICES=2,3,7 torchrun --standalone --nnodes 1 --nproc-per-node 3 vla-scripts/finetune.py \
# --vla_path "openvla/openvla-7b-finetuned-libero-spatial" \
# --data_root_dir "./modified_libero_rlds" \
# --dataset_name libero_spatial_no_noops \
# --run_root_dir "./run/libero_spatial/" \
# --adapter_tmp_dir "./run/libero_spatial/adapter" \
# --lora_rank 32 \
# --batch_size 6 \
# --grad_accumulation_steps 1 \
# --learning_rate 5e-4 \
# --image_aug True \
# --wandb_project openvla \
# --wandb_entity willhan327 \
# --save_steps 5000 \
# --loss_type "contrastive"




# CUDA_VISIBLE_DEVICES=2,3,7 torchrun --standalone --nnodes 1 --nproc-per-node 3 vla-scripts/finetune.py \
# --vla_path "openvla/openvla-7b-finetuned-libero-spatial" \
# --data_root_dir "./modified_libero_rlds" \
# --dataset_name libero_spatial_no_noops \
# --run_root_dir "./run/libero_spatial/" \
# --adapter_tmp_dir "./run/libero_spatial/adapter" \
# --lora_rank 32 \
# --batch_size 6 \
# --grad_accumulation_steps 1 \
# --learning_rate 5e-4 \
# --image_aug True \
# --wandb_project openvla \
# --wandb_entity willhan327 \
# --save_steps 5000 \
# --loss_type "both"