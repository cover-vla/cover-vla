# CUDA_VISIBLE_DEVICES=3 python ../scripts/finetune.py \
#     --epochs 150 \
#     --batch_size 128 \
#     --lr 7e-5 \
#     --save_name spatial_clip_action_encoder_only \
#     --dataset_path /home/xilun/LIBERO/libero/datasets \
#     --dataset_folders libero_spatial \
#     --use_wandb


CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_action_image.py \
    --epochs 5 \
    --batch_size 512 \
    --lr 7e-5 \
    --save_name all_clip_action_vision \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --augmented_dataset ../augmented_datasets/libero_all.pkl \
    --use_wandb

# CUDA_VISIBLE_DEVICES=3 python ../scripts/finetune.py \
#     --epochs 150 \
#     --batch_size 128 \
#     --lr 7e-5 \
#     --save_name spatial_clip_action_encoder_only_positive_only_augmented_dataset \
#     --dataset_path /home/xilun/LIBERO/libero/datasets \
#     --dataset_folders libero_spatial \
#     --augmented_dataset ../augmented_datasets/libero_spatial_positive_only.pkl \
#     --use_wandb

# wait

