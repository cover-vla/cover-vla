
CUDA_VISIBLE_DEVICES=6 python ../scripts/finetune.py \
    --epochs 150 \
    --batch_size 128 \
    --lr 7e-5 \
    --save_name 90_clip_action_encoder_only \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --dataset_folders libero_90 \
    --use_wandb

CUDA_VISIBLE_DEVICES=6 python ../scripts/finetune.py \
    --epochs 150 \
    --batch_size 128 \
    --lr 7e-5 \
    --save_name 90_clip_action_encoder_only_augmented_dataset \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --dataset_folders libero_90 \
    --augmented_dataset ../augmented_datasets/libero_90.pkl \
    --use_wandb

CUDA_VISIBLE_DEVICES=6 python ../scripts/finetune.py \
    --epochs 150 \
    --batch_size 128 \
    --lr 7e-5 \
    --save_name 90_clip_action_encoder_only_positive_only_augmented_dataset \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --dataset_folders libero_90 \
    --augmented_dataset ../augmented_datasets/libero_90_positive_only.pkl \
    --use_wandb

