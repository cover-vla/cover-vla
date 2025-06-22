
CUDA_VISIBLE_DEVICES=6 python ../scripts/finetune_trajectory_dino.py \
    --epochs 2000 \
    --batch_size 1024 \
    --lr 5e-5 \
    --history_length 8 \
    --augmented_dataset ../augmented_datasets/libero_spatial_oft.pkl \
    --save_name libero_spatial_oft_dino \
    --use_transformer \
    --use_wandb 