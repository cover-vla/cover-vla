
CUDA_VISIBLE_DEVICES=6 python ../scripts/finetune_trajectory_image.py \
    --epochs 2000 \
    --batch_size 4096 \
    --lr 1e-4 \
    --history_length 8 \
    --augmented_dataset ../augmented_datasets/libero_spatial_oft.pkl \
    --save_name libero_spatial_oft_image \
    --use_transformer \
    --use_wandb 
