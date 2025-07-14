
python ../scripts/finetune_trajectory_dino.py \
    --epochs 20 \
    --batch_size 1024 \
    --lr 5e-5 \
    --history_length 10 \
    --augmented_dataset ../augmented_datasets/libero_spatial_all_diverse.pkl \
    --save_name libero_spatial_all_diverse \
    --use_transformer \
    --use_wandb 


# CUDA_VISIBLE_DEVICES=2 python ../scripts/finetune_trajectory_dino.py \
#     --epochs 2000 \
#     --batch_size 1024 \
#     --lr 5e-5 \
#     --history_length 8 \
#     --augmented_dataset ../augmented_datasets/libero_spatial_oft.pkl \
#     --save_name libero_spatial_oft \
#     --use_transformer \
#     --use_wandb  &