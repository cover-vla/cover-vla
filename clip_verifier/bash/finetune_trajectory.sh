
# CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_trajectory_dino.py \
#     --epochs 10 \
#     --batch_size 2048 \
#     --lr 5e-5 \
#     --history_length 10 \
#     --augmented_dataset ../augmented_datasets/libero_spatial_all.pkl \
#     --save_name libero_spatial_all \
#     --use_transformer \
#     --use_wandb 


CUDA_VISIBLE_DEVICES=2 python ../scripts/finetune_trajectory_dino.py \
    --epochs 10 \
    --batch_size 2048 \
    --lr 5e-5 \
    --history_length 10 \
    --augmented_dataset ../augmented_datasets/libero_spatial_oft_all.pkl \
    --save_name libero_spatial_oft_all \
    --use_transformer \
    --use_wandb