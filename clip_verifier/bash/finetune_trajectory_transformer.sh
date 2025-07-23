
# python ../scripts/finetune_trajectory_dino.py \
#     --epochs 20 \
#     --batch_size 1024 \
#     --lr 5e-5 \
#     --history_length 10 \
#     --augmented_dataset ../augmented_datasets/libero_spatial_all_diverse.pkl \
#     --save_name libero_spatial_all_diverse \
#     --use_transformer \
#     --resume /root/vla-clip/clip_verifier/bash/trajectory_checkpoints/libero_spatial_all_diverse_epoch_6.pt \
#     --use_wandb 


CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_trajectory_image.py \
    --epochs 20 \
    --batch_size 1024 \
    --lr 5e-5 \
    --history_length 10 \
    --augmented_dataset ../augmented_datasets/libero_spatial_all_diverse.pkl \
    --save_name libero_spatial_all_diverse_clip \
    --use_transformer \
    --use_wandb 