
# CUDA_VISIBLE_DEVICES=3 python ../scripts/finetune_trajectory.py \
#     --epochs 600 \
#     --batch_size 2048 \
#     --lr 5e-5 \
#     --history_length 10 \
#     --augmented_dataset ../augmented_datasets/libero_spatial_pos_neg_globalstd_h10_padded.pkl \
#     --save_name spatial_transformer_h10_neg2_padded \
#     --use_transformer \
#     --resume /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/spatial_transformer_h10_neg2_padded_final_best.pt \
#     --use_wandb 


# CUDA_VISIBLE_DEVICES=5 python ../scripts/finetune_trajectory.py \
#     --epochs 200 \
#     --batch_size 2048 \
#     --lr 5e-5 \
#     --history_length 10 \
#     --augmented_dataset ../augmented_datasets/libero_all_pos_neg_globalstd_h10_padded.pkl \
#     --save_name all_transformer_h10_neg2_padded \
#     --use_transformer \
#     --resume /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/all_transformer_h10_neg2_padded_final_best.pt \
#     --use_wandb 

CUDA_VISIBLE_DEVICES=1 python ../scripts/finetune_trajectory_image.py \
    --epochs 3000 \
    --batch_size 4096 \
    --lr 5e-5 \
    --history_length 10 \
    --augmented_dataset ../augmented_datasets/libero_spatial_augmented_dataset.pkl \
    --save_name libero_spatial \
    --use_transformer \
    # --use_wandb 