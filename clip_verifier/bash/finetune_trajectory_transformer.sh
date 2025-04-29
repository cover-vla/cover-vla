
CUDA_VISIBLE_DEVICES=3 python ../scripts/finetune_trajectory.py \
    --epochs 600 \
    --batch_size 2048 \
    --lr 5e-5 \
    --history_length 10 \
    --augmented_dataset ../augmented_datasets/libero_spatial_pos_neg_globalstd_h10_padded.pkl \
    --save_name spatial_transformer_h10_neg2_padded \
    --use_transformer \
    --resume /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/spatial_transformer_h10_neg2_padded_final_best.pt \
    --use_wandb 
    # --use_transformer

