CUDA_VISIBLE_DEVICES=2 python ../scripts/finetune_trajectory.py \
    --epochs 300 \
    --batch_size 1024 \
    --lr 1e-4 \
    --history_length 10 \
    --augmented_dataset ../augmented_datasets/libero_spatial_pos_neg_globalstd_h10_padded.pkl \
    --save_name spatial_transformer_h10_neg0_padded_exp \
    --use_transformer \
    --neg_loss_weight 0 \
    --use_wandb 