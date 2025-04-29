
CUDA_VISIBLE_DEVICES=4 python ../scripts/finetune_trajectory.py \
    --epochs 300 \
    --batch_size 2048 \
    --lr 5e-5 \
    --history_length 10 \
    --augmented_dataset ../augmented_datasets/libero_spatial_pos_neg_globalstd_h10_padded.pkl \
    --save_name spatial_mlp_h10_neg2_padded \
    --use_wandb
    # --use_transformer

