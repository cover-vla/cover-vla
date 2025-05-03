CUDA_VISIBLE_DEVICES=7 python ../scripts/finetune_trajectory.py \
    --epochs 200 \
    --batch_size 2048 \
    --lr 1e-4 \
    --history_length 10 \
    --augmented_dataset ../augmented_datasets/libero_all_pos_neg_globalstd_h10_padded.pkl \
    --save_name all_transformer_h10_neg0_padded \
    --use_transformer \
    --neg_loss_weight 0 \
    --use_wandb 