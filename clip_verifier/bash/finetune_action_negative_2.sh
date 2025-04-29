

CUDA_VISIBLE_DEVICES=4 python ../scripts/finetune_negative.py \
    --epochs 200 \
    --batch_size 1024 \
    --lr 5e-4 \
    --save_name spatial-action-negative-gaussian-negative-only \
    --augmented_dataset ../augmented_datasets/libero_spatial_gaussian.pkl \
    --loss_type negative_only \
    --use_wandb



