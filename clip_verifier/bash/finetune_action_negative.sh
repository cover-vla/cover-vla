

CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_negative.py \
    --epochs 200 \
    --batch_size 1024 \
    --lr 2e-4 \
    --save_name spatial-action-negative \
    --augmented_dataset ../augmented_datasets/libero_spatial_all.pkl \
    --use_wandb



