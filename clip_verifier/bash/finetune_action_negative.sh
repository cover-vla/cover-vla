

CUDA_VISIBLE_DEVICES=1 python ../scripts/finetune_negative.py \
    --epochs 20 \
    --batch_size 512 \
    --lr 2e-4 \
    --save_name All_clip_action_negative \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --augmented_dataset ../augmented_datasets/libero_all_positive.pkl \
    --use_wandb



