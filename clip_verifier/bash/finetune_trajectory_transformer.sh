
CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_trajectory.py \
    --epochs 3 \
    --batch_size 512 \
    --lr 7e-5 \
    --save_name all_clip_trajectory_action_transformer \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --augmented_dataset ../augmented_datasets/libero_all.pkl \
    --use_transformer \
    --use_wandb \

