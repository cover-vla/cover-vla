CUDA_VISIBLE_DEVICES=3 python ../scripts/finetune_trajectory.py \
    --epochs 150 \
    --batch_size 128 \
    --lr 7e-5 \
    --save_name clip_trajectory_tranasformer_only_last_token_augmented_dataset \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --dataset_folders libero_spatial \
    --augmented_dataset ./augmented_dataset.pkl \
    --use_transformer \
    --use_wandb
#     # --checkpoint_dir model_checkpoints \
#     # --resume test_checkpoints/clip_trajectory_2_layer_action_encoder_only_last_token.pt \


CUDA_VISIBLE_DEVICES=3 python ../scripts/finetune_trajectory.py \
    --epochs 150 \
    --batch_size 128 \
    --lr 7e-5 \
    --save_name clip_trajectory_action_encoder_only_augmented_dataset \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --dataset_folders libero_spatial \
    --augmented_dataset ./augmented_dataset.pkl \
    --use_wandb \


CUDA_VISIBLE_DEVICES=3 python ../scripts/finetune_trajectory.py \
    --epochs 150 \
    --batch_size 128 \
    --lr 7e-5 \
    --save_name clip_trajectory_tranasformer_only_last_token_positive_only_augmented_dataset \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --dataset_folders libero_spatial \
    --augmented_dataset ./augmented_dataset_positive_only.pkl \
    --use_transformer \
    --use_wandb
#     # --checkpoint_dir model_checkpoints \
#     # --resume test_checkpoints/clip_trajectory_2_layer_action_encoder_only_last_token.pt \


CUDA_VISIBLE_DEVICES=3 python ../scripts/finetune_trajectory.py \
    --epochs 150 \
    --batch_size 128 \
    --lr 7e-5 \
    --save_name clip_trajectory_action_encoder_only_positive_only_augmented_dataset \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --dataset_folders libero_spatial \
    --augmented_dataset ./augmented_dataset_positive_only.pkl \
    --use_wandb \