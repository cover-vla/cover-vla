# CUDA_VISIBLE_DEVICES=5 python ../scripts/vla_clip_inference.py \
#     --model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/libero_spatial_epoch_500.pt \
#     --history_length 10 \
#     --use_transformer \
#     --augmented_dataset ../augmented_datasets/libero_spatial_augmented_dataset.pkl \
#     --num_samples 100 \
#     --action_pool_size 50


CUDA_VISIBLE_DEVICES=1 python ../scripts/vla_dino_inference.py \
    --model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/libero_spatial_all_epoch_6.pt \
    --history_length 10 \
    --use_transformer \
    --augmented_dataset ../augmented_datasets/libero_spatial_all.pkl \
    --num_samples 100 \
    --action_pool_size 50