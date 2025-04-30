CUDA_VISIBLE_DEVICES=4 python ../scripts/vla_clip_inference.py \
    --model_path /home/xilun/vla-clip/clip_verifier/bash/trajectory_checkpoints/spatial_transformer_h10_neg2_padded_final_best.pt \
    --history_length 10 \
    --use_transformer \
    --augmented_dataset ../augmented_datasets/libero_spatial_pos_neg_globalstd_h10_padded.pkl \
    --num_samples 100 \
    --action_pool_size 50