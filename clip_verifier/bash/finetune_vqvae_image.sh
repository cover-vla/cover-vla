CUDA_VISIBLE_DEVICES=2 python ../scripts/finetune_vqvae_image.py \
    --epochs 20 \
    --batch_size 2048 \
    --lr 5e-5 \
    --history_length 5 \
    --augmented_dataset ../augmented_datasets/libero_spatial_pos_rephrase_neg_negation_5.pkl \
    --save_name libero_spatial_vqvae \
    --vqvae_checkpoint /root/vqvla_weights/action_tokenizer_weight/all_data_vq.pth \
    --resume /root/vla-clip/clip_verifier/bash/vqvae_checkpoints/libero_spatial_vqvae_epoch_3.pt \
    --use_wandb 


# CUDA_VISIBLE_DEVICES=2 python ../scripts/finetune_vqvae_image.py \
#     --epochs 2000 \
#     --batch_size 1024 \
#     --lr 5e-5 \
#     --history_length 8 \
#     --augmented_dataset ../augmented_datasets/libero_spatial_oft.pkl \
#     --save_name libero_spatial_oft_vqvae \
#     --vqvae_checkpoint /path/to/vqvla/action_tokenizer_weight/all_data_vq.pth \
#     --use_wandb  &