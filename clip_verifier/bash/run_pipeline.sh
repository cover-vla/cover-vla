
## Generate augmented dataset 
python ../scripts/augment_dataset.py \
    --dataset_path /home/xilun/LIBERO/libero/datasets \
    --dataset_folders libero_spatial \
    --output_path augmented_dataset_positive_only.pkl


CUDA_VISIBLE_DEVICES=3 python ../scripts/finetune_trajectory.py \
    --epochs 150 \
    --batch_size 128 \
    --lr 7e-5 \
    --save_name clip_trajectory_action_encoder_only_positive_only_augmented_dataset \
    --augmented_dataset ./augmented_dataset_positive_only.pkl \
    --use_wandb \
    # These would be used if we want to finetune on the original dataset
    # --dataset_path /home/xilun/LIBERO/libero/datasets \
    # --dataset_folders libero_spatial 

python ../scripts/vla_clip_inference.py \
    --model_path model_checkpoints/clip_trajectory_action_encoder_only_positive_only_augmented_dataset_final.pt \
    --trajectory_mode \
    --augmented_path ../augmented_dataset_positive_only.pkl