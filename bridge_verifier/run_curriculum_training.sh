#!/bin/bash

# Example script for running curriculum learning training
# This script demonstrates the gradual introduction of policy-in-the-loop data

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1  # Set available GPUs

# Paths
BRIDGE_DATASET="/root/vla-clip/bridge_verifier/10episodes.json"  # Pure Bridge dataset
POLICY_DATASET="/root/vla-clip/bridge_verifier/bridge_dataset_with_hard_negatives.json"  # Policy-in-the-loop dataset
IMAGES_FOLDER="/root/vla-clip/bridge_verifier/10episodes_imgs"
CHECKPOINT_DIR="/root/vla-clip/bridge_verifier/checkpoints_curriculum"

# Training Parameters
HISTORY_LENGTH=10
ACTION_DIM=7
EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=1e-6
WORLD_SIZE=2

# Curriculum Learning Parameters
BRIDGE_DOMINANCE_RATIO=0.8  # Minimum 80% Bridge data
MAX_POLICY_RATIO=0.5        # Maximum 50% policy data

# Model Configuration
USE_TRANSFORMER=false
SAVE_NAME="curriculum_bridge_vla_clip"

echo "Starting Curriculum Learning Training:"
echo "  Bridge Dataset: ${BRIDGE_DATASET}"
echo "  Policy Dataset: ${POLICY_DATASET}"
echo "  Images Folder: ${IMAGES_FOLDER}"
echo "  History Length: ${HISTORY_LENGTH}"
echo "  Action Dim: ${ACTION_DIM}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Bridge Dominance: ${BRIDGE_DOMINANCE_RATIO}"
echo "  Max Policy Ratio: ${MAX_POLICY_RATIO}"
echo "  Use Transformer: ${USE_TRANSFORMER}"

# Build command arguments
CURRICULUM_ARGS="--bridge_dataset ${BRIDGE_DATASET} \
    --policy_dataset ${POLICY_DATASET} \
    --images_folder ${IMAGES_FOLDER} \
    --history_length ${HISTORY_LENGTH} \
    --action_dim ${ACTION_DIM} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --validation_split 0.1 \
    --save_name ${SAVE_NAME} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --bridge_dominance_ratio ${BRIDGE_DOMINANCE_RATIO} \
    --max_policy_ratio ${MAX_POLICY_RATIO} \
    --world_size ${WORLD_SIZE} \
    --port 12355"

# Add transformer flag if enabled
if [ "$USE_TRANSFORMER" = true ]; then
    CURRICULUM_ARGS="$CURRICULUM_ARGS --use_transformer"
fi

# Run curriculum learning training
python finetune_trajectory_bridge_curriculum_ddp.py $CURRICULUM_ARGS

echo "Curriculum Learning Training completed!"

# Example usage notes:
echo ""
echo "Curriculum Learning Schedule:"
echo "  Epochs 1-10:  100% Bridge data (pure expert demonstrations)"
echo "  Epochs 11-25: 100% → 70% Bridge, 0% → 30% Policy (gradual introduction)"
echo "  Epochs 26-50: 70% → 50% Bridge, 30% → 50% Policy (maintain Bridge dominance)"
echo ""
echo "This approach ensures:"
echo "  1. Strong foundation from expert demonstrations"
echo "  2. Gradual adaptation to policy distribution"
echo "  3. Bridge data remains dominant throughout training"
echo "  4. Conservative hard negative filtering"
