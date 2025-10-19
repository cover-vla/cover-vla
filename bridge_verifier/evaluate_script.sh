#!/bin/bash
set -e

cd /root/vla-clip/bridge_verifier
mkdir -p logs

# Common parameters
DATASET="bridge_dataset_with_rephrases.json"
IMAGES_FOLDER="bridge_dataset_with_rephrases_images"
HISTORY_LENGTH=10
NUM_SAMPLES=50
ACTION_POOL_SIZE=64
USE_TRANSFORMER="--use_transformer"
INFERENCE_BATCH_SIZE=4096  # Batch size for processing action histories (set to empty for no batching)

# Model-to-backbone mapping
declare -A BACKBONES
BACKBONES["500m"]="hf-hub:timm/ViT-B-16-SigLIP2-384"
BACKBONES["1b"]="hf-hub:timm/ViT-L-16-SigLIP2-384"
BACKBONES["2b"]="hf-hub:timm/ViT-gopt-16-SigLIP2-384"
BACKBONES["bridge"]="hf-hub:timm/ViT-L-16-SigLIP2-384"  # bridge checkpoints are 1b model

# Number of GPUs to use
NUM_GPUS=4
GPU_ID=0

# Function to launch a single evaluation
launch_eval() {
    local ckpt=$1
    local gpu_id=$2
    local backbone=$3
    local ckpt_name=$(basename "$ckpt")
    local log_file="logs/eval_${ckpt_name%.pt}.log"

    echo "🚀 [GPU $gpu_id] Evaluating $ckpt_name with $backbone"
    echo "📝 Logging to $log_file"

    # Build command with optional inference batch size
    local cmd="CUDA_VISIBLE_DEVICES=$gpu_id python3 vla_siglip_inference_bridge.py \
        --model_path \"$ckpt\" \
        --bridge_dataset \"$DATASET\" \
        --images_folder \"$IMAGES_FOLDER\" \
        --history_length $HISTORY_LENGTH \
        --num_samples $NUM_SAMPLES \
        --action_pool_size $ACTION_POOL_SIZE \
        $USE_TRANSFORMER \
        --backbone \"$backbone\""
    
    # Add inference batch size if set
    if [ -n "$INFERENCE_BATCH_SIZE" ]; then
        cmd="$cmd --inference_batch_size $INFERENCE_BATCH_SIZE"
    fi
    
    # Execute command in background
    eval "$cmd > \"$log_file\" 2>&1 &"
}

# Print configuration
echo "================================"
echo "Evaluation Configuration:"
echo "================================"
echo "Dataset: $DATASET"
echo "Images folder: $IMAGES_FOLDER"
echo "Number of GPUs: $NUM_GPUS"
echo "Samples per model: $NUM_SAMPLES"
echo "Action pool size: $ACTION_POOL_SIZE"
echo "History length: $HISTORY_LENGTH"
if [ -n "$INFERENCE_BATCH_SIZE" ]; then
    echo "Inference batch size: $INFERENCE_BATCH_SIZE"
else
    echo "Inference batch size: Process all at once"
fi
echo "Use transformer: $USE_TRANSFORMER"
echo "================================"
echo ""

# Iterate through checkpoints
for ckpt in downloads/*.pt; do
    ckpt_name=$(basename "$ckpt")
    prefix=$(echo $ckpt_name | cut -d'_' -f1)
    backbone=${BACKBONES[$prefix]}

    if [ -z "$backbone" ]; then
        echo "⚠️  Skipping $ckpt (unknown prefix: $prefix)"
        continue
    fi

    # Assign GPU in round-robin fashion
    GPU_ID=$((GPU_ID % NUM_GPUS))

    launch_eval "$ckpt" "$GPU_ID" "$backbone"

    GPU_ID=$((GPU_ID + 1))
done

# Additional model (vla_clip variant) - commented out as checkpoint not in downloads folder
# echo "🚀 [GPU 0] Evaluating bridge_rephrases_epoch_20.pt with vla_clip_inference_bridge.py"
# LOG_FILE="logs/eval_bridge_rephrases_epoch_20.log"
# CUDA_VISIBLE_DEVICES=0 python3 vla_clip_inference_bridge.py \
#     --model_path bridge_rephrases_epoch_20.pt \
#     --bridge_dataset "$DATASET" \
#     --images_folder "$IMAGES_FOLDER" \
#     --history_length "$HISTORY_LENGTH" \
#     --num_samples "$NUM_SAMPLES" \
#     --action_pool_size "$ACTION_POOL_SIZE" \
#     $USE_TRANSFORMER \
#     > "$LOG_FILE" 2>&1 &

wait
echo "✅ All evaluations completed. Check logs/ for outputs."
