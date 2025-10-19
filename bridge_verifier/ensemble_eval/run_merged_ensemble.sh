#!/bin/bash
# Script to run inference with merged ensemble checkpoint

# cd /root/vla-clip/bridge_verifier

echo "========================================"
echo "Efficient Ensemble Inference"
echo "(Using Merged Checkpoint)"
echo "========================================"
echo ""

python3 ensemble_eval/efficient_ensemble_merged.py \
  --merged_checkpoint downloads/ensemble_789_trainable_only.pt \
  --bridge_dataset bridge_dataset_with_rephrases.json \
  --images_folder bridge_dataset_with_rephrases_images \
  --num_samples 50 \
  --action_pool_size 20

echo ""
echo "========================================"
echo "✅ Evaluation complete!"
echo "========================================"

CUDA_VISIBLE_DEVICES=2 python3 efficient_ensemble_merged.py \
  --merged_checkpoint ../ensemble_789_trainable_only.pt \
  --bridge_dataset ../bridge_dataset_with_rephrases.json \
  --images_folder ../bridge_dataset_with_rephrases_images \
  --num_samples 50 \
  --action_pool_size 20