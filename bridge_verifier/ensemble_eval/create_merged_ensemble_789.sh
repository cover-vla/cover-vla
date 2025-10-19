#!/bin/bash
# Script to create merged ensemble checkpoint from epochs 7, 8, 9

cd /root/vla-clip/bridge_verifier

echo "========================================"
echo "Creating Merged Ensemble Checkpoint"
echo "========================================"
echo "Models: epochs 7, 8, 9"
echo ""

python3 ensemble_eval/merge_trainable_components.py \
  --model_paths \
    downloads/bridge_4096_6e5_64_epoch_7_trainloss_2.8373_valloss_1.7805.pt \
    downloads/bridge_4096_6e5_64_epoch_8_trainloss_2.3440_valloss_1.4271.pt \
    downloads/bridge_4096_6e5_64_epoch_9_trainloss_1.9012_valloss_1.0189.pt \
  --backbone hf-hub:timm/ViT-L-16-SigLIP2-384 \
  --use_transformer \
  --output downloads/ensemble_789_trainable_only.pt \
  --device cpu

echo ""
echo "========================================"
echo "✅ Merged checkpoint created!"
echo "Location: downloads/ensemble_789_trainable_only.pt"
echo "========================================"


