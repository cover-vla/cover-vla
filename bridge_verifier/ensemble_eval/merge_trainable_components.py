#!/usr/bin/env python3
"""
Script to extract and merge trainable components from multiple checkpoints.
Saves only the trainable components, excluding the frozen SigLIP encoder.
"""

import torch
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import ModelConfig
from finetune_trajectory_bridge_ddp import VLA_SigLIP2_Bridge
from open_clip import create_model_from_pretrained


def extract_trainable_components(model_path, backbone, use_transformer, device='cpu'):
    """
    Extract trainable components from a checkpoint
    
    Returns:
        dict: Dictionary containing trainable component state dicts
    """
    print(f"Loading checkpoint: {os.path.basename(model_path)}")
    
    # Load SigLIP model temporarily (just for initialization)
    siglip_model, _ = create_model_from_pretrained(backbone)
    siglip_model = siglip_model.to(device)
    siglip_model.eval()
    
    # Create model config
    model_config = ModelConfig(
        clip_model=siglip_model,
        history_length=10,  # Standard for Bridge
        action_dim=7
    )
    
    # Initialize full model
    full_model = VLA_SigLIP2_Bridge(model_config, use_transformer=use_transformer).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        full_model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_acc = checkpoint.get('val_img2act_top1_acc', 'N/A')
        print(f"  Epoch: {epoch}, Val Accuracy: {val_acc}")
    else:
        full_model.load_state_dict(checkpoint)
        print(f"  Loaded legacy format")
    
    # Extract trainable component state dicts
    trainable_state = {
        'text_aware_visual_extraction': full_model.text_aware_visual_extraction.state_dict(),
        'vision_poolings': full_model.vision_poolings.state_dict(),
        'text_pooling': full_model.text_pooling.state_dict(),
        'input_projection': full_model.input_projection.state_dict(),
        'action_padding_value': full_model.action_padding_value,
    }
    
    if use_transformer:
        trainable_state['single_step_action_encoder'] = full_model.single_step_action_encoder.state_dict()
        trainable_state['trajectory_encoder'] = full_model.trajectory_encoder.state_dict()
    else:
        trainable_state['complex_action_encoder'] = full_model.complex_action_encoder.state_dict()
    
    return trainable_state


def merge_checkpoints(model_paths, backbone, use_transformer, output_path, device='cpu'):
    """
    Merge trainable components from multiple checkpoints into a single file
    """
    print("="*60)
    print("Merging Trainable Components from Checkpoints")
    print("="*60)
    print(f"Number of models: {len(model_paths)}")
    print(f"Backbone: {backbone}")
    print(f"Use transformer: {use_transformer}")
    print(f"Output: {output_path}")
    print()
    
    ensemble_components = []
    
    for i, model_path in enumerate(model_paths):
        print(f"\n[Model {i+1}/{len(model_paths)}]")
        trainable_state = extract_trainable_components(model_path, backbone, use_transformer, device)
        ensemble_components.append(trainable_state)
    
    # Create merged checkpoint
    merged_checkpoint = {
        'ensemble_components': ensemble_components,
        'num_models': len(model_paths),
        'backbone': backbone,
        'use_transformer': use_transformer,
        'history_length': 10,
        'action_dim': 7,
        'source_checkpoints': [os.path.basename(p) for p in model_paths],
    }
    
    # Save merged checkpoint
    print(f"\n{'='*60}")
    print(f"Saving merged checkpoint...")
    torch.save(merged_checkpoint, output_path)
    
    # Calculate file sizes
    original_size = sum(os.path.getsize(p) for p in model_paths)
    merged_size = os.path.getsize(output_path)
    
    print(f"✅ Saved to: {output_path}")
    print(f"\nFile Size Comparison:")
    print(f"  Original checkpoints: {original_size / 1024**2:.1f} MB")
    print(f"  Merged file: {merged_size / 1024**2:.1f} MB")
    print(f"  Reduction: {(1 - merged_size/original_size)*100:.1f}%")
    print(f"  (Frozen encoder excluded from merged file)")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge trainable components from multiple checkpoints')
    
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints to merge')
    parser.add_argument('--backbone', type=str, default='hf-hub:timm/ViT-L-16-SigLIP2-384',
                       help='SigLIP backbone (must match training)')
    parser.add_argument('--use_transformer', action='store_true',
                       help='Models use transformer encoder')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for merged checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for loading (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Verify input files exist
    for model_path in args.model_paths:
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found: {model_path}")
            sys.exit(1)
    
    # Verify output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Merge checkpoints
    merge_checkpoints(
        model_paths=args.model_paths,
        backbone=args.backbone,
        use_transformer=args.use_transformer,
        output_path=args.output,
        device=args.device
    )


