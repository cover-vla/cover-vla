#!/usr/bin/env python3
"""
Curriculum Learning Training Script for VLA-CLIP Bridge Model
Implements gradual introduction of policy-in-the-loop data while maintaining Bridge dataset dominance.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import clip
from PIL import Image
from typing import Optional, List, Dict, Tuple
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from model import TextAwareVisualExtraction, AttentionPooling, ModelConfig
import numpy as np
import ijson
import json
import warnings
import time
import math
from collections import defaultdict

class CurriculumBridgeDataset(Dataset):
    def __init__(self, bridge_dataset_dict, policy_dataset_dict, history_length, 
                 images_folder, current_epoch=0, total_epochs=50, 
                 bridge_dominance_ratio=0.8, max_policy_ratio=0.5):
        """
        Curriculum learning dataset that gradually introduces policy data.
        
        Args:
            bridge_dataset_dict: Pure Bridge dataset (unbiased)
            policy_dataset_dict: Policy-in-the-loop dataset (may have bias)
            history_length: Expected length of action histories
            images_folder: Path to folder containing agent view images
            current_epoch: Current training epoch
            total_epochs: Total number of training epochs
            bridge_dominance_ratio: Minimum ratio of Bridge data (0.8 = 80%)
            max_policy_ratio: Maximum ratio of policy data (0.5 = 50%)
        """
        self.history_length = history_length
        self.images_folder = images_folder
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        self.bridge_dominance_ratio = bridge_dominance_ratio
        self.max_policy_ratio = max_policy_ratio
        
        # Load Bridge dataset (primary, unbiased)
        self.bridge_samples = self._load_bridge_dataset(bridge_dataset_dict)
        
        # Load policy dataset (secondary, for distribution matching)
        self.policy_samples = self._load_policy_dataset(policy_dataset_dict) if policy_dataset_dict else []
        
        # Get CLIP's image preprocessing pipeline
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                     (0.26862954, 0.26130258, 0.27577711))
        ])
        
        print(f"Curriculum Dataset Created:")
        print(f"  Bridge samples: {len(self.bridge_samples)}")
        print(f"  Policy samples: {len(self.policy_samples)}")
        print(f"  Current epoch: {current_epoch}/{total_epochs}")
        print(f"  Bridge ratio: {self._get_bridge_ratio():.2f}, Policy ratio: {self._get_policy_ratio():.2f}")
    
    def _get_bridge_ratio(self):
        """Calculate current Bridge data ratio based on curriculum schedule"""
        progress = self.current_epoch / self.total_epochs
        
        if progress < 0.2:  # First 20% of training: 100% Bridge
            return 1.0
        elif progress < 0.5:  # Next 30%: Gradually introduce policy (100% -> 70%)
            return 1.0 - 0.3 * ((progress - 0.2) / 0.3)
        else:  # Remaining 50%: Maintain Bridge dominance (70% -> 50%)
            return max(self.bridge_dominance_ratio, 
                      0.7 - 0.2 * ((progress - 0.5) / 0.5))
    
    def _get_policy_ratio(self):
        """Calculate current policy data ratio"""
        return 1.0 - self._get_bridge_ratio()
    
    def _load_bridge_dataset(self, bridge_dataset_dict):
        """Load Bridge dataset with instruction rephrasing but same actions"""
        # Store references to lookup tables (like original implementation)
        self.bridge_action_histories = bridge_dataset_dict.get('action_histories', {})
        self.bridge_instructions = bridge_dataset_dict.get('instructions', {})
        
        # Handle different Bridge dataset formats
        metadata = bridge_dataset_dict.get('_metadata', {})
        format_version = metadata.get('format_version', '1.0_legacy')
        
        if format_version in ['3.0_with_hard_negatives', '2.0_normalized']:
            samples = self._load_normalized_bridge_format(bridge_dataset_dict)
        else:
            samples = self._load_legacy_bridge_format(bridge_dataset_dict)
        
        print(f"Loaded {len(samples)} Bridge samples")
        return samples
    
    def _load_normalized_bridge_format(self, bridge_dataset_dict):
        """Load normalized Bridge format"""
        samples = []
        action_histories = bridge_dataset_dict['action_histories']
        instructions = bridge_dataset_dict['instructions']
        samples_data = bridge_dataset_dict['samples']
        
        for sample_data in samples_data:
            action_history_id = sample_data.get('action_history_id')
            agent_view_image_file = sample_data.get('agent_view_image_file')
            positives = sample_data.get('positives', [])
            
            if not all([action_history_id, agent_view_image_file]):
                continue
            
            # Validate action history shape
            if len(action_histories[action_history_id]) != self.history_length:
                continue
            
            # Get all positive instructions (rephrased versions of same task)
            positive_instructions = []
            for pos_id in positives:
                if pos_id in instructions:
                    positive_instructions.append(instructions[pos_id])
            
            if positive_instructions:
                samples.append({
                    'action_history_id': action_history_id,
                    'agent_view_image_file': agent_view_image_file,
                    'positive_instructions': positive_instructions,
                    'hard_negatives': [],  # No hard negatives for Bridge data
                    'sample_id': sample_data.get('sample_id', len(samples)),
                    'episode_id': sample_data.get('episode_id', -1),
                    'timestep': sample_data.get('timestep', -1),
                    'data_source': 'bridge'  # Mark as Bridge data
                })
        
        return samples
    
    def _load_legacy_bridge_format(self, bridge_dataset_dict):
        """Load legacy Bridge format"""
        samples = []
        
        for instruction, data in bridge_dataset_dict.items():
            if instruction == '_metadata':
                continue
            
            instruction_samples = data.get('samples', [])
            for sample_data in instruction_samples:
                agent_view_image_file = sample_data.get('agent_view_image_file')
                action_hist = sample_data.get('action_history')
                lang_instruction = sample_data.get('language_instruction')
                
                if not all([agent_view_image_file, action_hist, lang_instruction]):
                    continue
                
                if len(action_hist) != self.history_length:
                    continue
                
                samples.append({
                    'agent_view_image_file': agent_view_image_file,
                    'action_history': action_hist,
                    'positive_instructions': [lang_instruction],
                    'hard_negatives': [],
                    'sample_id': len(samples),
                    'data_source': 'bridge'
                })
        
        return samples
    
    def _load_policy_dataset(self, policy_dataset_dict):
        """Load policy-in-the-loop dataset with filtered hard negatives"""
        # Store references to lookup tables (like original implementation)
        self.policy_action_histories = policy_dataset_dict.get('action_histories', {})
        self.policy_instructions = policy_dataset_dict.get('instructions', {})
        
        samples = []
        action_histories = policy_dataset_dict['action_histories']
        instructions = policy_dataset_dict['instructions']
        samples_data = policy_dataset_dict['samples']
        
        for sample_data in samples_data:
            action_history_id = sample_data.get('action_history_id')
            agent_view_image_file = sample_data.get('agent_view_image_file')
            positives = sample_data.get('positives', [])
            hard_negatives = sample_data.get('hard_negatives', [])
            
            if not all([action_history_id, agent_view_image_file]):
                continue
            
            if len(action_histories[action_history_id]) != self.history_length:
                continue
            
            # Validate that all instruction IDs exist (but don't load them yet - lazy loading)
            valid_sample = True
            for pos_id in positives:
                if pos_id not in instructions:
                    valid_sample = False
                    break
            
            if valid_sample:
                for neg_data in hard_negatives:
                    if neg_data['instruction_id'] not in instructions:
                        valid_sample = False
                        break
            
            if valid_sample:
                samples.append({
                    'action_history_id': action_history_id,
                    'agent_view_image_file': agent_view_image_file,
                    'positives': positives,  # Store raw positive IDs for lazy loading
                    'hard_negatives': hard_negatives,  # Store raw hard negatives for lazy loading
                    'sample_id': sample_data.get('sample_id', len(samples)),
                    'episode_id': sample_data.get('episode_id', -1),
                    'timestep': sample_data.get('timestep', -1),
                    'data_source': 'policy'  # Mark as policy data
                })
        
        print(f"Loaded {len(samples)} policy samples (hard negatives will be filtered during training)")
        return samples
    
    def _filter_hard_negatives(self, hard_negatives, instructions, current_sample):
        """
        Select meaningful hard negatives based on semantic similarity and action differences.
        Only use negatives that represent semantically meaningful confusions.
        """
        filtered = []
        
        for neg_data in hard_negatives:
            neg_id = neg_data['instruction_id']
            similarity = neg_data.get('similarity', 0.0)
            error = neg_data.get('error', 1.0)
            
            if neg_id not in instructions:
                continue
            
            neg_instruction = instructions[neg_id]
            
            # Only use as hard negative if it represents a meaningful semantic confusion:
            # 1. Same task (high similarity) but different execution (high error)
            # 2. Different episodes (different tasks) with low similarity
            
            if similarity > 0.8 and error > 0.3:
                # Same semantic task but different action execution
                # This represents good instruction + bad execution vs good instruction + good execution
                filtered.append({
                    'instruction': neg_instruction,
                    'similarity': similarity,
                    'error': error,
                    'confidence': similarity * error,  # High similarity + high error = good hard negative
                    'type': 'same_task_different_execution'
                })
            elif similarity < 0.4 and neg_data.get('episode_id', -1) != current_sample.get('episode_id', -1):
                # Different semantic task (different episode)
                # This represents different tasks entirely
                filtered.append({
                    'instruction': neg_instruction,
                    'similarity': similarity,
                    'error': error,
                    'confidence': 1.0 - similarity,  # Low similarity = good different task negative
                    'type': 'different_task'
                })
        
        # Sort by confidence and take top 3 most meaningful negatives
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        return filtered[:3]
    
    def update_epoch(self, epoch):
        """Update current epoch for curriculum scheduling"""
        self.current_epoch = epoch
    
    def __len__(self):
        return len(self.bridge_samples) + len(self.policy_samples)
    
    def __getitem__(self, idx):
        # Determine data source based on curriculum schedule
        bridge_ratio = self._get_bridge_ratio()
        
        if idx < len(self.bridge_samples):
            # Bridge sample
            sample_info = self.bridge_samples[idx]
        else:
            # Policy sample (if available and within ratio)
            policy_idx = idx - len(self.bridge_samples)
            if policy_idx < len(self.policy_samples) and np.random.random() < self._get_policy_ratio():
                sample_info = self.policy_samples[policy_idx]
            else:
                # Fallback to Bridge sample
                sample_info = self.bridge_samples[idx % len(self.bridge_samples)]
        
        # Load data based on format (FIXED - proper lazy loading like original)
        if 'action_history_id' in sample_info:
            # Normalized format - load from proper dictionaries
            action_history_id = sample_info['action_history_id']
            if sample_info['data_source'] == 'bridge':
                action_hist = np.array(self.bridge_action_histories[action_history_id])
                # Get positive instructions from bridge instructions
                positive_instructions = []
                for pos_id in sample_info['positives']:
                    if pos_id in self.bridge_instructions:
                        positive_instructions.append(self.bridge_instructions[pos_id])
                sample_info['positive_instructions'] = positive_instructions
            else:
                action_hist = np.array(self.policy_action_histories[action_history_id])
                # Get positive instructions from policy instructions
                positive_instructions = []
                for pos_id in sample_info['positives']:
                    if pos_id in self.policy_instructions:
                        positive_instructions.append(self.policy_instructions[pos_id])
                sample_info['positive_instructions'] = positive_instructions
        else:
            # Legacy format
            action_hist = np.array(sample_info['action_history'])
        
        # Load image
        image_path = os.path.join(self.images_folder, sample_info['agent_view_image_file'])
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        img = self.preprocess(img)
        
        # Process hard negatives (only for policy data)
        hard_negative_data = []
        if sample_info.get('data_source') == 'policy' and 'hard_negatives' in sample_info:
            # Filter hard negatives conservatively
            filtered_hard_negatives = self._filter_hard_negatives(
                sample_info['hard_negatives'], self.policy_instructions, sample_info
            )
            
            # Convert filtered hard negatives to the format expected by training
            for filtered_neg in filtered_hard_negatives:
                hard_negative_data.append({
                    'instruction': filtered_neg['instruction'],
                    'similarity': filtered_neg['similarity'],
                    'error': filtered_neg['error']
                })
        
        return {
            'image': img,
            'positive_instructions': sample_info['positive_instructions'],
            'hard_negative_data': hard_negative_data,
            'action_history': action_hist,
            'sample_id': sample_info.get('sample_id', idx),
            'data_source': sample_info.get('data_source', 'bridge')
        }

# Import the model classes from the original file
from finetune_trajectory_bridge_ddp_with_hard_negatives import (
    VLA_CLIP_Bridge_HardNegatives,
    weighted_infonce_loss_with_hard_negatives,
    collate_hard_negatives_batch,
    setup_distributed,
    cleanup_distributed
)

def train_curriculum_clip_bridge_ddp(
    rank: int,
    world_size: int,
    bridge_dataset_path: str,
    policy_dataset_path: Optional[str],
    history_length: int,
    action_dim: int,
    images_folder: str,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-6,
    validation_split: float = 0.1,
    save_name: str = None,
    checkpoint_dir: str = "checkpoints_curriculum",
    use_wandb: bool = False,
    resume_checkpoint: str = None,
    use_transformer: bool = False,
    port: int = 12355,
    bridge_dominance_ratio: float = 0.8,
    max_policy_ratio: float = 0.5
):
    """Curriculum learning training function"""
    
    # Setup distributed training
    setup_distributed(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    
    # Initialize wandb only on rank 0
    if use_wandb and rank == 0:
        import wandb
        if save_name is None:
            save_name = f"vla_clip_bridge_curriculum_h{history_length}_{'transformer' if use_transformer else 'mlp'}_ddp"
        
        wandb.init(project="VLA-CLIP-Bridge-Curriculum", name=save_name)
        wandb.config.update({
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "bridge_dominance_ratio": bridge_dominance_ratio,
            "max_policy_ratio": max_policy_ratio,
            "use_transformer": use_transformer,
        })
    
    # Create checkpoint directory
    if rank == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if save_name is None:
            save_name = f"vla_clip_bridge_curriculum_h{history_length}_{'transformer' if use_transformer else 'mlp'}_ddp"
    
    # Load datasets
    if rank == 0:
        print(f"Loading Bridge dataset from {bridge_dataset_path}...")
    
    # Load Bridge dataset (primary)
    bridge_dataset_dict = load_dataset_with_streaming(bridge_dataset_path)
    
    # Load policy dataset (optional, secondary)
    policy_dataset_dict = None
    if policy_dataset_path and os.path.exists(policy_dataset_path):
        if rank == 0:
            print(f"Loading policy dataset from {policy_dataset_path}...")
        policy_dataset_dict = load_dataset_with_streaming(policy_dataset_path)
    else:
        if rank == 0:
            print("No policy dataset provided, using Bridge-only training")
    
    # Create curriculum dataset
    curriculum_dataset = CurriculumBridgeDataset(
        bridge_dataset_dict=bridge_dataset_dict,
        policy_dataset_dict=policy_dataset_dict,
        history_length=history_length,
        images_folder=images_folder,
        current_epoch=0,
        total_epochs=num_epochs,
        bridge_dominance_ratio=bridge_dominance_ratio,
        max_policy_ratio=max_policy_ratio
    )
    
    # Clear datasets from memory
    del bridge_dataset_dict, policy_dataset_dict
    
    # Train/validation split
    dataset_size = len(curriculum_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        curriculum_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, collate_fn=collate_hard_negatives_batch
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=1, pin_memory=True, collate_fn=collate_hard_negatives_batch
    )
    
    # Load CLIP model and create VLA model
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model_config = ModelConfig(clip_model=clip_model, history_length=history_length, action_dim=action_dim)
    model = VLA_CLIP_Bridge_HardNegatives(model_config, use_transformer=use_transformer).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Cosine annealing scheduler
    total_train_steps = len(train_dataloader) * num_epochs
    warmup_steps = len(train_dataloader) * 10  # 10 epoch warmup
    
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(optimizer, lr_lambda)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_train_steps)
    
    # Training loop with curriculum learning
    best_val_loss = float('inf')
    if rank == 0:
        best_model_path = os.path.join(checkpoint_dir, f"{save_name}_best.pt")
        epoch_pbar = tqdm(range(num_epochs), desc="Curriculum Training")
    
    for epoch in range(num_epochs):
        # Update curriculum schedule
        curriculum_dataset.update_epoch(epoch)
        
        # Set epoch for DistributedSampler
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_batch_count = 0
        
        if rank == 0:
            train_batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} (Train)", leave=False)
        else:
            train_batch_pbar = train_dataloader
        
        for batch_idx, batch_data in enumerate(train_batch_pbar):
            images = batch_data['images'].to(device)
            positive_instructions = batch_data['positive_instructions']
            hard_negative_data = batch_data['hard_negative_data']
            action_histories = batch_data['action_histories'].to(device)
            sample_indices = batch_data['sample_indices']
            
            # Tokenize positive instructions
            tokenized_positives = clip.tokenize(positive_instructions, truncate=True).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            images_for_positives = images[sample_indices]
            action_histories_for_positives = action_histories[sample_indices]
            
            combined_features, action_features = model(images_for_positives, tokenized_positives, action_histories_for_positives)
            
            # Prepare hard negatives (simplified version)
            hard_negative_features_list = []
            hard_negative_weights_list = []
            
            for i in range(len(positive_instructions)):
                if i < len(hard_negative_data) and hard_negative_data[i]:
                    # Process hard negatives if available
                    hn_features = []
                    hn_weights = []
                    for neg in hard_negative_data[i]:
                        # Simplified: just use similarity as weight
                        hn_weights.append(torch.tensor(neg.get('similarity', 0.5), device=device))
                    
                    if hn_features:
                        hard_negative_features_list.append(torch.stack(hn_features))
                        hard_negative_weights_list.append(torch.stack(hn_weights))
                    else:
                        hard_negative_features_list.append(torch.empty(0, combined_features.shape[1], device=device))
                        hard_negative_weights_list.append(torch.empty(0, device=device))
                else:
                    hard_negative_features_list.append(torch.empty(0, combined_features.shape[1], device=device))
                    hard_negative_weights_list.append(torch.empty(0, device=device))
            
            # Compute loss
            logit_scale = model.module.logit_scale.exp()
            loss, loss_metrics = weighted_infonce_loss_with_hard_negatives(
                combined_features, action_features, hard_negative_features_list,
                hard_negative_weights_list, logit_scale, device, alpha=1.0
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_batch_count += 1
            
            if rank == 0:
                train_batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'bridge_ratio': f'{curriculum_dataset._get_bridge_ratio():.2f}',
                    'policy_ratio': f'{curriculum_dataset._get_policy_ratio():.2f}'
                })
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_data in val_dataloader:
                # Similar validation logic as training
                images = batch_data['images'].to(device)
                positive_instructions = batch_data['positive_instructions']
                action_histories = batch_data['action_histories'].to(device)
                sample_indices = batch_data['sample_indices']
                
                tokenized_positives = clip.tokenize(positive_instructions, truncate=True).to(device)
                images_for_positives = images[sample_indices]
                action_histories_for_positives = action_histories[sample_indices]
                
                combined_features, action_features = model(images_for_positives, tokenized_positives, action_histories_for_positives)
                
                # Simplified validation loss (no hard negatives for speed)
                logit_scale = model.module.logit_scale.exp()
                positive_logits = logit_scale * torch.sum(action_features * combined_features, dim=1)
                
                # Simple InfoNCE loss for validation
                all_logits = logit_scale * torch.matmul(action_features, combined_features.T)
                mask = torch.eye(len(positive_instructions), device=device, dtype=torch.bool)
                negative_logits = all_logits.masked_fill(mask, float('-inf'))
                
                logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
                val_loss = F.cross_entropy(logits, torch.zeros(len(positive_instructions), dtype=torch.long, device=device))
                
                total_val_loss += val_loss.item()
                val_batch_count += 1
        
        # Calculate averages
        avg_train_loss = total_train_loss / train_batch_count
        avg_val_loss = total_val_loss / val_batch_count
        
        # Log metrics
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            bridge_ratio = curriculum_dataset._get_bridge_ratio()
            policy_ratio = curriculum_dataset._get_policy_ratio()
            
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'bridge_ratio': f'{bridge_ratio:.2f}',
                'policy_ratio': f'{policy_ratio:.2f}',
                'lr': f'{current_lr:.2e}'
            })
            
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "learning_rate": current_lr,
                    "curriculum/bridge_ratio": bridge_ratio,
                    "curriculum/policy_ratio": policy_ratio,
                })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.module.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Synchronize all processes
        dist.barrier(device_ids=[rank])
    
    # Cleanup
    if use_wandb and rank == 0:
        wandb.finish()
    
    cleanup_distributed()
    return model.module if rank == 0 else None

def load_dataset_with_streaming(json_path):
    """Load dataset from JSON file with streaming support"""
    print(f"Loading dataset from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        return dataset
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        raise

def ddp_main(rank, world_size, args):
    """Main function for each DDP process"""
    model = train_curriculum_clip_bridge_ddp(
        rank=rank,
        world_size=world_size,
        bridge_dataset_path=args.bridge_dataset,
        policy_dataset_path=args.policy_dataset,
        history_length=args.history_length,
        action_dim=args.action_dim,
        images_folder=args.images_folder,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validation_split=args.validation_split,
        save_name=args.save_name,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        resume_checkpoint=args.resume,
        use_transformer=args.use_transformer,
        port=args.port,
        bridge_dominance_ratio=args.bridge_dominance_ratio,
        max_policy_ratio=args.max_policy_ratio
    )
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Curriculum Learning Training for VLA-CLIP Bridge Model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (per GPU)')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of data to use for validation')
    
    # Model parameters
    parser.add_argument('--history_length', type=int, required=True, help='Action history length')
    parser.add_argument('--action_dim', type=int, default=7, help='Action dimension')
    parser.add_argument('--use_transformer', action='store_true', help='Use transformer for action encoding')
    
    # Dataset paths
    parser.add_argument('--bridge_dataset', type=str, required=True, help='Path to Bridge dataset JSON file')
    parser.add_argument('--policy_dataset', type=str, default=None, help='Path to policy dataset JSON file (optional)')
    parser.add_argument('--images_folder', type=str, required=True, help='Path to folder containing agent view images')
    
    # Curriculum learning parameters
    parser.add_argument('--bridge_dominance_ratio', type=float, default=0.8, help='Minimum ratio of Bridge data (0.8 = 80%)')
    parser.add_argument('--max_policy_ratio', type=float, default=0.5, help='Maximum ratio of policy data (0.5 = 50%)')
    
    # Other parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_curriculum', help='Directory to save checkpoints')
    parser.add_argument('--save_name', type=str, default=None, help='Name for saved model and wandb run')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--port', type=int, default=12355, help='Port for distributed communication')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.world_size > torch.cuda.device_count():
        print(f"Error: Requested {args.world_size} GPUs but only {torch.cuda.device_count()} available.")
        exit(1)
    
    if not os.path.exists(args.bridge_dataset):
        print(f"Error: Bridge dataset file not found at {args.bridge_dataset}")
        exit(1)
    
    if not os.path.exists(args.images_folder):
        print(f"Error: Images folder not found at {args.images_folder}")
        exit(1)
    
    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    print("Starting Curriculum Learning Training...")
    print(f"Bridge Dataset: {args.bridge_dataset}")
    print(f"Policy Dataset: {args.policy_dataset or 'None (Bridge-only)'}")
    print(f"Bridge Dominance Ratio: {args.bridge_dominance_ratio}")
    print(f"Max Policy Ratio: {args.max_policy_ratio}")
    print(f"Epochs: {args.epochs}")
    print(f"World Size: {args.world_size}")
    
    # Spawn processes for DDP training
    if args.world_size == 1:
        print("Running on single GPU (no multiprocessing)")
        finetuned_model = ddp_main(0, 1, args)
    else:
        print(f"Spawning {args.world_size} processes for DDP training")
        mp.spawn(ddp_main, args=(args.world_size, args), nprocs=args.world_size, join=True)
        finetuned_model = None
    
    print("Curriculum Learning Training completed!")
