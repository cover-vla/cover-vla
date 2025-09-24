import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
# Remove pickle import, add ijson for streaming JSON
import ijson
import json
import warnings
import time
import math

class BridgeDatasetWithHardNegatives(Dataset):
    def __init__(self, augmented_dataset_dict, history_length, images_folder, world_size=1, rank=0):
        """
        Memory-efficient Dataset for Bridge V2 with hard negatives and DDP support
        
        Args:
            augmented_dataset_dict: Dictionary loaded from the augmented Bridge dataset JSON file with hard negatives.
            history_length: Expected length of action histories (H).
            images_folder: Path to folder containing the agent view images as JPG files.
            world_size: Number of DDP processes (for sharding).
            rank: Current process rank (for sharding).
        """
        self.history_length = history_length
        self.images_folder = images_folder
        self.world_size = world_size
        self.rank = rank
        
        # Store references instead of loading all data into memory
        self.action_histories = None
        self.instructions = None
        self.sample_indices = []
        
        # Check if this is the new hard negatives format
        metadata = augmented_dataset_dict.get('_metadata', {})
        format_version = metadata.get('format_version', '1.0_legacy')
        
        if format_version == '3.0_with_hard_negatives':
            print(f"Loading hard negatives dataset format (rank {rank})...")
            self._load_hard_negatives_format_efficient(augmented_dataset_dict)
        elif format_version == '2.0_normalized':
            print(f"Loading optimized normalized dataset format (rank {rank})...")
            self._load_normalized_format_efficient(augmented_dataset_dict)
        else:
            print(f"Loading legacy dataset format (rank {rank})...")
            self._load_legacy_format_efficient(augmented_dataset_dict)

        print(f"Process {rank}: Created dataset with {len(self.sample_indices)} samples (DDP shard).")
        if not self.sample_indices:
            raise ValueError("Dataset creation resulted in 0 samples. Check input data format and history length.")

        # Get CLIP's image preprocessing pipeline
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                     (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def _load_hard_negatives_format_efficient(self, augmented_dataset_dict):
        """Memory-efficient loading with hard negatives and DDP sharding"""
        # Store references to lookup tables
        self.action_histories = augmented_dataset_dict['action_histories']
        self.instructions = augmented_dataset_dict['instructions']
        samples_data = augmented_dataset_dict['samples']
        
        total_samples = len(samples_data)
        print(f"Total samples: {total_samples:,}, Processing shard for rank {self.rank}/{self.world_size}")
        
        # Calculate shard boundaries for this rank
        samples_per_rank = total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        
        if self.rank == self.world_size - 1:
            # Last rank takes any remaining samples
            end_idx = total_samples
        else:
            end_idx = start_idx + samples_per_rank
        
        print(f"Rank {self.rank}: Processing samples {start_idx:,} to {end_idx:,}")
        
        # Only process this rank's shard
        valid_count = 0
        for i in tqdm(range(start_idx, end_idx), desc=f"Rank {self.rank} processing samples"):
            sample_data = samples_data[i]
            action_history_id = sample_data.get('action_history_id')
            agent_view_image_file = sample_data.get('agent_view_image_file')
            positives = sample_data.get('positives', [])
            hard_negatives = sample_data.get('hard_negatives', [])
            
            if not all([action_history_id, agent_view_image_file]):
                continue
                
            # Quick validation without loading full data
            if action_history_id not in self.action_histories:
                continue
            
            # Validate that all instruction IDs exist
            valid_sample = True
            for pos_id in positives:
                if pos_id not in self.instructions:
                    valid_sample = False
                    break
            
            if valid_sample:
                for neg_data in hard_negatives:
                    if neg_data['instruction_id'] not in self.instructions:
                        valid_sample = False
                        break
            
            if not valid_sample:
                continue
            
            # Validate action history shape quickly
            action_hist_shape = len(self.action_histories[action_history_id])
            if action_hist_shape != self.history_length:
                continue
            
            # Store the sample index and data for lazy loading
            self.sample_indices.append({
                'idx': i,
                'action_history_id': action_history_id,
                'agent_view_image_file': agent_view_image_file,
                'positives': positives,
                'hard_negatives': hard_negatives,
                'sample_id': sample_data.get('sample_id', i),
                'episode_id': sample_data.get('episode_id', -1),
                'timestep': sample_data.get('timestep', -1)
            })
            valid_count += 1
        
        print(f"Rank {self.rank}: {valid_count:,} valid samples from {end_idx - start_idx:,} processed")
    
    def _load_normalized_format_efficient(self, augmented_dataset_dict):
        """Memory-efficient loading with DDP sharding (fallback for older format)"""
        # Store references to lookup tables
        self.action_histories = augmented_dataset_dict['action_histories']
        self.instructions = augmented_dataset_dict['instructions']
        samples_data = augmented_dataset_dict['samples']
        
        total_samples = len(samples_data)
        print(f"Total samples: {total_samples:,}, Processing shard for rank {self.rank}/{self.world_size}")
        
        # Calculate shard boundaries for this rank
        samples_per_rank = total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        
        if self.rank == self.world_size - 1:
            # Last rank takes any remaining samples
            end_idx = total_samples
        else:
            end_idx = start_idx + samples_per_rank
        
        print(f"Rank {self.rank}: Processing samples {start_idx:,} to {end_idx:,}")
        
        # Only process this rank's shard
        valid_count = 0
        for i in tqdm(range(start_idx, end_idx), desc=f"Rank {self.rank} processing samples"):
            sample_data = samples_data[i]
            action_history_id = sample_data.get('action_history_id')
            instruction_id = sample_data.get('instruction_id')
            agent_view_image_file = sample_data.get('agent_view_image_file')
            
            if not all([action_history_id, instruction_id, agent_view_image_file]):
                continue
                
            # Quick validation without loading full data
            if action_history_id not in self.action_histories or instruction_id not in self.instructions:
                continue
            
            # Validate action history shape quickly
            action_hist_shape = len(self.action_histories[action_history_id])
            if action_hist_shape != self.history_length:
                continue
            
            # Store the sample index and IDs for lazy loading (convert to hard negatives format)
            self.sample_indices.append({
                'idx': i,
                'action_history_id': action_history_id,
                'agent_view_image_file': agent_view_image_file,
                'positives': [instruction_id],  # Single positive instruction
                'hard_negatives': [],  # No hard negatives in this format
                'sample_id': i,
                'episode_id': sample_data.get('episode_id', -1),
                'timestep': sample_data.get('timestep', -1)
            })
            valid_count += 1
        
        print(f"Rank {self.rank}: {valid_count:,} valid samples from {end_idx - start_idx:,} processed")
    
    def _load_legacy_format_efficient(self, augmented_dataset_dict):
        """Memory-efficient legacy format loading with DDP sharding"""
        # For legacy format, we need to flatten and shard differently
        all_samples = []
        
        print(f"Rank {self.rank}: Collecting legacy format samples...")
        for instruction, data in augmented_dataset_dict.items():
            if instruction == '_metadata':
                continue
                
            instruction_samples = data.get('samples', [])
            for sample_data in instruction_samples:
                agent_view_image_file = sample_data.get('agent_view_image_file')
                action_hist = sample_data.get('action_history')
                lang_instruction = sample_data.get('language_instruction')

                if agent_view_image_file is None or action_hist is None or lang_instruction is None:
                    continue
                
                # Quick validation
                if len(action_hist) != self.history_length:
                    continue
                
                all_samples.append({
                    'agent_view_image_file': agent_view_image_file,
                    'language_instruction': lang_instruction,
                    'action_history': action_hist,
                    'positives': [lang_instruction],  # Single positive instruction
                    'hard_negatives': []  # No hard negatives in legacy format
                })
        
        # Shard the samples
        total_samples = len(all_samples)
        samples_per_rank = total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        
        if self.rank == self.world_size - 1:
            end_idx = total_samples
        else:
            end_idx = start_idx + samples_per_rank
        
        print(f"Rank {self.rank}: Taking samples {start_idx:,} to {end_idx:,} from {total_samples:,}")
        
        # Store only this rank's samples
        for i in range(start_idx, end_idx):
            self.sample_indices.append(all_samples[i])

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        sample_info = self.sample_indices[idx]
        
        if isinstance(sample_info, dict) and 'action_history_id' in sample_info:
            # Hard negatives or normalized format - lazy loading
            action_history_id = sample_info['action_history_id']
            agent_view_image_file = sample_info['agent_view_image_file']
            positives = sample_info['positives']
            hard_negatives = sample_info['hard_negatives']
            
            # Load data on demand
            action_hist = np.array(self.action_histories[action_history_id])
            
            # Get positive instructions
            positive_instructions = []
            for pos_id in positives:
                if pos_id in self.instructions:
                    positive_instructions.append(self.instructions[pos_id])
            
            # Get hard negative instructions with their metadata
            hard_negative_data = []
            for neg_data in hard_negatives:
                neg_id = neg_data['instruction_id']
                if neg_id in self.instructions:
                    hard_negative_data.append({
                        'instruction': self.instructions[neg_id],
                        'positive_instruction_id': neg_data['positive_instruction_id'],
                        'similarity': neg_data['similarity'],
                        'error': neg_data['error']
                    })
            
        else:
            # Legacy format - data already loaded
            agent_view_image_file = sample_info['agent_view_image_file']
            positive_instructions = sample_info.get('positives', [sample_info.get('language_instruction', '')])
            hard_negative_data = sample_info.get('hard_negatives', [])
            action_hist = np.array(sample_info['action_history'])
        
        # Load image from file
        image_path = os.path.join(self.images_folder, agent_view_image_file)
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
            
        img = self.preprocess(img)
        
        return {
            'image': img,
            'positive_instructions': positive_instructions,
            'hard_negative_data': hard_negative_data,
            'action_history': action_hist,
            'sample_id': sample_info.get('sample_id', idx)
        }


class VLA_CLIP_Bridge_HardNegatives(nn.Module):
    def __init__(self, model_config, use_transformer=False):
        super().__init__()
        self.clip = model_config.clip_model
        # 1) Freeze everything
        for param in self.clip.parameters():
            param.requires_grad = False
            param.data = param.data.float()
            
        text_pooling_output_dim = model_config.text_pooling_output_dim
        pooling_heads = model_config.pooling_heads
        pooling_layers = model_config.pooling_layers
        self.num_readouts = model_config.num_readouts
        text_dim = self.clip.text_projection.shape[1]
        vision_dim = self.clip.visual.output_dim
        vision_pooling_output_dim = model_config.vision_pooling_output_dim
        
        self.visual_patch_size = self.clip.visual.conv1.kernel_size[0]
        self.num_img_patches = (224 // self.visual_patch_size) ** 2
        
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        
        # The number of patches depends on the CLIP model's vision transformer
        # For ViT-B/32, the image is divided into 7x7=49 patches (for 224x224 images)
        self.text_aware_visual_extraction = TextAwareVisualExtraction(
            num_img_patches=self.num_img_patches,  # For ViT-B/32
            vision_dim=vision_dim,
        )
        # Components
        self.text_pooling = AttentionPooling(
            text_dim, 
            text_pooling_output_dim,
            pooling_heads,
            pooling_layers, 
            num_readouts=self.num_readouts,
        )        
        self.vision_poolings = AttentionPooling(
            vision_dim,
            vision_pooling_output_dim,
            pooling_heads, 
            pooling_layers, 
            num_readouts=self.num_readouts
        )
        
        self.f_t_dim = text_pooling_output_dim + vision_pooling_output_dim
        
        self.input_projection = nn.Linear(self.f_t_dim, vision_pooling_output_dim)
        
        # Action trajectory processing components
        self.action_dim = model_config.action_dim
        self.history_length = model_config.history_length
        self.use_transformer = use_transformer

        if self.use_transformer:
            # Transformer expects input features per step
            self.single_step_action_encoder = nn.Linear(self.action_dim, vision_pooling_output_dim)
            # --- REMOVED batch_first=True ---
            encoder_layer = nn.TransformerEncoderLayer(
                    d_model=vision_pooling_output_dim,
                    nhead=8, # Ensure vision_pooling_output_dim is divisible by nhead
                    dim_feedforward=vision_pooling_output_dim * 2, # Common practice
                    # batch_first=True, # REMOVED! Default is False
                    dropout=0.1
                )
            self.trajectory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        else:
            # MLP processes flattened trajectory
            self.complex_action_encoder = nn.Sequential(
                nn.Linear(self.history_length * self.action_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, vision_pooling_output_dim)
            )

        # Hooks for extracting intermediate features (remain the same)
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook

        self.hooks = [
            self.clip.visual.transformer.resblocks[-1].attn.register_forward_hook(
                get_activation('image_patches')
            ),
            self.clip.transformer.register_forward_hook(
                get_activation('text_features')
            )
        ]
        
        # Add a placeholder for padding value if needed, e.g. for mask generation
        self.action_padding_value = -5.0 # Or another distinct value like -5.0

    def extract_clip_features(self, images, text):
        """
        Extract patch-level features from CLIP model
        
        Args:
            images: Tensor of shape (batch_size, C, H, W)
            text: Tokenized text of shape (batch_size, max_text_len)
            
        Returns:
            patch_features: Tensor of shape (batch_size, num_patches, embedding_dim)
            text_features: Tensor of shape (batch_size, num_tokens, embedding_dim)
        """
        
        # Forward pass through CLIP
        with torch.no_grad():
            _ = self.clip.encode_text(text)
            _ = self.clip.encode_image(images)
        text_features = self.activation['text_features'].permute(1, 0, 2)
        text_features = self.clip.ln_final(text_features).type(self.clip.dtype) @ self.clip.text_projection
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Process patch features from activations
        patch_features = self.activation['image_patches'][0]
        patch_features = patch_features.permute(1, 0, 2)[:, 1:, :]  # Shape: (batch_size, num_patches, embedding_dim)
        if hasattr(self.clip.visual, 'proj') and self.clip.visual.proj is not None:
            patch_features = patch_features @ self.clip.visual.proj
    
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        return patch_features, text_features
        
    def forward(self, image, text, action_histories):
        """
        Args:
            image: Tensor (B, C, H, W) - Single agent view image
            text: Tensor (B, SeqLen) - tokenized text
            action_histories: Tensor (B, H, D) - Batch of action histories, potentially padded.
        """
        # Extract image/text features
        patch_features, text_features = self.extract_clip_features(image, text)
        
        # Text-aware visual features
        text_aware_features = self.text_aware_visual_extraction(patch_features, text_features)
        vision_token = self.vision_poolings(text_aware_features)
        
        # Text pooling
        text_token = self.text_pooling(text_features)
        combined_features = torch.cat([text_token, vision_token], dim=-1)
        combined_features = self.input_projection(combined_features)
        combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
        
        # --- Encode Action History ---
        action_histories = action_histories.float().to(image.device) # Shape (B, H, D)

        if self.use_transformer:
            # --- Transformer Path (batch_first=False) ---
            # 1. Create Padding Mask: True where padded. Shape (B, H)
            padding_mask = (action_histories[:, :, 0] == self.action_padding_value)

            # 2. Encode each step. Shape: (B, H, D) -> (B, H, E)
            encoded_steps = self.single_step_action_encoder(action_histories)

            # 3. Permute for Transformer Encoder: (B, H, E) -> (H, B, E)
            encoded_steps_permuted = encoded_steps.permute(1, 0, 2)

            # 4. Pass through transformer encoder with mask. Mask shape remains (B, H).
            # Input: (H, B, E), Mask: (B, H) -> Output: (H, B, E)
            transformer_output_permuted = self.trajectory_encoder(encoded_steps_permuted, src_key_padding_mask=padding_mask)

            # 5. Permute output back: (H, B, E) -> (B, H, E)
            transformer_output = transformer_output_permuted.permute(1, 0, 2)

            # 6. Pool features (Masked Mean Pooling) - Apply mask to (B, H, E) output
            mask_expanded = (~padding_mask).unsqueeze(-1).float() # Shape (B, H, 1)
            summed_features = (transformer_output * mask_expanded).sum(dim=1) # Shape (B, E)
            num_non_padded = mask_expanded.sum(dim=1) # Shape (B, 1)
            num_non_padded = torch.clamp(num_non_padded, min=1e-9)
            projected_trajectory = summed_features / num_non_padded # Shape (B, E)
            # --- End Transformer Path ---

        else:
            # --- MLP Path (No changes needed here) ---
            batch_size = action_histories.shape[0]
            flat_actions = action_histories.reshape(batch_size, -1)
            projected_trajectory = self.complex_action_encoder(flat_actions)
            # --- End MLP Path ---

        # Normalize action history features
        projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)

        return combined_features, projected_trajectory


def weighted_infonce_loss_with_hard_negatives(
    image_text_features,      # (B, D) - combined image+text features
    action_features,          # (B, D) - action trajectory features  
    hard_negative_features,   # List of (N_i, D) tensors - hard negatives for each sample
    hard_negative_weights,    # List of (N_i,) tensors - weights for hard negatives
    logit_scale,              # Scalar - learned temperature parameter
    device,
    alpha=1.0                 # Weight for hard negatives vs in-batch negatives
):
    """
    Compute weighted InfoNCE loss with hard negatives in the denominator.
    
    Args:
        image_text_features: (B, D) - combined image+text features
        action_features: (B, D) - action trajectory features
        hard_negative_features: List of (N_i, D) tensors - hard negatives for each sample i
        hard_negative_weights: List of (N_i,) tensors - similarity-based weights for hard negatives
        logit_scale: Scalar - learned temperature parameter
        device: torch device
        alpha: Weight for hard negatives vs in-batch negatives
        
    Returns:
        loss: Scalar tensor
        metrics: Dict with accuracy and loss components
    """
    batch_size = image_text_features.shape[0]
    
    # Compute positive logits (action -> image+text)
    positive_logits = logit_scale * torch.sum(action_features * image_text_features, dim=1)  # (B,)
    
    # Compute in-batch negative logits
    # action_features[i] vs image_text_features[j] for all j != i
    all_logits = logit_scale * torch.matmul(action_features, image_text_features.T)  # (B, B)
    
    # Create mask to exclude positive pairs (diagonal)
    mask = torch.eye(batch_size, device=device, dtype=torch.bool)
    in_batch_negative_logits = all_logits.masked_fill(mask, float('-inf'))  # (B, B)
    
    total_losses = []
    correct_predictions = 0
    
    for i in range(batch_size):
        # Get in-batch negatives for sample i
        in_batch_negatives = in_batch_negative_logits[i]  # (B,) with -inf at position i
        
        # Start with in-batch negatives
        all_negative_logits = in_batch_negatives[~mask[i]]  # (B-1,)
        
        # Add hard negatives if available
        if i < len(hard_negative_features) and hard_negative_features[i].shape[0] > 0:
            # Compute hard negative logits
            hard_neg_logits = logit_scale * torch.matmul(
                action_features[i:i+1],  # (1, D)
                hard_negative_features[i].T  # (D, N_i)
            ).squeeze(0)  # (N_i,)
            
            # Apply weights to hard negatives
            if i < len(hard_negative_weights) and hard_negative_weights[i].shape[0] > 0:
                weighted_hard_neg_logits = hard_neg_logits + torch.log(alpha * hard_negative_weights[i])
            else:
                weighted_hard_neg_logits = hard_neg_logits + torch.log(torch.tensor(alpha, device=device))
            
            # Combine in-batch and hard negatives
            all_negative_logits = torch.cat([all_negative_logits, weighted_hard_neg_logits])
        
        # Compute InfoNCE loss for sample i using log-sum-exp trick
        # log(exp(pos) / (exp(pos) + sum(exp(negs)))) = pos - log(exp(pos) + sum(exp(negs)))
        all_logits_for_sample = torch.cat([positive_logits[i:i+1], all_negative_logits])
        
        # Use log-sum-exp for numerical stability
        log_sum_exp = torch.logsumexp(all_logits_for_sample, dim=0)
        loss_i = log_sum_exp - positive_logits[i]
        total_losses.append(loss_i)
        
        # Check if positive is ranked highest
        if positive_logits[i] > torch.max(all_negative_logits):
            correct_predictions += 1
    
    # Average loss across batch
    loss = torch.stack(total_losses).mean()
    
    # Compute accuracy
    accuracy = correct_predictions / batch_size
    
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy,
        'positive_logits_mean': positive_logits.mean().item(),
        'positive_logits_std': positive_logits.std().item()
    }
    
    return loss, metrics


def collate_hard_negatives_batch(batch):
    """
    Custom collate function to handle variable numbers of positives and hard negatives.
    
    Args:
        batch: List of sample dictionaries from dataset
        
    Returns:
        Collated batch with proper handling of variable-length sequences
    """
    images = torch.stack([item['image'] for item in batch])
    action_histories = torch.stack([torch.as_tensor(item['action_history'], dtype=torch.float32) for item in batch])
    
    # Handle multiple positive instructions per sample
    all_positive_instructions = []
    all_hard_negative_data = []
    sample_indices = []  # Track which sample each instruction belongs to
    
    for i, item in enumerate(batch):
        positives = item['positive_instructions']
        hard_negatives = item['hard_negative_data']
        
        # For each positive instruction, create a separate training instance
        for pos_instr in positives:
            all_positive_instructions.append(pos_instr)
            all_hard_negative_data.append(hard_negatives)  # All hard negatives apply to each positive
            sample_indices.append(i)
    
    return {
        'images': images,
        'positive_instructions': all_positive_instructions,
        'hard_negative_data': all_hard_negative_data,
        'action_histories': action_histories,
        'sample_indices': sample_indices,
        'batch_size': len(batch),
        'total_positives': len(all_positive_instructions)
    }


def setup_distributed(rank, world_size, port=12355):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Set the device before initializing the process group
    torch.cuda.set_device(rank)
    
    # Initialize process group with device_id to avoid NCCL warnings
    dist.init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}")
    )


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def calculate_accuracy_metrics(logits_per_image, logits_per_action, batch_size, device, k_values=[1, 5]):
    """Calculate top-k accuracy metrics for both directions"""
    metrics = {}
    
    # Create ground truth labels
    labels = torch.arange(batch_size, device=device)
    
    # Image-to-Action accuracy (image query, action retrieval)
    for k in k_values:
        if k <= batch_size:
            _, topk_indices = torch.topk(logits_per_image, k, dim=1)
            correct = topk_indices.eq(labels.view(-1, 1).expand_as(topk_indices))
            accuracy = correct.any(dim=1).float().mean().item()
            metrics[f'img2act_top{k}_acc'] = accuracy
    
    # Action-to-Image accuracy (action query, image retrieval)
    for k in k_values:
        if k <= batch_size:
            _, topk_indices = torch.topk(logits_per_action, k, dim=1)
            correct = topk_indices.eq(labels.view(-1, 1).expand_as(topk_indices))
            accuracy = correct.any(dim=1).float().mean().item()
            metrics[f'act2img_top{k}_acc'] = accuracy
    
    return metrics


def get_gpu_metrics(device):
    """Get GPU memory and utilization metrics"""
    if torch.cuda.is_available() and device.type == 'cuda':
        gpu_id = device.index
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3  # GB
        
        return {
            'gpu_memory_allocated_gb': allocated,
            'gpu_memory_reserved_gb': reserved,
            'gpu_memory_max_allocated_gb': max_allocated,
            'gpu_id': gpu_id
        }
    return {}


def calculate_gradient_metrics(model):
    """Calculate gradient norm and related metrics"""
    total_norm = 0
    param_count = 0
    grad_count = 0
    
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += p.numel()
            grad_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    return {
        'grad_norm': total_norm,
        'grad_param_count': param_count,
        'grad_layer_count': grad_count
    }


def train_clip_bridge_hard_negatives_ddp(
    rank: int,
    world_size: int,
    dataset_path: str,
    history_length: int,
    action_dim: int,
    images_folder: str,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-6,
    validation_split: float = 0.1,
    save_name = None,
    checkpoint_dir = "checkpoints",
    use_wandb = False,
    resume_checkpoint = None,
    use_transformer = False,
    port = 12355,
    warmup_epochs = 10,
    train_log_freq = 50,
    eval_log_freq = 500,
    hard_negative_alpha = 1.0,
    use_bf16 = None,
    cosine_cycles = 0.5
):
    """DDP training function with hard negatives support"""
    # Setup distributed training
    setup_distributed(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    
    # Initialize wandb only on rank 0
    if use_wandb and rank == 0:
        import wandb
        # Ensure save_name is set if using wandb
        if save_name is None:
             save_name = f"vla_clip_bridge_hn_h{history_length}_{'transformer' if use_transformer else 'mlp'}_ddp"
             print(f"Generated save_name for wandb: {save_name}")

        wandb.init(project="VLA-CLIP-Bridge-HardNegatives-DDP", name=save_name)
        wandb.config.update({
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "world_size": world_size,
            "device": f"cuda:{rank}",
            "history_length": history_length,
            "action_dim": action_dim,
            "use_transformer": use_transformer,
            "validation_split": validation_split,
            "warmup_epochs": warmup_epochs,
            "train_log_freq": train_log_freq,
            "eval_log_freq": eval_log_freq,
            "hard_negative_alpha": hard_negative_alpha,
        })

    # Create checkpoint directory if it doesn't exist (only on rank 0)
    if rank == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if save_name is None:
            save_name = f"vla_clip_bridge_hn_h{history_length}_{'transformer' if use_transformer else 'mlp'}_ddp"

    # Ensure all processes have the save_name
    dist.barrier(device_ids=[rank])

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    
    # Load the CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    
    # Create model configuration for Bridge, passing history_length and action_dim
    model_config = ModelConfig(clip_model=clip_model, history_length=history_length, action_dim=action_dim)
    
    # Initialize the model with bfloat16 support
    model = VLA_CLIP_Bridge_HardNegatives(model_config, use_transformer=use_transformer).to(device)
    
    # Convert model to bfloat16 for efficiency (except for loss computation)
    if use_bf16 is None:
        # Auto-detect bfloat16 support
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    if use_bf16:
        model = model.to(dtype=torch.bfloat16)
        if rank == 0:
            print("Using bfloat16 mixed precision training")
    else:
        model = model.float()
        if rank == 0:
            print("Using float32 training")
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Initialize GradScaler for mixed precision 
    # Note: bfloat16 doesn't need gradient scaling, only float16 does
    scaler = torch.amp.GradScaler('cuda', enabled=False)  # Always disabled for bfloat16
    
    # Note: scheduler will be created after we have train_dataloader
    scheduler = None
    start_epoch = 0
    global_step = 0

    # Load checkpoint if specified (all processes load independently)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Rank {rank}: Loading model state dict from {resume_checkpoint}")
        try:
            # Each process loads the checkpoint onto its own device
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            # The model is wrapped in DDP, so access the underlying module
            model.module.load_state_dict(checkpoint)
            print(f"Rank {rank}: Successfully loaded model weights.")
        except Exception as load_err:
            print(f"Rank {rank}: Error loading checkpoint: {load_err}. Starting from scratch.")
            start_epoch = 0
        
        # Use a barrier to ensure all processes have loaded the model before proceeding
        dist.barrier(device_ids=[rank])

    # Print model size and details (only on rank 0)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Load dataset in each process to avoid memory duplication
    if rank == 0:
        print(f"Loading dataset from {dataset_path} on each process...")
    
    try:
        # Each process loads the dataset independently to avoid massive memory usage
        augmented_dataset_dict = load_dataset_with_streaming(dataset_path)
        
        # Extract metadata for action_dim if needed (only print on rank 0)
        if rank == 0:
            metadata = augmented_dataset_dict.get('_metadata', {})
            if metadata:
                format_version = metadata.get('format_version', '1.0_legacy')
                print(f"Dataset format version: {format_version}")
                
                if format_version == '3.0_with_hard_negatives':
                    total_samples = metadata.get('total_samples', 0)
                    samples_with_hard_negatives = metadata.get('samples_with_hard_negatives', 0)
                    total_hard_negatives = metadata.get('total_hard_negatives', 0)
                    print(f"Total samples in dataset: {total_samples:,}")
                    print(f"Samples with hard negatives: {samples_with_hard_negatives:,}")
                    print(f"Total hard negative instances: {total_hard_negatives:,}")
        
        # Create dataset with DDP sharding
        dataset = BridgeDatasetWithHardNegatives(
            augmented_dataset_dict, 
            history_length=history_length, 
            images_folder=images_folder,
            world_size=world_size,
            rank=rank
        )
        
        # Clear the dictionary from memory after dataset creation to save memory
        del augmented_dataset_dict
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # Report memory usage after dataset loading (only on rank 0)
        if rank == 0 and torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory after dataset loading: {memory_gb:.2f} GB")
        
    except Exception as e:
        if rank == 0:
            print(f"Error loading/creating dataset: {e}")
        cleanup_distributed()
        return None

    # Train/Validation Split
    dataset_size = len(dataset)
    if dataset_size == 0:
        if rank == 0:
            print("Error: Dataset is empty. Exiting.")
        cleanup_distributed()
        return None
        
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    if val_size <= 0 and dataset_size > 0:
        val_size = max(1, int(0.1 * dataset_size))
        train_size = dataset_size - val_size
        if rank == 0:
            print(f"Adjusted validation size to {val_size} due to small dataset.")

    if train_size <= 0:
        if rank == 0:
            print(f"Error: No training samples after split (Dataset size: {dataset_size}, Val size: {val_size}). Exiting.")
        cleanup_distributed()
        return None

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    if rank == 0:
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

    # Create data loaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Can shuffle since each process has its own shard
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_hard_negatives_batch
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,  # Smaller batch size for validation to handle hard negatives
        shuffle=False, 
        num_workers=1,  # Reduced for evaluation to prevent worker abortion
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_hard_negatives_batch
    )
    
    # Create cosine annealing scheduler with linear warmup
    total_train_steps = len(train_dataloader) * num_epochs
    warmup_steps = len(train_dataloader) * warmup_epochs
    
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
        """
        Create a schedule with a learning rate that decreases following the values of the cosine function between the
        initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
        initial lr set in the optimizer.
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            # Cosine annealing after warmup
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(optimizer, lr_lambda, last_epoch)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
        num_cycles=cosine_cycles
    )

    # Training loop
    best_val_loss = float('inf')
    if rank == 0:
        best_model_path = os.path.join(checkpoint_dir, f"{save_name}_best.pt")
        epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Training epochs")
        print(f"Warmup steps: {warmup_steps}, Total training steps: {total_train_steps}")
        print(f"Training log frequency: every {train_log_freq} steps")
        print(f"Evaluation frequency: every {eval_log_freq} steps")
        print(f"Hard negative alpha: {hard_negative_alpha}")
    else:
        epoch_pbar = range(start_epoch, num_epochs)

    for epoch in epoch_pbar:
        
        # Training Phase
        model.train()
        epoch_start_time = time.time()
        
        # Training metrics
        total_train_loss = 0
        total_weighted_infonce_loss = 0
        total_grad_norm = 0
        train_batch_count = 0
        
        # Accuracy tracking
        total_hard_neg_accuracy = 0
        
        if rank == 0:
            train_batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} (Train)", leave=False)
        else:
            train_batch_pbar = train_dataloader

        for batch_idx, batch_data in enumerate(train_batch_pbar):
            batch_start_time = time.time()
            
            images = batch_data['images'].to(device)  # (B, C, H, W)
            positive_instructions = batch_data['positive_instructions']  # List of strings
            hard_negative_data = batch_data['hard_negative_data']  # List of lists
            action_histories = batch_data['action_histories'].to(device)  # (B, H, D)
            sample_indices = batch_data['sample_indices']  # List mapping positives to samples
            
            # Tokenize positive instructions
            tokenized_positives = clip.tokenize(positive_instructions, truncate=True).to(device)
            
            current_batch_size = images.shape[0]
            total_positives = len(positive_instructions)
            
            optimizer.zero_grad()
            
            # Forward pass through model with mixed precision - FIXED: Match dimensions using sample_indices
            # Use sample_indices to map each positive instruction to its corresponding image/action
            images_for_positives = images[sample_indices]  # (total_positives, C, H, W)
            action_histories_for_positives = action_histories[sample_indices]  # (total_positives, H, D)
            
            with torch.amp.autocast('cuda', enabled=use_bf16, dtype=torch.bfloat16):
                combined_features, action_features = model(images_for_positives, tokenized_positives, action_histories_for_positives)
            
            # Prepare hard negatives for weighted InfoNCE (EFFICIENT INDEXING VERSION)
            hard_negative_features_list = []
            hard_negative_weights_list = []
            
            # 1. Gather all hard negative texts and their source indices
            all_hn_texts = []
            hn_source_indices = []  # Maps each hard negative to its original batch index (0 to B-1)
            hard_neg_weights_flat = []
            hard_neg_batch_map = []  # To map features back to positives
            
            for i in range(total_positives):
                source_idx = sample_indices[i]  # Original batch index (0 to B-1)
                hard_negs = hard_negative_data[i]
                
                if hard_negs:
                    count = len(hard_negs)
                    for hn in hard_negs:
                        all_hn_texts.append(hn['instruction'])
                        hn_source_indices.append(source_idx)  # Map to original sample
                        hard_neg_weights_flat.append(hn['similarity'])
                    hard_neg_batch_map.append(count)
                else:
                    hard_neg_batch_map.append(0)
            
            # 2. Run a single forward pass for all hard negatives if any exist
            if all_hn_texts:
                # Tokenize all hard negative texts in a single batch
                tokenized_all_hn = clip.tokenize(all_hn_texts, truncate=True).to(device)
                
                # Use efficient indexing to select corresponding images and actions
                images_for_hn = images[hn_source_indices]  # (num_total_hard_negatives, C, H, W)
                actions_for_hn = action_histories[hn_source_indices]  # (num_total_hard_negatives, H, D)
                
                # SINGLE forward pass for all hard negatives with mixed precision
                with torch.amp.autocast('cuda', enabled=use_bf16, dtype=torch.bfloat16):
                    all_hn_features, _ = model(images_for_hn, tokenized_all_hn, actions_for_hn)
                
                # 3. Distribute features back to positives
                current_pos = 0
                for i in range(total_positives):
                    count = hard_neg_batch_map[i]
                    if count > 0:
                        features = all_hn_features[current_pos : current_pos + count]
                        weights = torch.tensor(hard_neg_weights_flat[current_pos : current_pos + count], device=device)
                        hard_negative_features_list.append(features)
                        hard_negative_weights_list.append(weights)
                        current_pos += count
                    else:
                        # Handle samples with no hard negatives
                        hard_negative_features_list.append(torch.empty(0, combined_features.shape[1], device=device))
                        hard_negative_weights_list.append(torch.empty(0, device=device))
            else:
                # Handle batch with no hard negatives at all
                for i in range(total_positives):
                    hard_negative_features_list.append(torch.empty(0, combined_features.shape[1], device=device))
                    hard_negative_weights_list.append(torch.empty(0, device=device))
            
            # Compute weighted InfoNCE loss with hard negatives
            logit_scale = model.module.logit_scale.exp()
            
            # Action features are already aligned with positives (no need to map)
            # Loss computation in float32 for numerical stability
            with torch.amp.autocast('cuda', enabled=False):  # Disable autocast for loss computation
                if use_bf16:
                    # Convert to float32 for loss computation
                    combined_features_fp32 = combined_features.float()
                    action_features_fp32 = action_features.float()
                    hard_negative_features_fp32 = []
                    for hn_feats in hard_negative_features_list:
                        if hn_feats.numel() > 0:
                            hard_negative_features_fp32.append(hn_feats.float())
                        else:
                            hard_negative_features_fp32.append(hn_feats)
                else:
                    combined_features_fp32 = combined_features
                    action_features_fp32 = action_features
                    hard_negative_features_fp32 = hard_negative_features_list
                
                loss, loss_metrics = weighted_infonce_loss_with_hard_negatives(
                    combined_features_fp32,
                    action_features_fp32,
                    hard_negative_features_fp32,
                    hard_negative_weights_list,
                    logit_scale,
                    device,
                    alpha=hard_negative_alpha
                )
            
            # Backward pass (no scaling needed for bfloat16)
            loss.backward()
            
            # Calculate gradient metrics before clipping
            grad_metrics = calculate_gradient_metrics(model)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()  # Step the cosine annealing scheduler
            global_step += 1
            
            # Accumulate metrics
            total_train_loss += loss.item()
            total_weighted_infonce_loss += loss_metrics['loss']
            total_grad_norm += grad_metrics['grad_norm']
            total_hard_neg_accuracy += loss_metrics['accuracy']
            train_batch_count += 1
            
            batch_time = time.time() - batch_start_time
            
            # Log training metrics every train_log_freq steps (with proper DDP synchronization)
            if global_step % train_log_freq == 0:
                # Create a tensor with all metrics to sync across processes
                step_metrics = torch.tensor([
                    loss.item(), 
                    loss_metrics['loss'], 
                    loss_metrics['accuracy'], 
                    grad_metrics['grad_norm'],
                    loss_metrics['positive_logits_mean'],
                    loss_metrics['positive_logits_std']
                ], device=device)
                
                # Sum metrics from all processes
                dist.all_reduce(step_metrics, op=dist.ReduceOp.SUM)
                
                # Average the metrics
                step_metrics /= world_size
                
                # Unpack averaged metrics
                avg_loss_item = step_metrics[0].item()
                avg_weighted_loss = step_metrics[1].item()
                avg_accuracy = step_metrics[2].item()
                avg_grad_norm = step_metrics[3].item()
                avg_pos_logits_mean = step_metrics[4].item()
                avg_pos_logits_std = step_metrics[5].item()
                
                # Log the averaged metrics only on rank 0
                if rank == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logit_scale_val = logit_scale.item()
                    gpu_metrics = get_gpu_metrics(device)
                    
                    log_dict = {
                        "step": global_step,
                        "epoch": epoch,
                        "learning_rate": current_lr,
                        "train/step_loss": avg_loss_item,
                        "train/step_weighted_infonce_loss": avg_weighted_loss,
                        "train/step_hard_neg_accuracy": avg_accuracy,
                        "train/step_grad_norm": avg_grad_norm,
                        "model/logit_scale": logit_scale_val,
                        "timing/batch_time_sec": batch_time,
                        "train/positive_logits_mean": avg_pos_logits_mean,
                        "train/positive_logits_std": avg_pos_logits_std,
                    }
                    
                    # Add GPU metrics if available
                    for key, value in gpu_metrics.items():
                        log_dict[f"gpu/{key}"] = value
                    
                    if use_wandb:
                        import wandb
                        wandb.log(log_dict)
                    
                    print(f"Step {global_step}: Loss={avg_loss_item:.4f}, LR={current_lr:.2e}, "
                          f"HardNegAcc={avg_accuracy:.3f}")
            
            if rank == 0:
                train_batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'hn_acc': f'{loss_metrics["accuracy"]:.3f}',
                    'grad': f'{grad_metrics["grad_norm"]:.3f}',
                    'step': global_step
                })
            
            # Run evaluation every eval_log_freq steps
            if global_step % eval_log_freq == 0 and global_step > 0:
                model.eval()
                eval_start_time = time.time()
                
                # Quick evaluation metrics
                eval_losses = []
                eval_hard_neg_accs = []
                
                with torch.no_grad():
                    for eval_batch_idx, eval_batch_data in enumerate(val_dataloader):
                        if eval_batch_idx >= 5:  # Only evaluate on first 5 batches for speed
                            break
                            
                        eval_images = eval_batch_data['images'].to(device)
                        eval_positive_instructions = eval_batch_data['positive_instructions']
                        eval_hard_negative_data = eval_batch_data['hard_negative_data']
                        eval_action_histories = eval_batch_data['action_histories'].to(device)
                        eval_sample_indices = eval_batch_data['sample_indices']
                        
                        eval_tokenized_positives = clip.tokenize(eval_positive_instructions, truncate=True).to(device)
                        
                        # FIXED: Match dimensions using sample_indices
                        eval_images_for_positives = eval_images[eval_sample_indices]
                        eval_action_histories_for_positives = eval_action_histories[eval_sample_indices]
                        
                        with torch.amp.autocast('cuda', enabled=use_bf16, dtype=torch.bfloat16):
                            eval_combined_features, eval_action_features = model(eval_images_for_positives, eval_tokenized_positives, eval_action_histories_for_positives)
                        
                        # Prepare evaluation hard negatives (EFFICIENT INDEXING VERSION)
                        eval_hard_negative_features_list = []
                        eval_hard_negative_weights_list = []
                        
                        # 1. Gather all hard negative texts and their source indices
                        eval_all_hn_texts = []
                        eval_hn_source_indices = []  # Maps each hard negative to its original batch index (0 to B-1)
                        eval_hard_neg_weights_flat = []
                        eval_hard_neg_batch_map = []  # To map features back to positives
                        
                        for i in range(len(eval_positive_instructions)):
                            source_idx = eval_sample_indices[i]  # Original batch index (0 to B-1)
                            hard_negs = eval_hard_negative_data[i]
                            
                            if hard_negs:
                                count = len(hard_negs)
                                for hn in hard_negs:
                                    eval_all_hn_texts.append(hn['instruction'])
                                    eval_hn_source_indices.append(source_idx)  # Map to original sample
                                    eval_hard_neg_weights_flat.append(hn['similarity'])
                                eval_hard_neg_batch_map.append(count)
                            else:
                                eval_hard_neg_batch_map.append(0)
                        
                        # 2. Run a single forward pass for all hard negatives if any exist
                        if eval_all_hn_texts:
                            # Tokenize all hard negative texts in a single batch
                            tokenized_all_eval_hn = clip.tokenize(eval_all_hn_texts, truncate=True).to(device)
                            
                            # Use efficient indexing to select corresponding images and actions
                            eval_images_for_hn = eval_images[eval_hn_source_indices]  # (num_total_hard_negatives, C, H, W)
                            eval_actions_for_hn = eval_action_histories[eval_hn_source_indices]  # (num_total_hard_negatives, H, D)
                            
                            # SINGLE forward pass for all hard negatives with mixed precision
                            with torch.amp.autocast('cuda', enabled=use_bf16, dtype=torch.bfloat16):
                                eval_all_hn_features, _ = model(eval_images_for_hn, tokenized_all_eval_hn, eval_actions_for_hn)
                            
                            # 3. Distribute features back to positives
                            eval_current_pos = 0
                            for i in range(len(eval_positive_instructions)):
                                count = eval_hard_neg_batch_map[i]
                                if count > 0:
                                    features = eval_all_hn_features[eval_current_pos : eval_current_pos + count]
                                    weights = torch.tensor(eval_hard_neg_weights_flat[eval_current_pos : eval_current_pos + count], device=device)
                                    eval_hard_negative_features_list.append(features)
                                    eval_hard_negative_weights_list.append(weights)
                                    eval_current_pos += count
                                else:
                                    # Handle samples with no hard negatives
                                    eval_hard_negative_features_list.append(torch.empty(0, eval_combined_features.shape[1], device=device))
                                    eval_hard_negative_weights_list.append(torch.empty(0, device=device))
                        else:
                            # Handle batch with no hard negatives at all
                            for i in range(len(eval_positive_instructions)):
                                eval_hard_negative_features_list.append(torch.empty(0, eval_combined_features.shape[1], device=device))
                                eval_hard_negative_weights_list.append(torch.empty(0, device=device))
                        
                        # Action features are already aligned with positives (no need to map)
                        eval_loss, eval_loss_metrics = weighted_infonce_loss_with_hard_negatives(
                            eval_combined_features,
                            eval_action_features,  # Already aligned with positives
                            eval_hard_negative_features_list,
                            eval_hard_negative_weights_list,
                            logit_scale,
                            device,
                            alpha=hard_negative_alpha
                        )
                        
                        eval_losses.append(eval_loss.item())
                        eval_hard_neg_accs.append(eval_loss_metrics['accuracy'])
                
                # Calculate averages
                avg_eval_loss = np.mean(eval_losses) if eval_losses else 0
                avg_eval_hard_neg_acc = np.mean(eval_hard_neg_accs) if eval_hard_neg_accs else 0
                eval_time = time.time() - eval_start_time
                
                if rank == 0:
                    eval_log_dict = {
                        "step": global_step,
                        "epoch": epoch,
                        "eval/step_loss": avg_eval_loss,
                        "eval/step_hard_neg_accuracy": avg_eval_hard_neg_acc,
                        "timing/eval_time_sec": eval_time,
                    }
                    
                    if use_wandb:
                        import wandb
                        wandb.log(eval_log_dict)
                    
                    print(f"Eval at step {global_step}: Loss={avg_eval_loss:.4f}, "
                          f"HardNegAcc={avg_eval_hard_neg_acc:.3f}")
                
                model.train()  # Switch back to training mode

        # Synchronize training metrics across all processes
        metrics_to_sync = torch.tensor([
            total_train_loss, total_weighted_infonce_loss, total_grad_norm, total_hard_neg_accuracy,
            train_batch_count
        ], device=device)
        
        dist.all_reduce(metrics_to_sync, op=dist.ReduceOp.SUM)
        
        (
            total_train_loss, total_weighted_infonce_loss, total_grad_norm, total_hard_neg_accuracy,
            train_batch_count
        ) = metrics_to_sync.tolist()
        
        # Calculate averages
        avg_train_loss = total_train_loss / train_batch_count
        avg_weighted_infonce_loss = total_weighted_infonce_loss / train_batch_count
        avg_grad_norm = total_grad_norm / train_batch_count
        avg_hard_neg_accuracy = total_hard_neg_accuracy / train_batch_count

        # Validation Phase (similar structure but with validation data)
        model.eval()
        val_start_time = time.time()
        
        # Validation metrics
        total_val_loss = 0
        total_val_weighted_infonce_loss = 0
        total_val_hard_neg_accuracy = 0
        val_batch_count = 0
        
        if rank == 0:
            val_batch_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch} (Val)", leave=False)
        else:
            val_batch_pbar = val_dataloader

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_batch_pbar):
                images = batch_data['images'].to(device)
                positive_instructions = batch_data['positive_instructions']
                hard_negative_data = batch_data['hard_negative_data']
                action_histories = batch_data['action_histories'].to(device)
                sample_indices = batch_data['sample_indices']
                
                tokenized_positives = clip.tokenize(positive_instructions, truncate=True).to(device)
                
                # FIXED: Match dimensions using sample_indices
                images_for_positives = images[sample_indices]
                action_histories_for_positives = action_histories[sample_indices]
                
                with torch.amp.autocast('cuda', enabled=use_bf16, dtype=torch.bfloat16):
                    combined_features, action_features = model(images_for_positives, tokenized_positives, action_histories_for_positives)
                
                # Prepare hard negatives for validation (EFFICIENT INDEXING VERSION)
                hard_negative_features_list = []
                hard_negative_weights_list = []
                
                # 1. Gather all hard negative texts and their source indices
                val_all_hn_texts = []
                val_hn_source_indices = []  # Maps each hard negative to its original batch index (0 to B-1)
                val_hard_neg_weights_flat = []
                val_hard_neg_batch_map = []  # To map features back to positives
                
                for i in range(len(positive_instructions)):
                    source_idx = sample_indices[i]  # Original batch index (0 to B-1)
                    hard_negs = hard_negative_data[i]
                    
                    if hard_negs:
                        count = len(hard_negs)
                        for hn in hard_negs:
                            val_all_hn_texts.append(hn['instruction'])
                            val_hn_source_indices.append(source_idx)  # Map to original sample
                            val_hard_neg_weights_flat.append(hn['similarity'])
                        val_hard_neg_batch_map.append(count)
                    else:
                        val_hard_neg_batch_map.append(0)
                
                # 2. Run a single forward pass for all hard negatives if any exist
                if val_all_hn_texts:
                    # Tokenize all hard negative texts in a single batch
                    tokenized_all_val_hn = clip.tokenize(val_all_hn_texts, truncate=True).to(device)
                    
                    # Use efficient indexing to select corresponding images and actions
                    val_images_for_hn = images[val_hn_source_indices]  # (num_total_hard_negatives, C, H, W)
                    val_actions_for_hn = action_histories[val_hn_source_indices]  # (num_total_hard_negatives, H, D)
                    
                    # SINGLE forward pass for all hard negatives with mixed precision
                    with torch.amp.autocast('cuda', enabled=use_bf16, dtype=torch.bfloat16):
                        val_all_hn_features, _ = model(val_images_for_hn, tokenized_all_val_hn, val_actions_for_hn)
                    
                    # 3. Distribute features back to positives
                    val_current_pos = 0
                    for i in range(len(positive_instructions)):
                        count = val_hard_neg_batch_map[i]
                        if count > 0:
                            features = val_all_hn_features[val_current_pos : val_current_pos + count]
                            weights = torch.tensor(val_hard_neg_weights_flat[val_current_pos : val_current_pos + count], device=device)
                            hard_negative_features_list.append(features)
                            hard_negative_weights_list.append(weights)
                            val_current_pos += count
                        else:
                            # Handle samples with no hard negatives
                            hard_negative_features_list.append(torch.empty(0, combined_features.shape[1], device=device))
                            hard_negative_weights_list.append(torch.empty(0, device=device))
                else:
                    # Handle batch with no hard negatives at all
                    for i in range(len(positive_instructions)):
                        hard_negative_features_list.append(torch.empty(0, combined_features.shape[1], device=device))
                        hard_negative_weights_list.append(torch.empty(0, device=device))
                
                # Action features are already aligned with positives (no need to map)
                val_loss, val_loss_metrics = weighted_infonce_loss_with_hard_negatives(
                    combined_features,
                    action_features,  # Already aligned with positives
                    hard_negative_features_list,
                    hard_negative_weights_list,
                    logit_scale,
                    device,
                    alpha=hard_negative_alpha
                )
                
                # Accumulate validation metrics
                total_val_loss += val_loss.item()
                total_val_weighted_infonce_loss += val_loss_metrics['loss']
                total_val_hard_neg_accuracy += val_loss_metrics['accuracy']
                val_batch_count += 1
                
                if rank == 0:
                    val_batch_pbar.set_postfix({
                        'loss': f'{val_loss.item():.4f}',
                        'hn_acc': f'{val_loss_metrics["accuracy"]:.3f}'
                    })

        # Synchronize validation metrics across all processes
        val_metrics_to_sync = torch.tensor([
            total_val_loss, total_val_weighted_infonce_loss, total_val_hard_neg_accuracy,
            val_batch_count
        ], device=device)
        
        dist.all_reduce(val_metrics_to_sync, op=dist.ReduceOp.SUM)
        
        (
            total_val_loss, total_val_weighted_infonce_loss, total_val_hard_neg_accuracy,
            val_batch_count
        ) = val_metrics_to_sync.tolist()
        
        # Calculate validation averages
        avg_val_loss = total_val_loss / val_batch_count
        avg_val_weighted_infonce_loss = total_val_weighted_infonce_loss / val_batch_count
        avg_val_hard_neg_accuracy = total_val_hard_neg_accuracy / val_batch_count
        
        # Calculate timing metrics
        epoch_time = time.time() - epoch_start_time
        val_time = time.time() - val_start_time
        train_time = epoch_time - val_time

        # Update progress bar and log (only on rank 0)
        if rank == 0:
            # Get current model-specific metrics
            current_lr = optimizer.param_groups[0]['lr']
            logit_scale_val = logit_scale.item()
            gpu_metrics = get_gpu_metrics(device)
            
            # Update progress bar with key metrics
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'train_hn_acc': f'{avg_hard_neg_accuracy:.3f}',
                'val_hn_acc': f'{avg_val_hard_neg_accuracy:.3f}',
                'lr': f'{current_lr:.2e}'
            })

            if use_wandb:
                # Comprehensive logging to wandb
                log_dict = {
                    # Basic metrics
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    
                    # Training metrics
                    "train/loss": avg_train_loss,
                    "train/weighted_infonce_loss": avg_weighted_infonce_loss,
                    "train/hard_neg_accuracy": avg_hard_neg_accuracy,
                    "train/grad_norm": avg_grad_norm,
                    
                    # Validation metrics
                    "val/loss": avg_val_loss,
                    "val/weighted_infonce_loss": avg_val_weighted_infonce_loss,
                    "val/hard_neg_accuracy": avg_val_hard_neg_accuracy,
                    
                    # Model-specific metrics
                    "model/logit_scale": logit_scale_val,
                    "model/temperature": 1.0 / logit_scale_val,
                    
                    # Timing metrics
                    "timing/epoch_time_sec": epoch_time,
                    "timing/train_time_sec": train_time,
                    "timing/val_time_sec": val_time,
                }
                
                # Add GPU metrics if available
                for key, value in gpu_metrics.items():
                    log_dict[f"gpu/{key}"] = value
                
                wandb.log(log_dict)
            
            # Print detailed metrics every 10 epochs
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch} Detailed Metrics:")
                print(f"  Training   - Loss: {avg_train_loss:.4f}, Hard Neg Acc: {avg_hard_neg_accuracy:.3f}")
                print(f"  Validation - Loss: {avg_val_loss:.4f}, Hard Neg Acc: {avg_val_hard_neg_accuracy:.3f}")
                print(f"  Model      - Logit Scale: {logit_scale_val:.3f}, Temperature: {1.0/logit_scale_val:.3f}")
                print(f"  Training   - Grad Norm: {avg_grad_norm:.3f}, LR: {current_lr:.2e}")
                print(f"  Timing     - Epoch: {epoch_time:.1f}s, Train: {train_time:.1f}s, Val: {val_time:.1f}s")
                if gpu_metrics:
                    print(f"  GPU        - Memory: {gpu_metrics.get('gpu_memory_allocated_gb', 0):.2f}GB allocated")

            # Save best model (only on rank 0)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.module.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.4f} at {best_model_path}")
                if use_wandb: 
                    wandb.run.summary["best_val_loss"] = best_val_loss

            # Periodic checkpoints (only on rank 0)
            if (epoch + 1) % 100 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"{save_name}_epoch_{epoch+1}.pt")
                torch.save(model.module.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        # Synchronize all processes before next epoch
        dist.barrier(device_ids=[rank])

    # Cleanup wandb (only on rank 0)
    if use_wandb and rank == 0:
        wandb.finish()

    # Load best model weights before returning (only on rank 0)
    if rank == 0:
        if os.path.exists(best_model_path):
            print(f"Loading best model weights from {best_model_path}")
            try:
                model.module.load_state_dict(torch.load(best_model_path, map_location=device))
            except Exception as e:
                print(f"Warning: Could not load best model weights after training: {e}. Returning last state.")
        else:
            print("Warning: Best model checkpoint not found. Returning last state.")

    cleanup_distributed()
    return model.module if rank == 0 else None


def save_finetuned_clip(model, save_path):
    """Save the finetuned CLIP model state dict"""
    torch.save(model.state_dict(), save_path)


def infer_action_dim_from_dataset(dataset_dict):
    """Infer action dimension from the dataset (supports both normalized and hard negatives formats)"""
    # Check format version
    metadata = dataset_dict.get('_metadata', {})
    format_version = metadata.get('format_version', '1.0_legacy')
    
    if format_version in ['3.0_with_hard_negatives', '2.0_normalized']:
        # New normalized format
        action_histories = dataset_dict.get('action_histories', {})
        if action_histories:
            # Get the first action history
            first_action_id = next(iter(action_histories))
            action_hist = action_histories[first_action_id]
            action_hist = np.array(action_hist)
            return action_hist.shape[1]
    else:
        # Legacy format
        for instruction, data in dataset_dict.items():
            # Skip metadata entry
            if instruction == '_metadata':
                continue
            samples = data.get('samples', [])
            if samples:
                action_hist = samples[0].get('action_history')
                if action_hist is not None:
                    # Convert to numpy array to get shape
                    action_hist = np.array(action_hist)
                    return action_hist.shape[1]
    
    raise ValueError("Could not infer action dimension from dataset")


def load_dataset_with_streaming(json_path, use_streaming=True):
    """Load dataset from JSON file with optional streaming support for large files"""
    if use_streaming:
        print(f"Loading dataset from {json_path} with streaming...")
        try:
            import ijson
            with open(json_path, 'rb') as f:
                # Try the most compatible ijson approach
                try:
                    # Method 1: Use common() backend which is most reliable
                    items = ijson.items(f, '', use_float=True)
                    dataset = next(items)
                    print("Successfully loaded with ijson streaming")
                    return dataset
                except:
                    # Method 2: Try without backend specification
                    f.seek(0)
                    items = ijson.items(f, '')
                    dataset = next(items)
                    print("Successfully loaded with ijson streaming (fallback)")
                    return dataset
        except Exception as e:
            print(f"Warning: Streaming failed ({e}), falling back to regular JSON loading...")
    
    # Regular JSON loading (fallback or when streaming is disabled)
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
    model = train_clip_bridge_hard_negatives_ddp(
        rank=rank,
        world_size=world_size,
        dataset_path=args.augmented_dataset,
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
        warmup_epochs=args.warmup_epochs,
        train_log_freq=args.train_log_freq,
        eval_log_freq=args.eval_log_freq,
        hard_negative_alpha=args.hard_negative_alpha,
        use_bf16=args.use_bf16 if hasattr(args, 'use_bf16') else None,
        cosine_cycles=args.cosine_cycles
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VLA-CLIP model for Bridge dataset with action trajectories, contrastive loss, and hard negatives using DDP')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (per GPU)')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of data to use for validation')

    # Model parameters
    parser.add_argument('--history_length', type=int, required=True, help='Action history length (must match dataset)')
    parser.add_argument('--action_dim', type=int, default=None, help='Action dimension (will be inferred from data if not specified)')
    parser.add_argument('--use_transformer', action='store_true', help='Use transformer for action history encoding instead of MLP')

    # Dataset and paths
    parser.add_argument('--augmented_dataset', type=str, required=True, help='Path to augmented Bridge dataset JSON file (with hard negatives)')
    parser.add_argument('--images_folder', type=str, required=True, help='Path to folder containing agent view images as JPG files')
    parser.add_argument('--checkpoint_dir', type=str, default='bridge_trajectory_hard_negatives_checkpoints_ddp', help='Directory to save checkpoints')
    parser.add_argument('--save_name', type=str, default=None, help='Name for saved model and wandb run (generated if None)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint state_dict to resume training from')

    # DDP parameters
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--port', type=int, default=12355, help='Port for distributed communication')

    # Hard negatives parameters
    parser.add_argument('--hard_negative_alpha', type=float, default=1.0, help='Weight for hard negatives vs in-batch negatives in InfoNCE loss')

    # Training optimization
    parser.add_argument('--use_bf16', action='store_true', help='Use bfloat16 mixed precision training (default: auto-detect)')
    parser.add_argument('--cosine_cycles', type=float, default=0.5, help='Number of cosine annealing cycles')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of epochs for linear warmup')
    parser.add_argument('--train_log_freq', type=int, default=50, help='Log training metrics every N steps')
    parser.add_argument('--eval_log_freq', type=int, default=500, help='Run evaluation every N steps')

    args = parser.parse_args()

    # Validate arguments
    if args.world_size > torch.cuda.device_count():
        print(f"Error: Requested {args.world_size} GPUs but only {torch.cuda.device_count()} available.")
        exit(1)

    if args.world_size < 1:
        print("Error: world_size must be at least 1.")
        exit(1)

    # Import wandb only if needed
    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed. Running without wandb logging.")
            args.use_wandb = False

    # Load augmented dataset
    if not os.path.exists(args.augmented_dataset):
        print(f"Error: Augmented dataset file not found at {args.augmented_dataset}")
        exit(1)

    if not os.path.exists(args.images_folder):
        print(f"Error: Images folder not found at {args.images_folder}")
        exit(1)

    # Lightweight dataset validation and action_dim inference
    print(f"Validating dataset file: {args.augmented_dataset}")
    
    # Infer action dimension if not provided (lightweight approach)
    if args.action_dim is None:
        try:
            print("Inferring action dimension from dataset...")
            # Load only metadata to get action_dim without loading full dataset
            import ijson
            with open(args.augmented_dataset, 'rb') as f:
                # Try to get metadata first
                try:
                    metadata_parser = ijson.items(f, '_metadata')
                    metadata = next(metadata_parser, {})
                    if metadata and 'action_dim' in metadata:
                        args.action_dim = metadata['action_dim']
                        print(f"Using action dimension from metadata: {args.action_dim}")
                    else:
                        # Fallback: load minimal data to infer action_dim
                        f.seek(0)
                        full_data = ijson.items(f, '')
                        dataset_sample = next(full_data)
                        args.action_dim = infer_action_dim_from_dataset(dataset_sample)
                        print(f"Inferred action dimension: {args.action_dim}")
                        del dataset_sample  # Free memory immediately
                except Exception:
                    # Final fallback: use default action_dim
                    args.action_dim = 7  # Default for Bridge dataset
                    print(f"Warning: Could not infer action dimension, using default: {args.action_dim}")
        except Exception as e:
            print(f"Error with action dimension inference: {e}")
            args.action_dim = 7  # Default fallback
            print(f"Using default action dimension: {args.action_dim}")
    
    print(f"Dataset will be loaded in each DDP process to minimize memory usage.")

    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Final save path (best model)
    if args.save_name is None:
        args.save_name = f"vla_clip_bridge_hn_h{args.history_length}_{'transformer' if args.use_transformer else 'mlp'}_ddp"
    FINAL_SAVE_PATH = os.path.join(args.checkpoint_dir, f"{args.save_name}_final_best.pt")

    print("Starting DDP training with hard negatives...")
    print(f"Config: History={args.history_length}, ActionDim={args.action_dim}, ActionEncoder={'Transformer' if args.use_transformer else 'MLP'}, LR={args.lr}, BS={args.batch_size}")
    print(f"DDP Config: World Size={args.world_size}, Port={args.port}")
    print(f"Hard Negatives Config: Alpha={args.hard_negative_alpha}")
    print(f"Warmup Config: Warmup Epochs={args.warmup_epochs}, Train Log Freq={args.train_log_freq}, Eval Log Freq={args.eval_log_freq}")
    print(f"Using wandb: {args.use_wandb}")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")

    # Spawn processes for DDP training
    if args.world_size == 1:
        # Single GPU training - no need for multiprocessing
        print("Running on single GPU (no multiprocessing)")
        finetuned_model = ddp_main(0, 1, args)
    else:
        # Multi-GPU training with multiprocessing
        print(f"Spawning {args.world_size} processes for DDP training")
        mp.spawn(
            ddp_main,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
        finetuned_model = None  # Model is saved in ddp_main, not returned

    # Save final model (only applicable for single GPU or when running on rank 0)
    if finetuned_model is not None:
        print(f"Saving final model (best validation weights) to {FINAL_SAVE_PATH}...")
        save_finetuned_clip(finetuned_model, FINAL_SAVE_PATH)
        print("Done!")
    else:
        print("DDP training completed. Check checkpoint directory for saved models.")