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
from typing import Optional
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

class BridgeDataset(Dataset):
    def __init__(self, augmented_dataset_dict, history_length, images_folder, world_size=1, rank=0):
        """
        Memory-efficient Dataset for Bridge V2 with DDP support
        
        Args:
            augmented_dataset_dict: Dictionary loaded from the augmented Bridge dataset JSON file.
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
        
        # Check if this is the new normalized format
        metadata = augmented_dataset_dict.get('_metadata', {})
        format_version = metadata.get('format_version', '1.0_legacy')
        
        if format_version == '2.0_normalized':
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
    
    def _load_normalized_format_efficient(self, augmented_dataset_dict):
        """Memory-efficient loading with DDP sharding"""
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
            
            # Store the sample index and IDs for lazy loading
            self.sample_indices.append({
                'idx': i,
                'action_history_id': action_history_id,
                'instruction_id': instruction_id,
                'agent_view_image_file': agent_view_image_file
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
                    'action_history': action_hist
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
            # Normalized format - lazy loading
            action_history_id = sample_info['action_history_id']
            instruction_id = sample_info['instruction_id']
            agent_view_image_file = sample_info['agent_view_image_file']
            
            # Load data on demand
            action_hist = np.array(self.action_histories[action_history_id])
            caption = self.instructions[instruction_id]
        else:
            # Legacy format - data already loaded
            agent_view_image_file = sample_info['agent_view_image_file']
            caption = sample_info['language_instruction']
            action_hist = np.array(sample_info['action_history'])
        
        # Load image from file
        image_path = os.path.join(self.images_folder, agent_view_image_file)
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
            
        img = self.preprocess(img)
        return img, caption, action_hist


class VLA_CLIP_Bridge(nn.Module):
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

        logits_scale = self.logit_scale.exp()
        # Calculate logits
        image_logits = logits_scale * torch.matmul(combined_features, projected_trajectory.T)
        action_logits = logits_scale * torch.matmul(projected_trajectory, combined_features.T)

        return image_logits, action_logits


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


def train_clip_bridge_ddp(
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
    eval_log_freq = 500
):
    """DDP training function"""
    # Setup distributed training
    setup_distributed(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    
    # Initialize wandb only on rank 0
    if use_wandb and rank == 0:
        import wandb
        # Ensure save_name is set if using wandb
        if save_name is None:
             save_name = f"vla_clip_bridge_h{history_length}_{'transformer' if use_transformer else 'mlp'}_ddp"
             print(f"Generated save_name for wandb: {save_name}")

        wandb.init(project="VLA-CLIP-Bridge-DDP", name=save_name)
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
        })

    # Create checkpoint directory if it doesn't exist (only on rank 0)
    if rank == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if save_name is None:
            save_name = f"vla_clip_bridge_h{history_length}_{'transformer' if use_transformer else 'mlp'}_ddp"

    # Ensure all processes have the save_name
    dist.barrier(device_ids=[rank])

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    
    # Load the CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    
    # Create model configuration for Bridge, passing history_length and action_dim
    model_config = ModelConfig(clip_model=clip_model, history_length=history_length, action_dim=action_dim)
    
    # Initialize the model
    model = VLA_CLIP_Bridge(model_config, use_transformer=use_transformer).float().to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Note: scheduler will be created after we have train_dataloader
    scheduler = None
    start_epoch = 0
    global_step = 0

    # Load checkpoint if specified (only on rank 0, then broadcast)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        if rank == 0:
            print(f"Loading model state dict from {resume_checkpoint}")
            try:
                checkpoint = torch.load(resume_checkpoint, map_location=device)
                model.module.load_state_dict(checkpoint)
                print("Successfully loaded model weights.")
            except Exception as load_err:
                print(f"Error loading checkpoint: {load_err}. Starting training from scratch.")
                start_epoch = 0
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
                
                if format_version == '2.0_normalized':
                    total_samples = metadata.get('total_samples', 0)
                    print(f"Total samples in dataset: {total_samples:,}")
        
        # Create dataset with DDP sharding
        dataset = BridgeDataset(
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

    # Create data loaders (no DistributedSampler needed - dataset is already sharded)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Can shuffle since each process has its own shard
        num_workers=4, 
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=2048, 
        shuffle=False, 
        num_workers=1,  # Reduced for evaluation to prevent worker abortion
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Now create the scheduler with the correct warmup steps
    total_train_steps = len(train_dataloader) * num_epochs
    warmup_steps = len(train_dataloader) * warmup_epochs
    
    def warmup_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1.0
    
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Training loop
    best_val_loss = float('inf')
    if rank == 0:
        best_model_path = os.path.join(checkpoint_dir, f"{save_name}_best.pt")
        epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Training epochs")
        print(f"Warmup steps: {warmup_steps}, Total training steps: {total_train_steps}")
        print(f"Training log frequency: every {train_log_freq} steps")
        print(f"Evaluation frequency: every {eval_log_freq} steps")
    else:
        epoch_pbar = range(start_epoch, num_epochs)

    for epoch in epoch_pbar:
        
        # Training Phase
        model.train()
        epoch_start_time = time.time()
        
        # Training metrics
        total_train_loss = 0
        total_image_loss = 0
        total_action_loss = 0
        total_grad_norm = 0
        train_batch_count = 0
        
        # Accuracy tracking
        total_img2act_top1 = 0
        total_img2act_top5 = 0
        total_act2img_top1 = 0
        total_act2img_top5 = 0
        
        if rank == 0:
            train_batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} (Train)", leave=False)
        else:
            train_batch_pbar = train_dataloader

        for batch_idx, (img, texts, action_hists) in enumerate(train_batch_pbar):
            batch_start_time = time.time()
            
            img = img.to(device)
            input_texts = clip.tokenize(texts).to(device)
            input_actions = torch.tensor(np.array(action_hists), dtype=torch.float32, device=device)
            current_batch_size = img.shape[0]
            
            optimizer.zero_grad()
            logits_per_image, logits_per_action = model(img, input_texts, input_actions)
            positive_labels = torch.arange(current_batch_size, device=device)
            
            # Calculate separate losses
            image_loss = F.cross_entropy(logits_per_image, positive_labels)
            action_loss = F.cross_entropy(logits_per_action, positive_labels)
            loss = (image_loss + action_loss) / 2
            
            loss.backward()
            
            # Calculate gradient metrics before clipping
            grad_metrics = calculate_gradient_metrics(model)
            
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Step the warmup scheduler
            global_step += 1
            
            # Calculate accuracy metrics
            acc_metrics = calculate_accuracy_metrics(logits_per_image, logits_per_action, current_batch_size, device)
            
            # Accumulate metrics
            total_train_loss += loss.item()
            total_image_loss += image_loss.item()
            total_action_loss += action_loss.item()
            total_grad_norm += grad_metrics['grad_norm']
            train_batch_count += 1
            
            # Accumulate accuracy metrics
            total_img2act_top1 += acc_metrics.get('img2act_top1_acc', 0)
            total_img2act_top5 += acc_metrics.get('img2act_top5_acc', 0)
            total_act2img_top1 += acc_metrics.get('act2img_top1_acc', 0)
            total_act2img_top5 += acc_metrics.get('act2img_top5_acc', 0)
            
            batch_time = time.time() - batch_start_time
            
            # Log training metrics every train_log_freq steps
            if rank == 0 and global_step % train_log_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logit_scale = model.module.logit_scale.exp().item()
                gpu_metrics = get_gpu_metrics(device)
                
                log_dict = {
                    "step": global_step,
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    "train/step_loss": loss.item(),
                    "train/step_image_loss": image_loss.item(),
                    "train/step_action_loss": action_loss.item(),
                    "train/step_grad_norm": grad_metrics['grad_norm'],
                    "train/step_img2act_top1_acc": acc_metrics.get('img2act_top1_acc', 0),
                    "train/step_act2img_top1_acc": acc_metrics.get('act2img_top1_acc', 0),
                    "model/logit_scale": logit_scale,
                    "timing/batch_time_sec": batch_time,
                }
                
                # Add GPU metrics if available
                for key, value in gpu_metrics.items():
                    log_dict[f"gpu/{key}"] = value
                
                if use_wandb:
                    import wandb
                    wandb.log(log_dict)
                
                print(f"Step {global_step}: Loss={loss.item():.4f}, LR={current_lr:.2e}, "
                      f"ImgAcc={acc_metrics.get('img2act_top1_acc', 0):.3f}, "
                      f"ActAcc={acc_metrics.get('act2img_top1_acc', 0):.3f}")
            
            if rank == 0:
                train_batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'img_acc': f'{acc_metrics.get("img2act_top1_acc", 0):.3f}',
                    'act_acc': f'{acc_metrics.get("act2img_top1_acc", 0):.3f}',
                    'grad': f'{grad_metrics["grad_norm"]:.3f}',
                    'step': global_step
                })
            
            # Run evaluation every eval_log_freq steps
            if global_step % eval_log_freq == 0 and global_step > 0:
                model.eval()
                eval_start_time = time.time()
                
                # Quick evaluation metrics
                eval_losses = []
                eval_img2act_accs = []
                eval_act2img_accs = []
                
                with torch.no_grad():
                    for eval_batch_idx, (eval_img, eval_texts, eval_action_hists) in enumerate(val_dataloader):
                        if eval_batch_idx >= 10:  # Only evaluate on first 10 batches for speed
                            break
                            
                        eval_img = eval_img.to(device)
                        eval_texts_tok = clip.tokenize(eval_texts).to(device)
                        eval_action_hists = torch.tensor(np.array(eval_action_hists), dtype=torch.float32, device=device)
                        eval_batch_size = eval_img.shape[0]
                        
                        eval_logits_per_image, eval_logits_per_action = model(eval_img, eval_texts_tok, eval_action_hists)
                        eval_labels = torch.arange(eval_batch_size, device=device)
                        
                        eval_image_loss = F.cross_entropy(eval_logits_per_image, eval_labels)
                        eval_action_loss = F.cross_entropy(eval_logits_per_action, eval_labels)
                        eval_loss = (eval_image_loss + eval_action_loss) / 2
                        
                        eval_acc_metrics = calculate_accuracy_metrics(eval_logits_per_image, eval_logits_per_action, eval_batch_size, device)
                        
                        eval_losses.append(eval_loss.item())
                        eval_img2act_accs.append(eval_acc_metrics.get('img2act_top1_acc', 0))
                        eval_act2img_accs.append(eval_acc_metrics.get('act2img_top1_acc', 0))
                
                # Calculate averages
                avg_eval_loss = np.mean(eval_losses) if eval_losses else 0
                avg_eval_img2act = np.mean(eval_img2act_accs) if eval_img2act_accs else 0
                avg_eval_act2img = np.mean(eval_act2img_accs) if eval_act2img_accs else 0
                eval_time = time.time() - eval_start_time
                
                if rank == 0:
                    eval_log_dict = {
                        "step": global_step,
                        "epoch": epoch,
                        "eval/step_loss": avg_eval_loss,
                        "eval/step_img2act_top1_acc": avg_eval_img2act,
                        "eval/step_act2img_top1_acc": avg_eval_act2img,
                        "timing/eval_time_sec": eval_time,
                    }
                    
                    if use_wandb:
                        import wandb
                        wandb.log(eval_log_dict)
                    
                    print(f"Eval at step {global_step}: Loss={avg_eval_loss:.4f}, "
                          f"ImgAcc={avg_eval_img2act:.3f}, ActAcc={avg_eval_act2img:.3f}")
                
                model.train()  # Switch back to training mode

        # Synchronize training metrics across all processes
        metrics_to_sync = torch.tensor([
            total_train_loss, total_image_loss, total_action_loss, total_grad_norm,
            total_img2act_top1, total_img2act_top5, total_act2img_top1, total_act2img_top5,
            train_batch_count
        ], device=device)
        
        dist.all_reduce(metrics_to_sync, op=dist.ReduceOp.SUM)
        
        (
            total_train_loss, total_image_loss, total_action_loss, total_grad_norm,
            total_img2act_top1, total_img2act_top5, total_act2img_top1, total_act2img_top5,
            train_batch_count
        ) = metrics_to_sync.tolist()
        
        # Calculate averages
        avg_train_loss = total_train_loss / train_batch_count
        avg_image_loss = total_image_loss / train_batch_count
        avg_action_loss = total_action_loss / train_batch_count
        avg_grad_norm = total_grad_norm / train_batch_count
        avg_img2act_top1 = total_img2act_top1 / train_batch_count
        avg_img2act_top5 = total_img2act_top5 / train_batch_count
        avg_act2img_top1 = total_act2img_top1 / train_batch_count
        avg_act2img_top5 = total_act2img_top5 / train_batch_count

        # Validation Phase
        model.eval()
        val_start_time = time.time()
        
        # Validation metrics
        total_val_loss = 0
        total_val_image_loss = 0
        total_val_action_loss = 0
        val_batch_count = 0
        
        # Validation accuracy tracking
        total_val_img2act_top1 = 0
        total_val_img2act_top5 = 0
        total_val_act2img_top1 = 0
        total_val_act2img_top5 = 0
        
        if rank == 0:
            val_batch_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch} (Val)", leave=False)
        else:
            val_batch_pbar = val_dataloader

        with torch.no_grad():
            for batch_idx, (img, texts, action_hists) in enumerate(val_batch_pbar):
                img = img.to(device)
                texts_tok = clip.tokenize(texts).to(device)
                action_hists = torch.tensor(np.array(action_hists), dtype=torch.float32, device=device)
                current_batch_size = img.shape[0]
                
                logits_per_image, logits_per_action = model(img, texts_tok, action_hists)
                positive_labels = torch.arange(current_batch_size, device=device)
                
                # Calculate separate validation losses
                val_image_loss = F.cross_entropy(logits_per_image, positive_labels)
                val_action_loss = F.cross_entropy(logits_per_action, positive_labels)
                val_loss = (val_image_loss + val_action_loss) / 2
                
                # Calculate validation accuracy metrics
                val_acc_metrics = calculate_accuracy_metrics(logits_per_image, logits_per_action, current_batch_size, device)
                
                # Accumulate validation metrics
                total_val_loss += val_loss.item()
                total_val_image_loss += val_image_loss.item()
                total_val_action_loss += val_action_loss.item()
                val_batch_count += 1
                
                # Accumulate validation accuracy metrics
                total_val_img2act_top1 += val_acc_metrics.get('img2act_top1_acc', 0)
                total_val_img2act_top5 += val_acc_metrics.get('img2act_top5_acc', 0)
                total_val_act2img_top1 += val_acc_metrics.get('act2img_top1_acc', 0)
                total_val_act2img_top5 += val_acc_metrics.get('act2img_top5_acc', 0)
                
                if rank == 0:
                    val_batch_pbar.set_postfix({
                        'loss': f'{val_loss.item():.4f}',
                        'img_acc': f'{val_acc_metrics.get("img2act_top1_acc", 0):.3f}',
                        'act_acc': f'{val_acc_metrics.get("act2img_top1_acc", 0):.3f}'
                    })

        # Synchronize validation metrics across all processes
        val_metrics_to_sync = torch.tensor([
            total_val_loss, total_val_image_loss, total_val_action_loss,
            total_val_img2act_top1, total_val_img2act_top5, total_val_act2img_top1, total_val_act2img_top5,
            val_batch_count
        ], device=device)
        
        dist.all_reduce(val_metrics_to_sync, op=dist.ReduceOp.SUM)
        
        (
            total_val_loss, total_val_image_loss, total_val_action_loss,
            total_val_img2act_top1, total_val_img2act_top5, total_val_act2img_top1, total_val_act2img_top5,
            val_batch_count
        ) = val_metrics_to_sync.tolist()
        
        # Calculate validation averages
        avg_val_loss = total_val_loss / val_batch_count
        avg_val_image_loss = total_val_image_loss / val_batch_count
        avg_val_action_loss = total_val_action_loss / val_batch_count
        avg_val_img2act_top1 = total_val_img2act_top1 / val_batch_count
        avg_val_img2act_top5 = total_val_img2act_top5 / val_batch_count
        avg_val_act2img_top1 = total_val_act2img_top1 / val_batch_count
        avg_val_act2img_top5 = total_val_act2img_top5 / val_batch_count
        
        # Calculate timing metrics
        epoch_time = time.time() - epoch_start_time
        val_time = time.time() - val_start_time
        train_time = epoch_time - val_time

        # Update progress bar and log (only on rank 0)
        if rank == 0:
            # Get current model-specific metrics
            current_lr = optimizer.param_groups[0]['lr']
            logit_scale = model.module.logit_scale.exp().item()
            gpu_metrics = get_gpu_metrics(device)
            
            # Update progress bar with key metrics
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'train_acc': f'{avg_img2act_top1:.3f}',
                'val_acc': f'{avg_val_img2act_top1:.3f}',
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
                    "train/image_loss": avg_image_loss,
                    "train/action_loss": avg_action_loss,
                    "train/grad_norm": avg_grad_norm,
                    
                    # Training accuracy metrics
                    "train/img2act_top1_acc": avg_img2act_top1,
                    "train/img2act_top5_acc": avg_img2act_top5,
                    "train/act2img_top1_acc": avg_act2img_top1,
                    "train/act2img_top5_acc": avg_act2img_top5,
                    
                    # Validation metrics
                    "val/loss": avg_val_loss,
                    "val/image_loss": avg_val_image_loss,
                    "val/action_loss": avg_val_action_loss,
                    
                    # Validation accuracy metrics
                    "val/img2act_top1_acc": avg_val_img2act_top1,
                    "val/img2act_top5_acc": avg_val_img2act_top5,
                    "val/act2img_top1_acc": avg_val_act2img_top1,
                    "val/act2img_top5_acc": avg_val_act2img_top5,
                    
                    # Model-specific metrics
                    "model/logit_scale": logit_scale,
                    "model/temperature": 1.0 / logit_scale,
                    
                    # Timing metrics
                    "timing/epoch_time_sec": epoch_time,
                    "timing/train_time_sec": train_time,
                    "timing/val_time_sec": val_time,
                    "timing/samples_per_sec": (train_batch_count * batch_size * world_size) / train_time,
                }
                
                # Add GPU metrics if available
                for key, value in gpu_metrics.items():
                    log_dict[f"gpu/{key}"] = value
                
                wandb.log(log_dict)
            
            # Print detailed metrics every 10 epochs
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch} Detailed Metrics:")
                print(f"  Training   - Loss: {avg_train_loss:.4f}, Img2Act Acc: {avg_img2act_top1:.3f}, Act2Img Acc: {avg_act2img_top1:.3f}")
                print(f"  Validation - Loss: {avg_val_loss:.4f}, Img2Act Acc: {avg_val_img2act_top1:.3f}, Act2Img Acc: {avg_val_act2img_top1:.3f}")
                print(f"  Model      - Logit Scale: {logit_scale:.3f}, Temperature: {1.0/logit_scale:.3f}")
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
    """Infer action dimension from the dataset (supports both normalized and legacy formats)"""
    # Check format version
    metadata = dataset_dict.get('_metadata', {})
    format_version = metadata.get('format_version', '1.0_legacy')
    
    if format_version == '2.0_normalized':
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
    model = train_clip_bridge_ddp(
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
        eval_log_freq=args.eval_log_freq
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VLA-CLIP model for Bridge dataset with action trajectories and contrastive loss using DDP')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (per GPU)')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of data to use for validation')

    # Model parameters
    parser.add_argument('--history_length', type=int, required=True, help='Action history length (must match dataset)')
    parser.add_argument('--action_dim', type=int, default=None, help='Action dimension (will be inferred from data if not specified)')
    parser.add_argument('--use_transformer', action='store_true', help='Use transformer for action history encoding instead of MLP')

    # Dataset and paths
    parser.add_argument('--augmented_dataset', type=str, required=True, help='Path to augmented Bridge dataset JSON file (with histories)')
    parser.add_argument('--images_folder', type=str, required=True, help='Path to folder containing agent view images as JPG files')
    parser.add_argument('--checkpoint_dir', type=str, default='bridge_trajectory_checkpoints_ddp', help='Directory to save checkpoints')
    parser.add_argument('--save_name', type=str, default=None, help='Name for saved model and wandb run (generated if None)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint state_dict to resume training from')

    # DDP parameters
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--port', type=int, default=12355, help='Port for distributed communication')

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
        args.save_name = f"vla_clip_bridge_h{args.history_length}_{'transformer' if args.use_transformer else 'mlp'}_ddp"
    FINAL_SAVE_PATH = os.path.join(args.checkpoint_dir, f"{args.save_name}_final_best.pt")

    print("Starting DDP training...")
    print(f"Config: History={args.history_length}, ActionDim={args.action_dim}, ActionEncoder={'Transformer' if args.use_transformer else 'MLP'}, LR={args.lr}, BS={args.batch_size}")
    print(f"DDP Config: World Size={args.world_size}, Port={args.port}")
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