import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
from typing import Optional
import h5py
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from model import TextAwareVisualExtraction, AttentionPooling, ModelConfig
import numpy as np
import pickle

class CustomDataset(Dataset):
    def __init__(self, dataset_dict, trajectory_length=20):
        """
        Args:
            dataset_dict: Dictionary containing language instructions and corresponding image sequences
            trajectory_length: Number of past actions to include in each trajectory
        """
        self.samples = []
        self.trajectory_length = trajectory_length
        
        # # Flatten the dataset into (image, text, action_trajectory) triplets
        # for instruction, data in dataset_dict.items():
        #     images = data['images']
        #     actions = data['actions']
            
        #     for i, img_seq in enumerate(images):
        #         for j, frame in enumerate(img_seq):
        #             # Create action trajectory (past actions)
        #             action_trajectory = []   
        #             # Add past actions, padding with zeros if not enough history
        #             for k in range(j - self.trajectory_length, j):
        #                 if k < 0:
        #                     # if j > 0:  # If we have at least one valid action
        #                     #     action_trajectory.append(actions[i][0])  # Use the first action
        #                     # else:
        #                     #     if k == j - 1:  # Last position before current frame
        #                     #         # Use zeros (neutral action) instead of padding token
        #                     #         action_trajectory.append(np.zeros(actions[i][0].shape))
        #                     #     else:
        #                     action_trajectory.append(np.ones(actions[i][0].shape) * -5.0)  # Padding for earlier positions
        #                 else:
        #                     action_trajectory.append(actions[i][k])
                    
        #             # Convert to numpy array for consistency
        #             action_trajectory = np.array(action_trajectory)
                    
        #             # Verify that not all values are padding
        #             if np.all(action_trajectory[:, 0] == -5.0):
        #                 print(f"Warning: Found all-padding trajectory for instruction {instruction}, fixing...")
        #                 # Force at least one non-padding value
        #                 action_trajectory[-1] = np.zeros(action_trajectory[-1].shape)
                    
        #             # Add to samples
        #             self.samples.append((frame, instruction, action_trajectory))
        
        # Flatten the dataset into (image, text, action_trajectory) triplets
        for instruction, data in dataset_dict.items():
            images = data['images']
            actions = data['actions']
            
            for i, img_seq in enumerate(images):
                for j, frame in enumerate(img_seq):
                    # Skip if not enough future actions to form a trajectory
                    if j + self.trajectory_length > len(actions[i]):
                        continue  # Skip this sample

                    # Create action trajectory from current time step onward
                    action_trajectory = actions[i][j : j + self.trajectory_length]
                    
                    # Convert to numpy array for consistency
                    action_trajectory = np.array(action_trajectory)
                    
                    self.samples.append((frame, instruction, action_trajectory))
        
        # Get CLIP's image preprocessing pipeline
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, caption, action_trajectory = self.samples[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(image.astype('uint8'))
        image = self.preprocess(image)
        
        return image, caption, action_trajectory


class VLA_CLIP(nn.Module):
    def __init__(self, model_config, use_transformer=False):
        super().__init__()
        self.clip = model_config.clip_model
        for name, param in self.clip.named_parameters():
            param.requires_grad = False
            
        text_pooling_output_dim = model_config.text_pooling_output_dim
        pooling_heads = model_config.pooling_heads
        pooling_layers = model_config.pooling_layers
        self.num_readouts = model_config.num_readouts
        text_dim = self.clip.text_projection.shape[1]
        vision_dim = self.clip.visual.output_dim
        vision_pooling_output_dim = model_config.vision_pooling_output_dim
        
        self.visual_patch_size = self.clip.visual.conv1.kernel_size[0]
        self.num_img_patches = (224 // self.visual_patch_size) ** 2
        
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
        
        # Create action trajectory processing components
        self.action_dim = model_config.action_dim
        self.trajectory_length = model_config.trajectory_length
        
        # Create an embedding for the padding token
        self.padding_embedding = nn.Parameter(torch.randn(vision_pooling_output_dim))
        
        # Process each action in the trajectory
        hidden_dim = 128  # Use a reasonable hidden dimension
        self.action_encoder = nn.Linear(self.trajectory_length * self.action_dim, vision_pooling_output_dim)
        self.single_step_action_encoder = nn.Linear(self.action_dim, vision_pooling_output_dim)
        
        self.complex_action_encoder = nn.Sequential(
            nn.Linear(self.trajectory_length * self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, vision_pooling_output_dim)
        )
        
        # Process the entire trajectory with a transformer
        self.trajectory_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=vision_pooling_output_dim,  # Match the hidden dimension from action_encoder
                nhead=8,  # Number of attention heads (must divide hidden_dim evenly)
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=4
        )
        
        # Final projection to match vision_pooling_output_dim
        self.trajectory_projection = nn.Linear(vision_pooling_output_dim, vision_pooling_output_dim)
        
        self.use_transformer = use_transformer
        
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
        
    def forward(self, image, text, action_trajectories):
        # Extract patch-level features and text features
        patch_features, text_features = self.extract_clip_features(image, text)
        # Process through text-aware visual extraction
        text_aware_features = self.text_aware_visual_extraction(patch_features, text_features)
        vision_token = self.vision_poolings(text_aware_features)
        
        text_token = self.text_pooling(text_features)
        combined_features = torch.cat([text_token, vision_token], dim=-1)
        # Project combined features
        combined_features = self.input_projection(combined_features)
        # Normalize features
        combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
        # Convert action trajectories to tensors if needed
        if not isinstance(action_trajectories, torch.Tensor):
            # Handle different possible input types
            if isinstance(action_trajectories, list):
                if isinstance(action_trajectories[0], np.ndarray):
                    action_trajectories = [torch.tensor(traj, device=image.device) for traj in action_trajectories]
                action_trajectories = torch.stack(action_trajectories)
            else:
                action_trajectories = torch.tensor(action_trajectories, device=image.device)
        
        # Create padding mask (True where padding exists)
        padding_mask = (action_trajectories[:, :, 0] == -5.0).to(image.device)  # Ensure mask is on the same device
        # Encode each action in the trajectory
        if self.use_transformer:
            # encode each step in the trajectory
            encoded_trajectory = self.single_step_action_encoder(action_trajectories.float().to(image.device))
            encoded_trajectory = self.trajectory_encoder(encoded_trajectory, src_key_padding_mask=padding_mask)
            
            # find the last non-padding token
            batch_indices = torch.arange(encoded_trajectory.size(0), device=encoded_trajectory.device)
            last_non_padding = (~padding_mask).sum(dim=1) - 1
            last_token_representation = encoded_trajectory[batch_indices, last_non_padding]
            pooled_trajectory = last_token_representation

            # Final projection
            projected_trajectory = self.trajectory_projection(pooled_trajectory)
            ##### 
        else:
            ## direct action projection
            batch_size = action_trajectories.shape[0]
            flat_actions = action_trajectories.reshape(batch_size, -1).to(image.device)
            # encoded_actions = self.action_encoder(flat_actions.float())
            encoded_actions = self.complex_action_encoder(flat_actions.float())
            projected_trajectory = encoded_actions

        # Normalize
        projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)
        
        # Get logits
        image_logits = torch.matmul(combined_features, projected_trajectory.T)
        action_logits = torch.matmul(projected_trajectory, combined_features.T)
        return image_logits, action_logits
        
def train_clip(
    dataset_dict: dict,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-7,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_name = None,
    checkpoint_dir = "checkpoints",
    use_wandb = False,
    resume_checkpoint = None,
    use_transformer = False
):
    # Initialize wandb if enabled
    if use_wandb:
        import wandb
        wandb.init(project="VLA-CLIP", name=save_name)
        
        # Log hyperparameters
        wandb.config.update({
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "device": device,
        })
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load the CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    # Create model configuration
    model_config = ModelConfig(clip_model=clip_model)
    
    # Initialize the model with the configuration
    model = VLA_CLIP(model_config, use_transformer=use_transformer).float().to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize variables for training loop
    start_epoch = 0
    
    # Load checkpoint if specified
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size in MB: {total_params * 4 / (1024 * 1024):.2f}")
    
    # Check if model is actually on GPU
    print(f"Model device: {next(model.parameters()).device}")
    
    # Print GPU memory usage before training
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Verify optimizer state
    print("\nOptimizer state:")
    print(f"Number of parameter groups: {len(optimizer.param_groups)}")
    print(f"Parameters in optimizer: {sum(len(g['params']) for g in optimizer.param_groups)}")
    
    # Prepare dataset and dataloader
    dataset = CustomDataset(dataset_dict)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8  # Match CPU thread count
    )
    
    # Training loop with tqdm
    model.train()
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Training epochs")
    
    for epoch in epoch_pbar:
        total_loss = 0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        
        for batch_idx, (images, texts, actions) in enumerate(batch_pbar):
            images = images.to(device)
            texts = clip.tokenize(texts).to(device)
            if batch_idx == 0:  # Print memory usage after first batch
                print(f"\nMemory after loading first batch:")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            logits_per_image, logits_per_action = model(images, texts, actions)
            
            # CLIP uses contrastive loss, meaning each image should match its corresponding text
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            # Normalize the logits using temperature scaling (default in CLIP)
            logits_per_image = logits_per_image / 0.04
            logits_per_action = logits_per_action / 0.04 
            # Compute symmetric contrastive loss
            loss = (nn.CrossEntropyLoss()(logits_per_image, ground_truth) + 
                    nn.CrossEntropyLoss()(logits_per_action, ground_truth)) / 2

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log metrics to wandb if enabled
            if use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch": batch_idx + epoch * len(dataloader)
                })
            
            # Check optimizer state after backward
            if batch_idx == 0:
                print("\nAfter first backward pass:")
                print(f"Optimizer state dict size: {len(optimizer.state_dict()['state'])}")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # Update progress bar
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        
        # Log epoch metrics to wandb if enabled
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch_avg_loss": avg_loss,
            })
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{save_name}_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    # Close wandb run if enabled
    if use_wandb:
        wandb.finish()
    
    return model

def save_finetuned_clip(model, save_path):
    """Save the finetuned CLIP model"""
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train CLIP model with action trajectories')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=7e-5, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='model_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_name', type=str, default='clip_trajectory_encoder', help='Name for saved model')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--dataset_path', type=str, default='/home/xilun/LIBERO/libero/datasets', help='Path to dataset')
    parser.add_argument('--dataset_folders', nargs='+', default=['libero_spatial'], help='Dataset folders to use')
    parser.add_argument('--augmented_dataset', type=str, default=None, 
                        help='Path to augmented dataset file (if None, will use regular dataset)')
    parser.add_argument('--use_transformer', action='store_true', help='Use transformer for action encoding')
    
    args = parser.parse_args()
    
    # Import wandb only if needed
    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed. Running without wandb logging.")
            args.use_wandb = False
    
    # Load dataset
    if args.augmented_dataset and os.path.exists(args.augmented_dataset):
        print(f"Loading augmented dataset from {args.augmented_dataset}...")
        with open(args.augmented_dataset, 'rb') as f:
            dataset_dict = pickle.load(f)
        print(f"Loaded augmented dataset with {len(dataset_dict)} instructions")
    else:
        # Original dataset loading code
        print("Loading original dataset...")
        libero_base_path = args.dataset_path
        dataset_dict = {}
        # Process each dataset folder
        for folder in args.dataset_folders:
            folder_path = os.path.join(libero_base_path, folder)
            
            if not os.path.exists(folder_path):
                print(f"Warning: Dataset folder {folder_path} does not exist. Skipping.")
                continue
            
            # Process each task file in the folder
            for task in tqdm(os.listdir(folder_path), desc=f"\nLoading dataset {folder}"):
                if not task.endswith('.hdf5'):
                    continue
                    
                # Extract task name without .hdf5 extension as the language instruction
                language_instruction = task.replace('.hdf5', '').replace('_', ' ')
                dataset_dict[language_instruction] = {
                    'actions': [],
                    'images': []
                }
                
                task_path = os.path.join(folder_path, task)
                
                with h5py.File(task_path, 'r') as f:
                    for demo_key in f['data'].keys():
                        demo_data = f['data'][demo_key]
                        
                        # Get actions data
                        actions = demo_data['actions'][()]
                        dataset_dict[language_instruction]['actions'].append(actions)
                        
                        # Get observation data
                        obs_group = demo_data['obs']
                        obs_data = obs_group['agentview_rgb'][()]
                        dataset_dict[language_instruction]['images'].append(obs_data)
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    SAVE_PATH = os.path.join(args.checkpoint_dir, f"{args.save_name}_final.pt")
    print("Starting training...")
    print(f"Using wandb: {args.use_wandb}")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
    
    finetuned_model = train_clip(
        dataset_dict=dataset_dict,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_name=args.save_name,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        resume_checkpoint=args.resume,
        use_transformer=args.use_transformer
    )
    
    print("Saving final model...")
    save_finetuned_clip(finetuned_model, SAVE_PATH)
    print("Done!")