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

class CustomDataset(Dataset):
    def __init__(self, dataset_dict, trajectory_length=20):
        """
        Args:
            dataset_dict: Dictionary containing language instructions and corresponding image sequences
            trajectory_length: Number of past actions to include in each trajectory
        """
        self.samples = []
        self.trajectory_length = trajectory_length
        
        # Flatten the dataset into (image, text, action_trajectory) triplets
        for instruction, data in dataset_dict.items():
            images = data['images']
            actions = data['actions']
            
            for i, img_seq in enumerate(images):
                for j, frame in enumerate(img_seq):
                    # Create action trajectory (past actions)
                    action_trajectory = []   
                    # Add past actions, padding with zeros if not enough history
                    for k in range(j - self.trajectory_length, j):
                        if k < 0:
                            if j > 0:  # If we have at least one valid action
                                action_trajectory.append(actions[i][0])  # Use the first action
                            else:
                                if k == j - 1:  # Last position before current frame
                                    # Use zeros (neutral action) instead of padding token
                                    action_trajectory.append(np.zeros(actions[i][0].shape))
                                else:
                                    action_trajectory.append(np.ones(actions[i][0].shape) * -5.0)  # Padding for earlier positions
                        else:
                            action_trajectory.append(actions[i][k])
                    
                    # Convert to numpy array for consistency
                    action_trajectory = np.array(action_trajectory)
                    
                    # Verify that not all values are padding
                    if np.all(action_trajectory[:, 0] == -5.0):
                        print(f"Warning: Found all-padding trajectory for instruction {instruction}, fixing...")
                        # Force at least one non-padding value
                        action_trajectory[-1] = np.zeros(action_trajectory[-1].shape)
                    
                    # Add to samples
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
    def __init__(self, model_config):
        super().__init__()
        self.clip = model_config.clip_model
        # for param in self.clip.parameters():
        #     param.requires_grad = False
        
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
        
        # Add a projection layer to match dimensions
        # self.input_projection = nn.Sequential(
        #     nn.Linear(self.f_t_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, vision_pooling_output_dim)
        # )
        self.input_projection = nn.Linear(self.f_t_dim, vision_pooling_output_dim)
        
        # Create action trajectory processing components
        self.action_dim = model_config.action_dim
        self.trajectory_length = model_config.trajectory_length
        
        # Create an embedding for the padding token
        self.padding_embedding = nn.Parameter(torch.randn(vision_pooling_output_dim))
        
        # Process each action in the trajectory
        self.action_encoder = nn.Linear(self.action_dim, 128)
        
        # Process the entire trajectory with a transformer
        self.trajectory_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Final projection to match vision_pooling_output_dim
        self.trajectory_projection = nn.Linear(128, vision_pooling_output_dim)
        
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
        # Normalize features
        combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
        # Project combined features
        combined_features = self.input_projection(combined_features)
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
        # Reshape to process all actions at once
        traj_shape = action_trajectories.shape
        flat_actions = action_trajectories.reshape(-1, self.action_dim).to(image.device)
        encoded_actions = self.action_encoder(flat_actions.float())
        encoded_actions = encoded_actions.reshape(traj_shape[0], traj_shape[1], -1)
        # Process with transformer
        encoded_trajectory = self.trajectory_encoder(encoded_actions, src_key_padding_mask=padding_mask) 
        # Pool across time dimension (mean of non-padding tokens)
        # Create a mask for non-padding tokens (1 for real, 0 for padding)
        non_padding_mask = (~padding_mask).float().unsqueeze(-1).clamp(min=1e-6)
        # Apply mask and take mean
        pooled_trajectory = (encoded_trajectory * non_padding_mask).sum(dim=1) / non_padding_mask.sum(dim=1)
        # Final projection
        projected_trajectory = self.trajectory_projection(pooled_trajectory.float())
        # Normalize
        projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)
        
        # Get logits
        image_logits = torch.matmul(combined_features, projected_trajectory.T)
        action_logits = torch.matmul(projected_trajectory, combined_features.T)
        # input()
        return image_logits, action_logits
        
def train_clip(
    dataset_dict: dict,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-7,
    weight_decay: float = 0.2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # Load the CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    # Create model configuration
    model_config = ModelConfig(clip_model=clip_model)
    
    # Initialize the model with the configuration
    model = VLA_CLIP(model_config).float().to(device)
    
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
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
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
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs")
    for epoch in epoch_pbar:
        total_loss = 0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        
        for batch_idx, (images, texts, actions) in enumerate(batch_pbar):
            images = images.to(device)
            # print (texts)
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
            # Check optimizer state after backward
            if batch_idx == 0:
                print("\nAfter first backward pass:")
                print(f"Optimizer state dict size: {len(optimizer.state_dict()['state'])}")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # Update progress bar
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
    
    return model

def save_finetuned_clip(model, save_path):
    """Save the finetuned CLIP model"""
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    # load libero dataset
    libero_base_path = "/home/xilun/LIBERO/libero/datasets"
    
    dataset_dict = {}
    # Get all subdirectories in the datasets folder
    dataset_folders = [f for f in os.listdir(libero_base_path) 
                      if os.path.isdir(os.path.join(libero_base_path, f))]
    
    # Process each dataset folder
    for folder in dataset_folders:
        folder_path = os.path.join(libero_base_path, folder)
        print(f"\nLoading dataset folder: {folder}")
        
        # Process each task file in the folder
        for task in os.listdir(folder_path):
            if not task.endswith('.hdf5'):
                continue
                
            # Extract task name without .hdf5 extension as the language instruction
            language_instruction = task.replace('.hdf5', '')
            dataset_dict[language_instruction] = {
                'actions': [],
                'images': []
            }
            
            task_path = os.path.join(folder_path, task)
            print(f"Loading task: {task}")
            
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
                    
                
            print(f"Processed {task}: {len(dataset_dict[language_instruction]['actions'])} demonstrations")
            print(f"Total images: {len(dataset_dict[language_instruction]['images'])}")
    
    SAVE_PATH = "finetuned_clip_trajectory.pt"
    
    print("Starting training...")
    finetuned_model = train_clip(
        dataset_dict=dataset_dict,
        num_epochs=30,
        batch_size=64,
        learning_rate=1e-5,
        weight_decay=0.2
    )
    
    print("Saving model...")
    save_finetuned_clip(finetuned_model, SAVE_PATH)
    print("Done!")