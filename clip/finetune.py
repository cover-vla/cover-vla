import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
import h5py
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from model import TextAwareVisualExtraction

class CustomDataset(Dataset):
    def __init__(self, dataset_dict):
        """
        Args:
            dataset_dict: Dictionary containing language instructions and corresponding image sequences
        """
        self.samples = []
        
        # Flatten the dataset into (image, text) pairs
        for instruction, data in dataset_dict.items():
            images = data['images']  # Shape: (103, 128, 128, 3)
            # Use each frame in the sequence as a separate training example
            actions = data['actions']
            for i, img_seq in enumerate(images):
                for j, frame in enumerate(img_seq):  # Now frame is (128, 128, 3)
                    self.samples.append((frame, instruction, actions[i][j]))
        
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
        image, caption, action = self.samples[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(image.astype('uint8'))
        image = self.preprocess(image)
        
        return image, caption, action


class VLA_CLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        
        # The number of patches depends on the CLIP model's vision transformer
        # For ViT-B/32, the image is divided into 7x7=49 patches (for 224x224 images)
        # For ViT-B/16, it would be 14x14=196 patches
        # You're using ViT-B/32, so num_img_patches should be 49
        self.text_aware_visual_extraction = TextAwareVisualExtraction(
            num_img_patches=49,  # Changed from 103 to 49 for ViT-B/32
            vision_dim=512
        )
        
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
        batch_size = images.shape[0]
        
        # Register hooks to capture intermediate activations
        activations = {}
        
        def hook_fn_image(module, input, output):
            activations['image_patches'] = output
            
        def hook_fn_text(module, input, output):
            activations['text_features'] = output
        
        # Register hooks on appropriate layers
        image_hook = self.clip.visual.transformer.resblocks[-1].register_forward_hook(hook_fn_image)
        text_hook = self.clip.transformer.resblocks[-1].register_forward_hook(hook_fn_text)
        
        # Forward pass through CLIP
        with torch.no_grad():
            _ = self.clip.encode_text(text)
            _ = self.clip.encode_image(images)
        
        # Remove hooks
        image_hook.remove()
        text_hook.remove()
        
        # Process patch features from activations
        patch_features = activations['image_patches'][1:, :, :]  # Shape: (num_patches, batch_size, embedding_dim)
        patch_features = patch_features.permute(1, 0, 2)
        
        # Apply layer normalization and projection if needed
        if hasattr(self.clip.visual, 'ln_post'):
            patch_features = self.clip.visual.ln_post(patch_features)
            
        if hasattr(self.clip.visual, 'proj') and self.clip.visual.proj is not None:
            patch_features = patch_features @ self.clip.visual.proj
            
        # Normalize features
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        
        # Get text features from activations
        text_features = activations['text_features']
        
        if text_features.shape[0] != batch_size:
            # If first dimension is not batch_size, permute dimensions
            text_features = text_features.permute(1, 0, 2)
        
        # Print shapes for debugging
        # print(f"Text features shape: {text_features.shape}")
        # print(f"Patch features shape: {patch_features.shape}")
        
        return patch_features, text_features
        
    def forward(self, image, text, actions):
        # Extract patch-level features and text features
        patch_features, text_features = self.extract_clip_features(image, text)
        
        # Process through text-aware visual extraction
        text_aware_features = self.text_aware_visual_extraction(patch_features, text_features)
        
        # Average pool across tokens to get a single vector per image
        enhanced_image_features = text_aware_features.mean(dim=1)
        
        # Normalize features
        enhanced_image_features = enhanced_image_features / enhanced_image_features.norm(dim=-1, keepdim=True)
        
        # Get action features - properly tokenize the actions
        action_texts = [str(a) for a in actions]  # Convert each action to string
        action_tokens = clip.tokenize(action_texts).to(image.device)  # Tokenize and move to device
        action_features = self.clip.encode_text(action_tokens)
        action_features = action_features / action_features.norm(dim=-1, keepdim=True)
        
        # Get logits
        image_logits = torch.matmul(enhanced_image_features, action_features.T)
        action_logits = torch.matmul(action_features, enhanced_image_features.T)
        
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
    model = VLA_CLIP(clip_model).float().to(device)
    
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
            logits_per_image = logits_per_image / 0.07  # CLIP default temperature
            logits_per_action = logits_per_action / 0.07  # Ensure same scaling

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
    libero_path = "/home/xilun/LIBERO/libero/datasets/libero_spatial"
    dataset_dict = {}
    
    for task in os.listdir(libero_path):
        if not task.endswith('.hdf5'):
            continue
            
        # Extract task name without .hdf5 extension as the language instruction
        language_instruction = task.replace('.hdf5', '')
        dataset_dict[language_instruction] = {
            'actions': [],
            'images': []
        }
        
        task_path = os.path.join(libero_path, task)
        print(f"Loading task: {task}")
        
        with h5py.File(task_path, 'r') as f:
            for demo_key in f['data'].keys():
                demo_data = f['data'][demo_key]
                
                # Get actions data
                actions = demo_data['actions'][()]
                dataset_dict[language_instruction]['actions'].append(actions)
                
                # Get observation data
                # Since 'obs' is a group with 7 members, we need to handle it appropriately
                obs_group = demo_data['obs']
                obs_data = obs_group['agentview_rgb'][()]
                dataset_dict[language_instruction]['images'].append(obs_data)
                
        print(f"Processed {task}: {len(dataset_dict[language_instruction]['actions'])} demonstrations")
        print(f"Total images: {len(dataset_dict[language_instruction]['images'])}")
    
    SAVE_PATH = "finetuned_clip.pt"
    
    print("Starting training...")
    finetuned_model = train_clip(
        dataset_dict=dataset_dict,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-7,
        weight_decay=0.2
    )
    
    print("Saving model...")
    save_finetuned_clip(finetuned_model, SAVE_PATH)
    print("Done!")