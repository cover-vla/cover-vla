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
import pickle
import argparse
import numpy as np
class CustomDataset(Dataset):
    def __init__(self, dataset_dict):
        """
        Args:
            dataset_dict: Dictionary loaded from the augmented dataset pickle file.
                          Assumes structure where negative entries correspond to positive ones.
        """
        self.samples = []
        
        # --- Data Preprocessing ---
        # Group data by (original_instruction, demo_idx, frame_idx) to find pairs
        processed_data = {} # Key: (orig_instr, demo_idx, frame_idx), Value: {'image':..., 'pos_action':..., 'neg_action':...}

        print("Processing augmented dataset to find positive/negative pairs...")
        for instruction_key, data in tqdm(dataset_dict.items(), desc="Processing instructions"):
            is_positive = data.get('is_positive', True) # Originals are positive
            is_original = data.get('is_original', False)
            original_instruction = data.get('original_instruction', instruction_key) # Use original instruction text

            # Ensure actions and images lists exist and are not empty
            if not data.get('actions') or not data.get('images'):
                # print(f"Warning: Skipping instruction '{instruction_key}' due to missing/empty actions or images.")
                continue

            num_demos = len(data['images'])
            if num_demos != len(data['actions']):
                 print(f"Warning: Mismatch in number of image/action sequences for '{instruction_key}'. Skipping.")
                 continue

            for demo_idx in range(num_demos):
                img_seq = data['images'][demo_idx]
                action_seq = data['actions'][demo_idx]

                if img_seq.shape[0] != action_seq.shape[0]:
                    print(f"Warning: Mismatch in sequence length for demo {demo_idx} of '{instruction_key}'. Skipping demo.")
                    continue
                    
                for frame_idx in range(img_seq.shape[0]):
                    frame = img_seq[frame_idx] # Shape (H, W, C)
                    action = action_seq[frame_idx] # Shape (D,)
                    
                    # Create a unique key for this timestep across transforms
                    data_key = (original_instruction, demo_idx, frame_idx)
                    
                    if data_key not in processed_data:
                        processed_data[data_key] = {}
                        
                    # Store image only once (from positive/original entry)
                    if is_positive:
                         if 'image' not in processed_data[data_key]:
                             processed_data[data_key]['image'] = frame
                         processed_data[data_key]['pos_action'] = action
                    else: # is_negative
                         processed_data[data_key]['neg_action'] = action

        # --- Create Final Sample List ---
        print("Creating paired samples...")
        skipped_incomplete = 0
        for key, value in processed_data.items():
            # Only include samples that have both a positive and negative action paired
            if 'image' in value and 'pos_action' in value and 'neg_action' in value:
                original_instruction = key[0]
                self.samples.append(
                    (value['image'], original_instruction, value['pos_action'], value['neg_action'])
                )
            else:
                skipped_incomplete += 1
        
        if skipped_incomplete > 0:
             print(f"Warning: Skipped {skipped_incomplete} incomplete samples (missing image, pos_action, or neg_action).")
             
        print(f"Created {len(self.samples)} paired samples.")
        if not self.samples:
            raise ValueError("Dataset creation resulted in 0 paired samples. Check input data and processing logic.")

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
        image, caption, pos_action, neg_action = self.samples[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(image.astype('uint8'))
        image = self.preprocess(image)
        
        # Convert actions to tensors here if not already done, maintain numpy for now
        # Let the training loop handle conversion and device placement
        
        return image, caption, pos_action, neg_action


class VLA_CLIP(nn.Module):
    def __init__(self, model_config, use_transformer = False):
        super().__init__()
        self.clip = model_config.clip_model
        for param in self.clip.parameters():
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
        
        # Create action projection layers in __init__ so they're properly tracked
        # Determine action dimension from your dataset (e.g., 7 for a 7-DOF robot)
        self.action_dim = model_config.action_dim
        self.action_projection = nn.Linear(self.action_dim, vision_pooling_output_dim)
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
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (B, first_k_tokens, text_dim)
        # Process patch features from activations
        patch_features = self.activation['image_patches'][0]
        patch_features = patch_features.permute(1, 0, 2)[:, 1:, :]  # Shape: (batch_size, num_patches, embedding_dim)
        if hasattr(self.clip.visual, 'proj') and self.clip.visual.proj is not None:
            patch_features = patch_features @ self.clip.visual.proj
            
        # Normalize features
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        
        return patch_features, text_features
        
    def forward(self, image, text, actions):
        # Extract patch-level features and text features
        patch_features, text_features = self.extract_clip_features(image, text)
        
        # Process through text-aware visual extraction
        text_aware_features = self.text_aware_visual_extraction(patch_features, text_features)
        vision_token = self.vision_poolings(text_aware_features)
        
        text_token = self.text_pooling(text_features)
        
        combined_features = torch.cat([text_token, vision_token], dim=-1)
        
        combined_features = self.input_projection(combined_features)
        
        # Normalize features
        combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
        
        # Convert actions to tensors
        action_tensors = torch.stack([torch.tensor(a, device=image.device) for a in actions])
        
        # Project action tensors using the properly initialized projection layers
        projected_actions = self.action_projection(action_tensors.float())
        
        # Normalize action features
        projected_actions = projected_actions / projected_actions.norm(dim=-1, keepdim=True)
        
        # Get logits
        image_logits = torch.matmul(combined_features, projected_actions.T)
        action_logits = torch.matmul(projected_actions, combined_features.T)
        
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
    validation_split = 0.1,
    neg_loss_weight: float = 0.2 # Add weight parameter
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
            "validation_split": validation_split,
        })
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load the CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model_config = ModelConfig(clip_model=clip_model)
    model = VLA_CLIP(model_config).float().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    start_epoch = 0

    # Load checkpoint if specified
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading model state dict from {resume_checkpoint}")
        # Load only the state_dict, optimizer state is not saved/resumed here
        try:
             # Attempt to load only weights first (safer if optimizer changed)
             model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
             print("Successfully loaded model weights.")
        except RuntimeError as e:
             print(f"Could not load state_dict directly: {e}. Trying to load full checkpoint...")
             try:
                  # Fallback to loading the whole checkpoint object if state_dict fails
                  # This assumes the checkpoint saved the state_dict under 'model_state_dict'
                  # Adjust key if needed based on how checkpoints were saved previously
                  checkpoint = torch.load(resume_checkpoint, map_location=device)
                  if 'model_state_dict' in checkpoint:
                       model.load_state_dict(checkpoint['model_state_dict'])
                       print("Successfully loaded model weights from checkpoint object.")
                       # Optionally load optimizer state if saved and needed:
                       # if 'optimizer_state_dict' in checkpoint:
                       #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                       #     print("Successfully loaded optimizer state.")
                       # if 'epoch' in checkpoint:
                       #      start_epoch = checkpoint['epoch'] + 1 # Resume from next epoch
                       #      print(f"Resuming training from epoch {start_epoch}")
                  else:
                       # If it's just the state_dict saved directly
                       model.load_state_dict(checkpoint)
                       print("Successfully loaded model weights (assumed direct state_dict save).")

             except Exception as load_err:
                  print(f"Error loading checkpoint: {load_err}. Starting training from scratch.")
                  start_epoch = 0 # Ensure we start from epoch 0 if loading fails

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
    
    # Prepare dataset using the modified CustomDataset
    try:
        dataset = CustomDataset(dataset_dict)
        print(f"Successfully created paired dataset with {len(dataset)} samples.")
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        return None # Exit if dataset creation fails

    if len(dataset) == 0:
        print("Error: Dataset is empty after processing. Exiting.")
        return None

    # Split dataset into training and validation
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Ensure val_size is not larger than dataset size if dataset is small
    if val_size <= 0 and dataset_size > 0:
         val_size = max(1, int(0.1 * dataset_size)) # Ensure at least 1 val sample or 10%
         train_size = dataset_size - val_size
         print(f"Adjusted validation size to {val_size} due to small dataset.")
    
    if train_size <= 0:
        print(f"Error: No training samples after split (Dataset size: {dataset_size}, Val size: {val_size}). Exiting.")
        return None

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) # Use generator for reproducibility
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Use shuffle=True for training loader
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    
    # Training loop
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Training epochs")
    best_val_loss = float('inf')

    for epoch in epoch_pbar:
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} (Train)", leave=False)
        
        for batch_idx, (images, texts, pos_actions, neg_actions) in enumerate(train_batch_pbar):
            # Move data to device
            images = images.to(device)
            # Tokenize texts *after* moving other data, reduces CPU overhead if texts are strings
            texts_tok = clip.tokenize(texts).to(device) 
            pos_actions = pos_actions.to(device).float() # Ensure actions are float
            neg_actions = neg_actions.to(device).float() # Ensure actions are float
            
            current_batch_size = images.shape[0] # Actual batch size (might be smaller for last batch)
            
            # Prepare inputs for the model: duplicate image/text, concatenate actions
            input_images = torch.cat([images, images], dim=0)           # Shape (2*B, C, H, W)
            input_texts = torch.cat([texts_tok, texts_tok], dim=0)      # Shape (2*B, SeqLen)
            input_actions = torch.cat([pos_actions, neg_actions], dim=0)# Shape (2*B, ActionDim)

            optimizer.zero_grad()
            
            # Forward pass
            logits_per_image, logits_per_action = model(input_images, input_texts, input_actions)
            # logits_per_image shape: (2*B, 2*B)
            # logits_per_action shape: (2*B, 2*B)

            # --- Loss Calculation ---
            # Ground truth for positive contrastive loss (aligns i-th image/text with i-th positive action)
            ground_truth = torch.arange(current_batch_size, dtype=torch.long, device=device)

            # Positive Contrastive Loss (Image/Text vs Positive Action)
            # Compare top-left block: (pos_img/txt features) vs (pos_action features)
            loss_i2t_pos = nn.CrossEntropyLoss()(logits_per_image[:current_batch_size, :current_batch_size], ground_truth)
            loss_a2t_pos = nn.CrossEntropyLoss()(logits_per_action[:current_batch_size, :current_batch_size], ground_truth)
            
            # Explicit Negative Loss (Image/Text vs Negative Action)
            # Compare top-right block: (pos_img/txt features) vs (neg_action features)
            # We want sim(pos_img/txt_i, neg_action_i) to be low.
            neg_logits_i2t = logits_per_image[:current_batch_size, current_batch_size:]
            neg_logits_a2t = logits_per_action[:current_batch_size, current_batch_size:]

            eps = 1e-6
            # Apply loss to all pairs in the block (push positive image/text away from all negative actions in batch)
            neg_loss_i2t = -torch.log(1 - torch.sigmoid(neg_logits_i2t) + eps).mean()
            neg_loss_a2t = -torch.log(1 - torch.sigmoid(neg_logits_a2t) + eps).mean()
            
            # Combine losses
            # Contrastive loss averages image->text and text->image directions
            # Negative loss averages image->text and text->image directions
            loss = (loss_i2t_pos + loss_a2t_pos) / 2 + neg_loss_weight * (neg_loss_i2t + neg_loss_a2t) / 2
                  
            loss.backward()
            
            # Gradient Clipping (optional but often helpful)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if use_wandb: wandb.log({"train_batch_loss": loss.item()}) # Log simplified batch loss
            train_batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        val_batch_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch} (Val)", leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, texts, pos_actions, neg_actions) in enumerate(val_batch_pbar):
                images = images.to(device)
                texts_tok = clip.tokenize(texts).to(device)
                pos_actions = pos_actions.to(device).float()
                neg_actions = neg_actions.to(device).float()
                
                current_batch_size = images.shape[0]

                input_images = torch.cat([images, images], dim=0)
                input_texts = torch.cat([texts_tok, texts_tok], dim=0)
                input_actions = torch.cat([pos_actions, neg_actions], dim=0)
                
                logits_per_image, logits_per_action = model(input_images, input_texts, input_actions)
                
                ground_truth = torch.arange(current_batch_size, dtype=torch.long, device=device)

                loss_i2t_pos = nn.CrossEntropyLoss()(logits_per_image[:current_batch_size, :current_batch_size], ground_truth)
                loss_a2t_pos = nn.CrossEntropyLoss()(logits_per_action[:current_batch_size, :current_batch_size], ground_truth)
                
                neg_logits_i2t = logits_per_image[:current_batch_size, current_batch_size:]
                neg_logits_a2t = logits_per_action[:current_batch_size, current_batch_size:]

                eps = 1e-6
                neg_loss_i2t = -torch.log(1 - torch.sigmoid(neg_logits_i2t) + eps).mean()
                neg_loss_a2t = -torch.log(1 - torch.sigmoid(neg_logits_a2t) + eps).mean()

                val_loss = (loss_i2t_pos + loss_a2t_pos) / 2 + neg_loss_weight * (neg_loss_i2t + neg_loss_a2t) / 2
                
                total_val_loss += val_loss.item()
                val_batch_pbar.set_postfix({'loss': f'{val_loss.item():.4f}'})
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        epoch_pbar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 'val_loss': f'{avg_val_loss:.4f}'})
        
        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, f"{save_name}_best.pt")
            # Save only the model state dict for the best model
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            if use_wandb: wandb.run.summary["best_val_loss"] = best_val_loss
        
        # Save regular checkpoint (consider saving optimizer state here too if resuming is important)
        # For simplicity, saving only model state_dict here.
        if (epoch + 1) % 50 == 0: # Save every 5 epochs
             checkpoint_path = os.path.join(checkpoint_dir, f"{save_name}_epoch_{epoch+1}.pt")
             torch.save(model.state_dict(), checkpoint_path)
             print(f"Checkpoint saved at {checkpoint_path}")

    if use_wandb: wandb.finish()
    
    # Return the model with the best validation weights loaded
    print(f"Loading best model weights from {best_model_path} before returning.")
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    except Exception as e:
        print(f"Warning: Could not load best model weights after training: {e}. Returning last state.")

    return model

def save_finetuned_clip(model, save_path):
    """Save the finetuned CLIP model"""
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CLIP model with paired positive/negative actions')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training') # Adjusted default maybe
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate') # Adjusted default maybe
    parser.add_argument('--checkpoint_dir', type=str, default='neg_model_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_name', type=str, default='vla_clip_neg', help='Name for saved model')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint state_dict to resume training from')
    parser.add_argument('--augmented_dataset', type=str, required=True, # Make augmented dataset path required
                        help='Path to augmented dataset pickle file')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--neg_loss_weight', type=float, default=0.2, help='Weight for the explicit negative loss term')

    args = parser.parse_args()
    
    # Import wandb only if needed
    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed. Running without wandb logging.")
            args.use_wandb = False
    # Load dataset
    if not os.path.exists(args.augmented_dataset):
         print(f"Error: Augmented dataset file not found at {args.augmented_dataset}")
         exit(1)
         
    print(f"Loading augmented dataset from {args.augmented_dataset}...")
    try:
        with open(args.augmented_dataset, 'rb') as f:
            dataset_dict = pickle.load(f)
        print(f"Loaded augmented dataset.")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        exit(1)

    # Create checkpoint directory
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
        validation_split=args.validation_split,
        neg_loss_weight=args.neg_loss_weight # Pass the weight
    )
    
    if finetuned_model:
        print("Saving final model (best validation weights)...")
        save_finetuned_clip(finetuned_model, SAVE_PATH)
        print("Done!")
    else:
        print("Training failed or was interrupted.")