import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import torch.hub
from PIL import Image
from typing import Optional
import h5py
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from model import TextAwareVisualExtraction, AttentionPooling, ModelConfig
import numpy as np
import pickle
import warnings

class CustomDataset(Dataset):
    def __init__(self, augmented_dataset_dict, history_length, image_keys=('agentview_rgb', 'eye_in_hand_rgb')):
        """
        Args:
            augmented_dataset_dict: Dictionary loaded from the augmented dataset pickle file.
                                Expected structure: {instruction: {'samples': [sample_dict, ...], ...}}
                                where sample_dict = {'images': {...}, 'pos_action_hist': ...}
            history_length: Expected length of action histories (H).
            image_keys: List of image keys to try in order (e.g., ['agentview_rgb', 'eye-in-hand_rgb'])
        """
        self.samples = []
        self.history_length = history_length
        self.image_keys = image_keys

        print("Processing augmented dataset with histories...")
        for instruction, data in tqdm(augmented_dataset_dict.items(), desc="Loading instructions"):
            instruction_samples = data.get('samples', [])
            if not instruction_samples:
                continue

            for sample_data in instruction_samples:
                images_dict = sample_data.get('images')
                pos_hist = sample_data.get('pos_action_hist')

                # Get both agentview and handview images
                try:
                    img1 = images_dict[self.image_keys[0]]
                    img2 = images_dict[self.image_keys[1]]
                except Exception:
                    continue  # Skip if either view is missing

                if img1 is None or img2 is None or pos_hist is None:
                    continue
                if pos_hist.shape[0] != self.history_length:
                    warnings.warn(f"Incorrect history length for instruction '{instruction}'. Skipping sample.")
                    continue
                if pos_hist.ndim != 2:
                    warnings.warn(f"Incorrect action history dimensions for instruction '{instruction}'. Skipping sample.")
                    continue

                self.samples.append((img1, img2, instruction, pos_hist))

        print(f"Created dataset with {len(self.samples)} (agentview, handview, instruction, pos_hist) samples.")
        if not self.samples:
            raise ValueError("Dataset creation resulted in 0 samples. Check input data format and history length.")

        # DINO v2 image preprocessing pipeline (similar to ImageNet normalization)
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1, img2, caption, pos_action_hist = self.samples[idx]
        img1 = Image.fromarray(img1.astype('uint8'))
        img2 = Image.fromarray(img2.astype('uint8'))
        img1 = self.preprocess(img1)
        img2 = self.preprocess(img2)
        
        # Return raw caption - we'll tokenize in the training loop to handle batching properly
        return img1, img2, caption, pos_action_hist


class VLA_BERT_DINO(nn.Module):
    def __init__(self, model_config, use_transformer=False):
        super().__init__()
        
        # Initialize BERT for text encoding
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Initialize DINO v2 for image encoding
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # Freeze DINO parameters
        for param in self.dino.parameters():
            param.requires_grad = False
            
        # Model dimensions
        text_pooling_output_dim = model_config.text_pooling_output_dim
        pooling_heads = model_config.pooling_heads
        pooling_layers = model_config.pooling_layers
        self.num_readouts = model_config.num_readouts
        
        # BERT outputs 768-dim features, DINO v2 ViT-B/14 outputs 768-dim features
        text_dim = 768  # BERT base dimension
        vision_dim = 768  # DINO v2 ViT-B/14 dimension
        vision_pooling_output_dim = model_config.vision_pooling_output_dim
        
        # DINO v2 ViT-B/14 - determine actual number of patches dynamically
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_features = self.dino.get_intermediate_layers(dummy_input, n=1)[0]
            self.num_img_patches = dummy_features.shape[1] - 1  # Subtract 1 for CLS token
        
        print(f"DINO v2 actual number of patches: {self.num_img_patches}")
        
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # ln(1/0.07) for better initialization
        
        # Text-aware visual extraction adapted for DINO features
        self.text_aware_visual_extraction = TextAwareVisualExtraction(
            num_img_patches=self.num_img_patches,
            vision_dim=vision_dim,
        )
        
        # Pooling components
        self.text_pooling = AttentionPooling(
            text_dim, 
            text_pooling_output_dim,
            pooling_heads,
            pooling_layers, 
            num_readouts=self.num_readouts,
        )        
        self.agentview_vision_poolings = AttentionPooling(
            vision_dim,
            vision_pooling_output_dim,
            pooling_heads, 
            pooling_layers, 
            num_readouts=self.num_readouts
        )
        self.handview_vision_poolings = AttentionPooling(
            vision_dim,
            vision_pooling_output_dim,
            pooling_heads, 
            pooling_layers, 
            num_readouts=self.num_readouts
        )
        self.f_t_dim = text_pooling_output_dim + vision_pooling_output_dim
        
        self.input_projection = nn.Linear(self.f_t_dim, vision_pooling_output_dim)
        
        # Multi-view fusion
        self.image_fusion_proj = nn.Sequential(
            nn.Linear(2 * vision_pooling_output_dim, 2 * vision_pooling_output_dim),
            nn.ReLU(),
            nn.Linear(2 * vision_pooling_output_dim, vision_pooling_output_dim)
        )
        
        # Action trajectory processing components
        self.action_dim = model_config.action_dim
        self.history_length = model_config.history_length
        self.use_transformer = use_transformer

        if self.use_transformer:
            self.single_step_action_encoder = nn.Linear(self.action_dim, vision_pooling_output_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                    d_model=vision_pooling_output_dim,
                    nhead=8,
                    dim_feedforward=vision_pooling_output_dim * 2,
                    dropout=0.1
                )
            self.trajectory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        else:
            self.complex_action_encoder = nn.Sequential(
                nn.Linear(self.history_length * self.action_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, vision_pooling_output_dim)
            )

        self.action_padding_value = -5.0

    def extract_bert_features(self, tokenized_text):
        """
        Extract features from BERT model
        
        Args:
            tokenized_text: Dict with 'input_ids', 'attention_mask', etc.
            
        Returns:
            text_features: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']
        
        # Note: BERT parameters are frozen in __init__, but we need gradients on outputs
        # for downstream trainable components
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        
        return text_features

    def extract_dino_features(self, images):
        """
        Extract patch-level features from DINO v2 model
        
        Args:
            images: Tensor of shape (batch_size, C, H, W)
            
        Returns:
            patch_features: Tensor of shape (batch_size, num_patches, embedding_dim)
        """
        # Note: DINO parameters are frozen in __init__, but we need gradients on outputs
        # for downstream trainable components
        all_tokens = self.dino.get_intermediate_layers(images, n=1)[0]
        # Remove CLS token (first token) to get only patch features
        patch_features = all_tokens[:, 1:, :]  # (batch_size, num_patches, embed_dim)
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True) # normalize patch features
        
        return patch_features
        
    def forward(self, image1, image2, tokenized_text, action_histories):
        """
        Args:
            image1: Tensor (B, C, H, W)
            image2: Tensor (B, C, H, W)
            tokenized_text: Dict with tokenized text from BERT tokenizer
            action_histories: Tensor (B, H, D) - Batch of action histories, potentially padded.
        """
        # Extract image features from DINO
        patch_features1 = self.extract_dino_features(image1)
        patch_features2 = self.extract_dino_features(image2)
        
        # Extract text features from BERT
        text_features = self.extract_bert_features(tokenized_text)
        
        # Text-aware visual features for both views - RE-ENABLE THIS!
        text_aware_features1 = self.text_aware_visual_extraction(patch_features1, text_features)
        text_aware_features2 = self.text_aware_visual_extraction(patch_features2, text_features)
        
        # Vision pooling
        vision_token1 = self.agentview_vision_poolings(text_aware_features1)
        vision_token2 = self.handview_vision_poolings(text_aware_features2)
        
        # Concatenate and project
        fused_vision = torch.cat([vision_token1, vision_token2], dim=-1)
        fused_vision = self.image_fusion_proj(fused_vision)
        
        # Text pooling
        text_token = self.text_pooling(text_features)
        combined_features = torch.cat([text_token, fused_vision], dim=-1)
        combined_features = self.input_projection(combined_features)
        combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

        # Encode Action History
        action_histories = action_histories.float().to(image1.device)

        if self.use_transformer:
            # Transformer Path
            padding_mask = (action_histories[:, :, 0] == self.action_padding_value)
            encoded_steps = self.single_step_action_encoder(action_histories)
            encoded_steps_permuted = encoded_steps.permute(1, 0, 2)
            transformer_output_permuted = self.trajectory_encoder(encoded_steps_permuted, src_key_padding_mask=padding_mask)
            transformer_output = transformer_output_permuted.permute(1, 0, 2)
            
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            summed_features = (transformer_output * mask_expanded).sum(dim=1)
            num_non_padded = mask_expanded.sum(dim=1)
            num_non_padded = torch.clamp(num_non_padded, min=1e-9)
            projected_trajectory = summed_features / num_non_padded
        else:
            # MLP Path
            batch_size = action_histories.shape[0]
            flat_actions = action_histories.reshape(batch_size, -1)
            projected_trajectory = self.complex_action_encoder(flat_actions)

        # Normalize action history features
        projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)

        logits_scale = self.logit_scale.exp()
        # Calculate logits
        image_logits = torch.matmul(combined_features, projected_trajectory.T) * logits_scale
        action_logits = torch.matmul(projected_trajectory, combined_features.T) * logits_scale

        return image_logits, action_logits

    def get_similarity_score(self, image1, image2, tokenized_text, action_histories):
        """
        Computes the unscaled cosine similarity for inference.
        Args:
            image1: Tensor (B, C, H, W)
            image2: Tensor (B, C, H, W)
            tokenized_text: Dict with tokenized text from BERT tokenizer
            action_histories: Tensor (B, H, D) - Batch of action histories.
        Returns:
            similarity: Tensor (B,) - The cosine similarity for each sample in the batch.
        """
        # Extract image features from DINO
        patch_features1 = self.extract_dino_features(image1)
        patch_features2 = self.extract_dino_features(image2)
        
        # Extract text features from BERT
        text_features = self.extract_bert_features(tokenized_text)
        
        # Text-aware visual features
        text_aware_features1 = self.text_aware_visual_extraction(patch_features1, text_features)
        text_aware_features2 = self.text_aware_visual_extraction(patch_features2, text_features)
        
        # Vision pooling
        vision_token1 = self.agentview_vision_poolings(text_aware_features1)
        vision_token2 = self.handview_vision_poolings(text_aware_features2)
        
        # Concatenate and project
        fused_vision = torch.cat([vision_token1, vision_token2], dim=-1)
        fused_vision = self.image_fusion_proj(fused_vision)
        
        # Text pooling
        text_token = self.text_pooling(text_features)
        combined_features = torch.cat([text_token, fused_vision], dim=-1)
        combined_features = self.input_projection(combined_features)
        combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

        # Encode Action History
        action_histories = action_histories.float().to(image1.device)

        if self.use_transformer:
            # Transformer Path
            padding_mask = (action_histories[:, :, 0] == self.action_padding_value)
            encoded_steps = self.single_step_action_encoder(action_histories)
            encoded_steps_permuted = encoded_steps.permute(1, 0, 2)
            transformer_output_permuted = self.trajectory_encoder(encoded_steps_permuted, src_key_padding_mask=padding_mask)
            transformer_output = transformer_output_permuted.permute(1, 0, 2)
            
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            summed_features = (transformer_output * mask_expanded).sum(dim=1)
            num_non_padded = mask_expanded.sum(dim=1)
            num_non_padded = torch.clamp(num_non_padded, min=1e-9)
            projected_trajectory = summed_features / num_non_padded
        else:
            # MLP Path
            batch_size = action_histories.shape[0]
            flat_actions = action_histories.reshape(batch_size, -1)
            projected_trajectory = self.complex_action_encoder(flat_actions)

        # Normalize action history features
        projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity (dot product of normalized vectors) for each sample
        similarity = (combined_features * projected_trajectory).sum(dim=-1)

        return similarity

def train_clip(
    augmented_dataset_dict: dict,
    history_length: int,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-6,
    validation_split: float = 0.1,
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
        # Ensure save_name is set if using wandb
        if save_name is None:
             save_name = f"vla_bert_dino_traj_h{history_length}_{'transformer' if use_transformer else 'mlp'}"
             print(f"Generated save_name for wandb: {save_name}")

        wandb.init(project="VLA-BERT-DINO-Trajectory", name=save_name)
        wandb.config.update({
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "device": device,
            "history_length": history_length,
            "use_transformer": use_transformer,
            "validation_split": validation_split,
        })

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if save_name is None: # Ensure save_name is set even if not using wandb
         save_name = f"vla_bert_dino_traj_h{history_length}_{'transformer' if use_transformer else 'mlp'}_local"

    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_text_length = 77
    
    # Create model configuration for BERT/DINO (no need to load CLIP)
    model_config = ModelConfig(clip_model=None, history_length=history_length, text_pooling_output_dim=768, vision_pooling_output_dim=768)  # Pass None since we're not using CLIP
    
    # Initialize the model
    model = VLA_BERT_DINO(model_config, use_transformer=use_transformer).float().to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) # Filter frozen params
    start_epoch = 0

    # --- [Checkpoint Loading Logic - Adapted from finetune_negative.py] ---
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading model state dict from {resume_checkpoint}")
        try:
             model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
             print("Successfully loaded model weights.")
             # Consider loading optimizer state and epoch if saved in checkpoint
             start_epoch = 6
        except Exception as load_err:
              print(f"Error loading checkpoint: {load_err}. Starting training from scratch.")
              start_epoch = 0
    # --- [End Checkpoint Loading] ---

    # Print model size and details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Prepare dataset using the modified CustomDataset
    try:
        dataset = CustomDataset(augmented_dataset_dict, history_length=history_length)
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        return None

    # --- [Train/Validation Split Logic - Adapted from finetune_negative.py] ---
    dataset_size = len(dataset)
    if dataset_size == 0:
        print("Error: Dataset is empty. Exiting.")
        return None
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    if val_size <= 0 and dataset_size > 0:
         val_size = max(1, int(0.1 * dataset_size)) # Ensure at least 1 val sample
         train_size = dataset_size - val_size
         print(f"Adjusted validation size to {val_size} due to small dataset.")

    if train_size <= 0:
        print(f"Error: No training samples after split (Dataset size: {dataset_size}, Val size: {val_size}). Exiting.")
        return None

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Training loop
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Training epochs")
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, f"{save_name}_best.pt") # Define best model path

    for epoch in epoch_pbar:
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} (Train)", leave=False)

        for batch_idx, (img1, img2, texts, pos_hists) in enumerate(train_batch_pbar):
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Tokenize text with BERT tokenizer
            tokenized = tokenizer(
                list(texts),  # Convert tuple to list
                padding='max_length',
                truncation=True,
                max_length=max_text_length,
                return_tensors='pt'
            )
            
            # Move tokenized text to device
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            
            input_actions = torch.tensor(np.array(pos_hists), dtype=torch.float32, device=device)
            current_batch_size = img1.shape[0]
            optimizer.zero_grad()
            logits_per_image, logits_per_action = model(img1, img2, tokenized, input_actions)
            positive_labels = torch.arange(current_batch_size, device=device)
            loss = (F.cross_entropy(logits_per_image, positive_labels) +
                    F.cross_entropy(logits_per_action, positive_labels)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            train_batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        val_batch_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch} (Val)", leave=False)

        with torch.no_grad():
            for batch_idx, (img1, img2, texts, pos_hists) in enumerate(val_batch_pbar):
                img1 = img1.to(device)
                img2 = img2.to(device)
                
                # Tokenize text with BERT tokenizer
                tokenized = tokenizer(
                    list(texts),  # Convert tuple to list
                    padding='max_length',
                    truncation=True,
                    max_length=max_text_length,
                    return_tensors='pt'
                )
                
                # Move tokenized text to device
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                
                pos_hists = torch.tensor(np.array(pos_hists), dtype=torch.float32, device=device)
                current_batch_size = img1.shape[0]
                logits_per_image, logits_per_action = model(img1, img2, tokenized, pos_hists)
                positive_labels = torch.arange(current_batch_size, device=device)
                val_loss = (F.cross_entropy(logits_per_image, positive_labels) +
                            F.cross_entropy(logits_per_action, positive_labels)) / 2
                total_val_loss += val_loss.item()
                val_batch_pbar.set_postfix({'loss': f'{val_loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(val_dataloader)

        epoch_pbar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 'val_loss': f'{avg_val_loss:.4f}'})

        if use_wandb:
            wandb.log({"epoch": epoch,
                       "train_loss": avg_train_loss,
                       "val_loss": avg_val_loss})

        # --- Save Best Model Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved with validation loss: {best_val_loss:.4f} at {best_model_path}")
            if use_wandb: wandb.run.summary["best_val_loss"] = best_val_loss
        # --- End Save Best Model ---

        # Optionally save periodic checkpoints
        if (epoch + 1) % 1 == 0:
             checkpoint_path = os.path.join(checkpoint_dir, f"{save_name}_epoch_{epoch+1}.pt")
             torch.save(model.state_dict(), checkpoint_path)
             print(f"Checkpoint saved at {checkpoint_path}")

    if use_wandb: wandb.finish()

    # Load best model weights before returning
    if os.path.exists(best_model_path):
         print(f"Loading best model weights from {best_model_path}")
         try:
             model.load_state_dict(torch.load(best_model_path, map_location=device))
         except Exception as e:
             print(f"Warning: Could not load best model weights after training: {e}. Returning last state.")
    else:
         print("Warning: Best model checkpoint not found. Returning last state.")

    return model

def save_finetuned_model(model, save_path):
    """Save the finetuned BERT-DINO model state dict"""
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VLA-BERT-DINO model with action trajectories and contrastive loss')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs') # Increased default
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training') # Adjusted default
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate') # Adjusted default
    parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of data to use for validation')

    # Model parameters
    parser.add_argument('--history_length', type=int, required=True, help='Action history length (must match dataset)')
    parser.add_argument('--use_transformer', action='store_true', help='Use transformer for action history encoding instead of MLP')

    # Dataset and paths
    parser.add_argument('--augmented_dataset', type=str, required=True, help='Path to augmented dataset pickle file (with histories)')
    parser.add_argument('--checkpoint_dir', type=str, default='trajectory_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_name', type=str, default=None, help='Name for saved model and wandb run (generated if None)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint state_dict to resume training from')

    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')

    args = parser.parse_args()

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

    print(f"Loading augmented dataset from {args.augmented_dataset}...")
    try:
        with open(args.augmented_dataset, 'rb') as f:
            augmented_dataset_dict = pickle.load(f)
        print(f"Loaded augmented dataset with {len(augmented_dataset_dict)} instructions.")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        exit(1)

    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Final save path (best model)
    if args.save_name is None:
         # Generate a default name if not provided
         args.save_name = f"vla_bert_dino_traj_h{args.history_length}_{'transformer' if args.use_transformer else 'mlp'}"
    FINAL_SAVE_PATH = os.path.join(args.checkpoint_dir, f"{args.save_name}_final_best.pt")

    print("Starting training...")
    print(f"Config: History={args.history_length}, ActionEncoder={'Transformer' if args.use_transformer else 'MLP'}, LR={args.lr}, BS={args.batch_size}")
    print(f"Using wandb: {args.use_wandb}")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")

    finetuned_model = train_clip(
        augmented_dataset_dict=augmented_dataset_dict,
        history_length=args.history_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validation_split=args.validation_split,
        save_name=args.save_name,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        resume_checkpoint=args.resume,
        use_transformer=args.use_transformer
    )

    if finetuned_model:
        print(f"Saving final model (best validation weights) to {FINAL_SAVE_PATH}...")
        save_finetuned_model(finetuned_model, FINAL_SAVE_PATH)
        print("Done!")
    else:
        print("Training failed or was interrupted.")