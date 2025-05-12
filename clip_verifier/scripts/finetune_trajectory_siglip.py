import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional
import h5py
import os
from tqdm import tqdm
from model import TextAwareVisualExtraction, AttentionPooling, ModelConfig
import numpy as np
import pickle
import warnings
from transformers import SiglipProcessor, SiglipModel

class CustomDataset(Dataset):
    def __init__(self, augmented_dataset_dict, history_length, processor):
        self.samples = []
        self.history_length = history_length
        self.processor = processor
        print("Processing augmented dataset with histories...")
        for instruction, data in tqdm(augmented_dataset_dict.items(), desc="Loading instructions"):
            instruction_samples = data.get('samples', [])
            if not instruction_samples:
                continue
            for sample_data in instruction_samples:
                image = sample_data.get('image')
                pos_hist = sample_data.get('pos_action_hist')
                neg_hist = sample_data.get('neg_action_hist')
                if image is None or pos_hist is None or neg_hist is None:
                    continue
                if pos_hist.shape[0] != self.history_length or neg_hist.shape[0] != self.history_length:
                    warnings.warn(f"Incorrect history length found for instruction '{instruction}'. "
                                  f"Expected {self.history_length}, got pos={pos_hist.shape[0]}, neg={neg_hist.shape[0]}. Skipping sample.")
                    continue
                if pos_hist.ndim != 2 or neg_hist.ndim != 2:
                    warnings.warn(f"Incorrect action history dimensions for instruction '{instruction}'. "
                                  f"Expected 2, got pos={pos_hist.ndim}, neg={neg_hist.ndim}. Skipping sample.")
                    continue
                self.samples.append((image, instruction, pos_hist, neg_hist))
        print(f"Created dataset with {len(self.samples)} (image, instruction, pos_hist, neg_hist) samples.")
        if not self.samples:
            raise ValueError("Dataset creation resulted in 0 samples. Check input data format and history length.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, caption, pos_action_hist, neg_action_hist = self.samples[idx]
        # Convert numpy array to PIL Image
        image = Image.fromarray(image.astype('uint8'))
        # Use SigLIP processor for image preprocessing
        image_tensor = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        return image_tensor, caption, pos_action_hist, neg_action_hist

class VLA_SigLIP(nn.Module):
    def __init__(self, model_config, use_transformer=False):
        super().__init__()
        self.siglip = model_config.siglip_model
        self.processor = model_config.processor
        for param in self.siglip.parameters():
            param.requires_grad = False
        text_pooling_output_dim = model_config.text_pooling_output_dim
        pooling_heads = model_config.pooling_heads
        pooling_layers = model_config.pooling_layers
        self.num_readouts = model_config.num_readouts
        # SigLIP dims
        text_dim = self.siglip.config.text_config.hidden_size
        vision_dim = self.siglip.config.vision_config.hidden_size
        vision_pooling_output_dim = model_config.vision_pooling_output_dim
        self.visual_patch_size = self.siglip.config.vision_config.patch_size
        self.num_img_patches = (224 // self.visual_patch_size) ** 2
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.text_aware_visual_extraction = TextAwareVisualExtraction(
            num_img_patches=self.num_img_patches,
            vision_dim=vision_dim,
        )
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

    def extract_siglip_features(self, images, texts):
        # images: (B, C, H, W), texts: list of str
        with torch.no_grad():
            inputs = self.processor(text=texts, images=None, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(images.device) for k, v in inputs.items() if k.startswith('input_') or k == 'attention_mask'}
            outputs = self.siglip(pixel_values=images, **text_inputs, output_hidden_states=True)
            # outputs.image_embeds: (B, D), outputs.text_embeds: (B, D)
            # outputs.vision_model_output.last_hidden_state: (B, num_patches+1, D)
            # outputs.text_model_output.last_hidden_state: (B, seq_len, D)
            patch_features = outputs.vision_model_output.last_hidden_state[:, 1:, :]
            text_features = outputs.text_model_output.last_hidden_state
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return patch_features, text_features

    def forward(self, image, texts, action_histories):
        patch_features, text_features = self.extract_siglip_features(image, texts)
        text_aware_features = self.text_aware_visual_extraction(patch_features, text_features)
        vision_token = self.vision_poolings(text_aware_features)
        text_token = self.text_pooling(text_features)
        combined_features = torch.cat([text_token, vision_token], dim=-1)
        combined_features = self.input_projection(combined_features)
        combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
        action_histories = action_histories.float().to(image.device)
        if self.use_transformer:
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
            batch_size = action_histories.shape[0]
            flat_actions = action_histories.reshape(batch_size, -1)
            projected_trajectory = self.complex_action_encoder(flat_actions)
        projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)
        logits_scale = self.logit_scale.exp()
        image_logits = logits_scale * torch.matmul(combined_features, projected_trajectory.T)
        action_logits = logits_scale * torch.matmul(projected_trajectory, combined_features.T)
        return image_logits, action_logits

def train_siglip(
    augmented_dataset_dict: dict,
    history_length: int,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-6,
    neg_loss_weight: float = 0.5,
    validation_split: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_name = None,
    checkpoint_dir = "checkpoints",
    use_wandb = False,
    resume_checkpoint = None,
    use_transformer = False
):
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    siglip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device)
    model_config = ModelConfig(siglip_model=siglip_model, processor=processor, history_length=history_length)
    model = VLA_SigLIP(model_config, use_transformer=use_transformer).float().to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    start_epoch = 0

    # --- wandb logging ---
    if use_wandb:
        import wandb
        if save_name is None:
            save_name = f"vla_siglip_traj_h{history_length}_{'transformer' if use_transformer else 'mlp'}"
            print(f"Generated save_name for wandb: {save_name}")
        wandb.init(project="VLA-SigLIP-Trajectory", name=save_name)
        wandb.config.update({
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "device": device,
            "history_length": history_length,
            "use_transformer": use_transformer,
            "neg_loss_weight": neg_loss_weight,
            "validation_split": validation_split,
        })
    # --- end wandb logging ---

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading model state dict from {resume_checkpoint}")
        try:
            model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
            print("Successfully loaded model weights.")
            start_epoch = 300
        except Exception as load_err:
            print(f"Error loading checkpoint: {load_err}. Starting training from scratch.")
            start_epoch = 0
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    try:
        dataset = CustomDataset(augmented_dataset_dict, history_length=history_length, processor=processor)
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        return None
    dataset_size = len(dataset)
    if dataset_size == 0:
        print("Error: Dataset is empty. Exiting.")
        return None
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    if val_size <= 0 and dataset_size > 0:
        val_size = max(1, int(0.1 * dataset_size))
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    positive_loss_fn = nn.BCEWithLogitsLoss()
    negative_loss_fn = nn.BCEWithLogitsLoss()
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Training epochs")
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, f"{save_name}_best.pt")
    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0
        train_batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} (Train)", leave=False)
        for batch_idx, (images, texts, pos_hists, neg_hists) in enumerate(train_batch_pbar):
            images = images.to(device)
            pos_hists = torch.tensor(np.array(pos_hists), dtype=torch.float32, device=device)
            neg_hists = torch.tensor(np.array(neg_hists), dtype=torch.float32, device=device)
            current_batch_size = images.shape[0]
            input_images = torch.cat([images, images], dim=0)
            input_texts = texts + texts
            input_actions = torch.cat([pos_hists, neg_hists], dim=0)
            optimizer.zero_grad()
            logits_per_image, logits_per_action = model(input_images, input_texts, input_actions)
            positive_labels = torch.arange(current_batch_size, device=device)
            loss_pos = (F.cross_entropy(logits_per_image[:current_batch_size, :current_batch_size], positive_labels) +
                        F.cross_entropy(logits_per_action[:current_batch_size, :current_batch_size], positive_labels)) / 2
            neg_logits_i2t = logits_per_image[:current_batch_size, current_batch_size:]
            neg_logits_a2t = logits_per_action[:current_batch_size, current_batch_size:]
            negative_targets = torch.zeros((current_batch_size, current_batch_size), device=device)
            loss_neg = (negative_loss_fn(neg_logits_i2t, negative_targets) +
                        negative_loss_fn(neg_logits_a2t, negative_targets)) / 2
            loss = loss_pos + neg_loss_weight * loss_neg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            train_pos_loss_item = loss_pos.item()
            train_neg_loss_item = loss_neg.item()
            if use_wandb:
                wandb.log({
                    "train_batch_loss": loss.item(),
                    "train_pos_loss": train_pos_loss_item,
                    "train_neg_loss": train_neg_loss_item
                })
            train_batch_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'pos': f'{train_pos_loss_item:.4f}', 'neg': f'{train_neg_loss_item:.4f}'})
        avg_train_loss = total_train_loss / len(train_dataloader)
        model.eval()
        total_val_loss = 0
        total_val_pos_loss = 0
        total_val_neg_loss = 0
        val_batch_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch} (Val)", leave=False)
        with torch.no_grad():
            for batch_idx, (images, texts, pos_hists, neg_hists) in enumerate(val_batch_pbar):
                images = images.to(device)
                pos_hists = torch.tensor(np.array(pos_hists), dtype=torch.float32, device=device)
                neg_hists = torch.tensor(np.array(neg_hists), dtype=torch.float32, device=device)
                current_batch_size = images.shape[0]
                input_images = torch.cat([images, images], dim=0)
                input_texts = texts + texts
                input_actions = torch.cat([pos_hists, neg_hists], dim=0)
                logits_per_image, logits_per_action = model(input_images, input_texts, input_actions)
                positive_labels = torch.arange(current_batch_size, device=device)
                loss_pos = (F.cross_entropy(logits_per_image[:current_batch_size, :current_batch_size], positive_labels) +
                            F.cross_entropy(logits_per_action[:current_batch_size, :current_batch_size], positive_labels)) / 2
                neg_logits_i2t = logits_per_image[:current_batch_size, current_batch_size:]
                neg_logits_a2t = logits_per_action[:current_batch_size, current_batch_size:]
                negative_targets = torch.zeros((current_batch_size, current_batch_size), device=device)
                loss_neg = (negative_loss_fn(neg_logits_i2t, negative_targets) +
                            negative_loss_fn(neg_logits_a2t, negative_targets)) / 2
                val_loss = loss_pos + neg_loss_weight * loss_neg
                total_val_loss += val_loss.item()
                total_val_pos_loss += loss_pos.item()
                total_val_neg_loss += loss_neg.item()
                if use_wandb:
                    wandb.log({
                        "val_batch_loss": val_loss.item(),
                        "val_pos_loss": loss_pos.item(),
                        "val_neg_loss": loss_neg.item()
                    })
                val_batch_pbar.set_postfix({'loss': f'{val_loss.item():.4f}'})
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_pos_loss = total_val_pos_loss / len(val_dataloader)
        avg_val_neg_loss = total_val_neg_loss / len(val_dataloader)
        epoch_pbar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 'val_loss': f'{avg_val_loss:.4f}'})
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_pos_loss": avg_val_pos_loss,
                "val_neg_loss": avg_val_neg_loss
            })
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f} at {best_model_path}")
            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{save_name}_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    if use_wandb:
        wandb.finish()
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from {best_model_path}")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        except Exception as e:
            print(f"Warning: Could not load best model weights after training: {e}. Returning last state.")
    else:
        print("Warning: Best model checkpoint not found. Returning last state.")
    return model

def save_finetuned_siglip(model, save_path):
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VLA-SigLIP model with action trajectories and contrastive loss')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--neg_loss_weight', type=float, default=2, help='Weight for the explicit negative loss term')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--history_length', type=int, required=True, help='Action history length (must match dataset)')
    parser.add_argument('--use_transformer', action='store_true', help='Use transformer for action history encoding instead of MLP')
    parser.add_argument('--augmented_dataset', type=str, required=True, help='Path to augmented dataset pickle file (with histories)')
    parser.add_argument('--checkpoint_dir', type=str, default='trajectory_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_name', type=str, default=None, help='Name for saved model and run (generated if None)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint state_dict to resume training from')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    args = parser.parse_args()
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
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if args.save_name is None:
        args.save_name = f"vla_siglip_traj_h{args.history_length}_{'transformer' if args.use_transformer else 'mlp'}"
    FINAL_SAVE_PATH = os.path.join(args.checkpoint_dir, f"{args.save_name}_final_best.pt")
    print("Starting training...")
    print(f"Config: History={args.history_length}, ActionEncoder={'Transformer' if args.use_transformer else 'MLP'}, LR={args.lr}, BS={args.batch_size}, NegWeight={args.neg_loss_weight}")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
    finetuned_model = train_siglip(
        augmented_dataset_dict=augmented_dataset_dict,
        history_length=args.history_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        neg_loss_weight=args.neg_loss_weight,
        validation_split=args.validation_split,
        save_name=args.save_name,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        resume_checkpoint=args.resume,
        use_transformer=args.use_transformer
    )
    if finetuned_model:
        print(f"Saving final model (best validation weights) to {FINAL_SAVE_PATH}...")
        save_finetuned_siglip(finetuned_model, FINAL_SAVE_PATH)
        print("Done!")
    else:
        print("Training failed or was interrupted.")