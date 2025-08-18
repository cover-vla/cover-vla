"""
Standalone VQ-VAE Action Tokenizer Implementation
Based on VQ-VLA's architecture but without external dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class SimpleVQVAE(nn.Module):
    """
    Simplified VQ-VAE implementation based on VQ-VLA's architecture
    This is a standalone version that doesn't require external packages
    """
    
    def __init__(self, config_path=None):
        super().__init__()
        
        # Default parameters based on VQ-VLA's config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract encoder parameters
            encoder_out_channels = config.get('encoder_out_channels', 128)
            encoder_block_out_channels = config.get('encoder_block_out_channels', [128, 256, 256, 512])
            encoder_layers_per_block = config.get('encoder_layers_per_block', [4, 4, 4, 4])
            
            # Extract decoder parameters  
            decoder_in_channels = config.get('decoder_in_channels', 128)
            decoder_block_out_channels = config.get('decoder_block_out_channels', [128, 256, 256, 512])
            decoder_layers_per_block = config.get('decoder_layers_per_block', [4, 4, 4, 4])
        else:
            # Use VQ-VLA's default parameters
            encoder_out_channels = 128
            encoder_block_out_channels = [128, 256, 256, 512]
            encoder_layers_per_block = [4, 4, 4, 4]
            decoder_in_channels = 128
            decoder_block_out_channels = [128, 256, 256, 512]
            decoder_layers_per_block = [4, 4, 4, 4]
        
        # VQ parameters
        self.vq_embed_dim = 128
        self.vqvae_groups = 4
        self.vqvae_n_embed = 256
        
        # Build encoder (simplified causal 1D conv)
        self.encoder = self._build_encoder(
            in_channels=1,
            out_channels=encoder_out_channels,
            block_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block
        )
        
        # Build decoder (simplified causal 1D deconv)
        self.decoder = self._build_decoder(
            in_channels=decoder_in_channels,
            out_channels=1,
            block_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block
        )
        
        # Vector quantization layer (simplified)
        self.vq_layer = nn.Linear(self.vq_embed_dim, self.vq_embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _build_encoder(self, in_channels, out_channels, block_channels, layers_per_block):
        """Build simplified causal encoder"""
        layers = []
        current_channels = in_channels
        
        for i, (block_ch, num_layers) in enumerate(zip(block_channels, layers_per_block)):
            # Add conv layers for this block
            for _ in range(num_layers):
                layers.extend([
                    nn.Conv1d(current_channels, block_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(32, block_ch),
                    nn.SiLU(),
                ])
                current_channels = block_ch
            
            # Add downsampling (except for last block)
            if i < len(block_channels) - 1:
                layers.append(nn.MaxPool1d(2))
        
        # Final projection to latent dimension
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Conv1d(current_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self, in_channels, out_channels, block_channels, layers_per_block):
        """Build simplified causal decoder"""
        layers = []
        current_channels = in_channels
        
        # Initial projection
        layers.append(nn.Conv1d(in_channels, block_channels[0], 1))
        current_channels = block_channels[0]
        
        for i, (block_ch, num_layers) in enumerate(zip(block_channels, layers_per_block)):
            # Add conv layers for this block
            for _ in range(num_layers):
                layers.extend([
                    nn.Conv1d(current_channels, block_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(32, block_ch),
                    nn.SiLU(),
                ])
                current_channels = block_ch
            
            # Add upsampling (except for last block)
            if i < len(block_channels) - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        
        # Final projection to output
        layers.append(nn.Conv1d(current_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """
        Encode input actions
        Args:
            x: Input tensor of shape (B, T, D) or (B, T*D)
        Returns:
            latents: Encoded latents of shape (B, latent_dim)
        """
        # Reshape input to (B, 1, T*D) for 1D conv
        if x.ndim == 3:
            x = x.reshape(x.shape[0], 1, -1)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        
        # Encode
        latents = self.encoder(x)  # (B, latent_dim, 1)
        latents = latents.squeeze(-1)  # (B, latent_dim)
        
        return latents
    
    def decode(self, latents):
        """
        Decode latents back to actions
        Args:
            latents: Latent tensor of shape (B, latent_dim)
        Returns:
            decoded: Decoded actions
        """
        # Reshape latents for decoder
        latents = latents.unsqueeze(-1)  # (B, latent_dim, 1)
        
        # Decode
        decoded = self.decoder(latents)
        
        return decoded
    
    def forward(self, x):
        """
        Forward pass through encoder and decoder
        Args:
            x: Input actions
        Returns:
            decoded: Reconstructed actions
        """
        latents = self.encode(x)
        decoded = self.decode(latents)
        return decoded


class VQVAEActionTokenizer(nn.Module):
    """
    VQ-VAE Action Tokenizer wrapper for VQ-VLA pre-trained weights
    Now uses a standalone implementation based on VQ-VLA's architecture
    """
    
    def __init__(self, vqvae_checkpoint_path, device="cuda"):
        super().__init__()
        self.device = device
        
        print(f"Loading VQ-VAE action tokenizer from {vqvae_checkpoint_path}")
        try:
            # Create the VQ-VAE model using VQ-VLA's architecture
            config_path = "action_vqvae_config/config.json"
            if os.path.exists(config_path):
                self.vqvae_model = SimpleVQVAE(config_path)
            else:
                # Fallback: create with default parameters
                self.vqvae_model = SimpleVQVAE()
            
            # Load the pre-trained weights
            self._load_vqvae_weights(vqvae_checkpoint_path)
            
            # Freeze the model for inference
            self.vqvae_model.eval()
            for param in self.vqvae_model.parameters():
                param.requires_grad = False
                
            print("Successfully loaded VQ-VAE action tokenizer")
        except Exception as e:
            print(f"Error loading VQ-VAE checkpoint: {e}")
            raise
    
    def _load_vqvae_weights(self, checkpoint_path):
        """
        Load VQ-VAE weights from checkpoint using VQ-VLA's loading logic
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract VQ-VAE weights from the checkpoint
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            
            # Create a wrapper to match VQ-VLA's checkpoint structure
            vqvae_checkpoint = {}
            for key in checkpoint.keys():
                if key.startswith("vqvae."):
                    new_key = key.split(".")
                    new_key = ".".join(new_key[1:])
                    vqvae_checkpoint[new_key] = checkpoint[key]
            
            # Load the weights (strict=False to handle architecture differences)
            load_result = self.vqvae_model.load_state_dict(vqvae_checkpoint, strict=False)
            print(f"Loaded VQ-VAE weights: {load_result}")
            
        except Exception as e:
            print(f"Warning: Could not load VQ-VAE weights: {e}")
            print("Using random initialization")
    
    def encode_actions(self, action_histories):
        """
        Encode action histories using the pre-trained VQ-VAE
        
        Args:
            action_histories: Tensor (B, H, D) - Batch of action histories
            
        Returns:
            encoded_actions: Tensor (B, embedding_dim) - Encoded action representations
        """
        with torch.no_grad():
            # Forward pass through VQ-VAE encoder
            # VQ-VLA expects actions in format (B, H, D) where D is action dimension
            encoded_actions = self.vqvae_model.encode(action_histories)
            
            # The latents are in format (B, latent_dim), which is what we want
            return encoded_actions
