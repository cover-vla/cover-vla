from collections import OrderedDict

import torch
import torch.nn as nn
import json
import os

from .modeling_causal_vae import ActionVQVAE, ActionVQVAEPE


class ActionVQVAELossWrapper(nn.Module):
    """
    The causal action vqvae training and inference running wrapper
    """

    def __init__(
        self,
        model_path,
        freeze=False,
        checkpoint_path=None,
        use_action_type_pe=False,
        use_time_pe=False,
        resume=False,
        is_eval=False,
        **kwargs,
    ):
        super().__init__()

        if use_action_type_pe and use_time_pe:
            config_path = f"{model_path}/config_action_type_pe_time_pe.json"
        elif use_action_type_pe:
            config_path = f"{model_path}/config_action_type_pe.json"
        elif use_time_pe:
            config_path = f"{model_path}/config_time_pe.json"
        else:
            config_path = f"{model_path}/config.json"
        
        # Load config manually and create model
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract parameters from config
            encoder_in_channels = config.get('encoder_in_channels', 1)
            encoder_out_channels = config.get('encoder_out_channels', 128)
            encoder_layers_per_block = tuple(config.get('encoder_layers_per_block', [4, 4, 4, 4]))
            encoder_block_out_channels = tuple(config.get('encoder_block_out_channels', [128, 256, 256, 512]))
            encoder_block_dropout = tuple(config.get('encoder_block_dropout', [0.0, 0.0, 0.0, 0.0]))
            encoder_act_fn = config.get('encoder_act_fn', 'silu')
            encoder_norm_num_groups = config.get('encoder_norm_num_groups', 32)
            encoder_double_z = config.get('encoder_double_z', True)
            encoder_type = config.get('encoder_type', 'causal_vae_conv')
            
            decoder_in_channels = config.get('decoder_in_channels', 128)
            decoder_out_channels = config.get('decoder_out_channels', 1)
            decoder_layers_per_block = tuple(config.get('decoder_layers_per_block', [4, 4, 4, 4]))
            decoder_block_out_channels = tuple(config.get('decoder_block_out_channels', [128, 256, 256, 512]))
            decoder_block_dropout = tuple(config.get('decoder_block_dropout', [0.0, 0.0, 0.0, 0.0]))
            decoder_act_fn = config.get('decoder_act_fn', 'silu')
            decoder_norm_num_groups = config.get('decoder_norm_num_groups', 32)
            decoder_type = config.get('decoder_type', 'causal_vae_conv')
            
            vq_embed_dim = config.get('vq_embed_dim', 128)
            num_vq_embeddings = config.get('num_vq_embeddings', 256)
            action_window_size = config.get('action_window_size', 5)
            vq_groups = config.get('vq_groups', 4)
            temporal_compression_ratio = config.get('temporal_compression_ratio', 5)
            
            # Create model with loaded config
            # Use ActionVQVAEPE when both action type and time PE are enabled
            if use_action_type_pe and use_time_pe:
                self.vqvae = ActionVQVAEPE(
                    encoder_in_channels=encoder_in_channels,
                    encoder_out_channels=encoder_out_channels,
                    encoder_layers_per_block=encoder_layers_per_block,
                    encoder_block_out_channels=encoder_block_out_channels,
                    encoder_block_dropout=encoder_block_dropout,
                    encoder_act_fn=encoder_act_fn,
                    encoder_norm_num_groups=encoder_norm_num_groups,
                    encoder_double_z=encoder_double_z,
                    encoder_type=encoder_type,
                    decoder_in_channels=decoder_in_channels,
                    decoder_out_channels=decoder_out_channels,
                    decoder_layers_per_block=decoder_layers_per_block,
                    decoder_block_out_channels=decoder_block_out_channels,
                    decoder_block_dropout=decoder_block_dropout,
                    decoder_act_fn=decoder_act_fn,
                    decoder_norm_num_groups=decoder_norm_num_groups,
                    decoder_type=decoder_type,
                    vq_embed_dim=vq_embed_dim,
                    num_vq_embeddings=num_vq_embeddings,
                    temporal_compression_ratio=temporal_compression_ratio,
                    device="cuda",
                    use_action_type_pe=use_action_type_pe,
                    use_time_pe=use_time_pe
                )
            else:
                self.vqvae = ActionVQVAE(
                    encoder_in_channels=encoder_in_channels,
                    encoder_out_channels=encoder_out_channels,
                    encoder_layers_per_block=encoder_layers_per_block,
                    encoder_block_out_channels=encoder_block_out_channels,
                    encoder_block_dropout=encoder_block_dropout,
                    encoder_act_fn=encoder_act_fn,
                    encoder_norm_num_groups=encoder_norm_num_groups,
                    encoder_double_z=encoder_double_z,
                    encoder_type=encoder_type,
                    decoder_in_channels=decoder_in_channels,
                    decoder_out_channels=decoder_out_channels,
                    decoder_layers_per_block=decoder_layers_per_block,
                    decoder_block_out_channels=decoder_block_out_channels,
                    decoder_block_dropout=decoder_block_dropout,
                    decoder_act_fn=decoder_act_fn,
                    decoder_norm_num_groups=decoder_norm_num_groups,
                    decoder_type=decoder_type,
                    vq_embed_dim=vq_embed_dim,
                    num_vq_embeddings=num_vq_embeddings,
                    action_window_size=action_window_size,
                    vq_groups=vq_groups,
                    device="cuda"
                )
        else:
            # Fallback to default parameters if config not found
            print(f"Warning: Config file {config_path} not found, using default parameters")
            self.vqvae = ActionVQVAE()
        self.token_num = self.vqvae.vqvae_groups

        if resume:
            assert checkpoint_path is not None, "resume mode must provide checkpoint path"
            self.load_checkpoint(checkpoint_path)
        elif checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        if is_eval:
            self.vqvae.encoder.eval()
            self.vqvae.decoder.eval()
            self.vqvae.vq_layer.eval()

        if freeze:
            for parameter in self.vqvae.encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.vqvae.decoder.parameters():
                parameter.requires_grad = False
            for parameter in self.vqvae.vq_layer.parameters():
                parameter.requires_grad = False

        self.loss = None

    def load_checkpoint(self, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        vqvae_checkpoint = OrderedDict()

        for key in checkpoint.keys():
            if key.startswith("vqvae."):
                new_key = key.split(".")
                new_key = ".".join(new_key[1:])
                vqvae_checkpoint[new_key] = checkpoint[key]

        vqvae_ckpt_load_result = self.vqvae.load_state_dict(vqvae_checkpoint, strict=True)
        print(f"Load vae checkpoint from {checkpoint_path}, load result: {vqvae_ckpt_load_result}")

    def forward(self, act, robot_type=None, frequency=None):
        commit_loss, recon_loss, loss = self.vqvae(act, robot_type=robot_type, frequency=frequency, return_dict=False)
        return commit_loss, recon_loss, loss

    def get_code(self, x):
        with torch.no_grad():
            latents = self.vqvae.encode(x).latents
            B = latents.shape[0]  # B
            state_rep_flat = latents.view(latents.size(0), -1, latents.size(1))
            state_rep_flat, vq_code, vq_loss_state = self.vqvae.vq_layer(state_rep_flat)
            vq_code = vq_code.view(B, -1)  # b,2
        return vq_code

    def get_embeddings(self, x):
        """
        Get the actual VQ-VAE embeddings (latents) instead of discrete indices
        This is what we need for the VLA-CLIP integration
        """
        with torch.no_grad():
            latents = self.vqvae.encode(x).latents
            # latents shape: (B, 128) - this is what we want!
            return latents

    def draw_code_forward(self, encoding_indices):
        with torch.no_grad():
            z_embed = self.vqvae.vq_layer.get_output_from_indices(encoding_indices)
        return z_embed

    def get_action_from_latent(self, latent, robot_type=None, control_frequency=None):
        dec = self.vqvae.decode(latent, robot_type, control_frequency)
        return dec

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
