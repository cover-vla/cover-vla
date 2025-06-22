import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from timm.layers.mlp import Mlp

class CrossAttentionBlock(nn.Module):
    def __init__(self, kv_input_dim : int, q_dim : int, mlp_dim : int, num_heads : int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            q_dim, 
            num_heads, 
            batch_first=True, 
            kdim=kv_input_dim, 
            vdim=kv_input_dim,
        )
        self.mlp = Mlp(
            in_features=q_dim, 
            hidden_features=mlp_dim, 
            out_features=q_dim,
        )
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.layer_norm = nn.LayerNorm(q_dim)
    
    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
        the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.

        mask will be true at start of text, end of text, and padding tokens
        """
        # mask: [batch, seq_len], True for tokens to attend to
        q = self.q_layer_norm(q)
        attn_out, _ = self.attention(q, kv, kv, key_padding_mask=mask)
        q = q + attn_out  # Add residual connection
        q = self.layer_norm(q)
        x = self.mlp(q)
        return q + x

def sincos_position_embedding(seq_len: int, dim: int) -> torch.Tensor:
    """Generate sinusoidal positional embedding"""
    # shape: [seq_len, dim]
    pos = torch.arange(seq_len).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
    emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
    return emb


class TextAwareVisualExtraction(nn.Module):
    """Extract text-aware visual features using CLIP, following ClearCLIP approach"""
    def __init__(self, num_img_patches : int, vision_dim : int, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        # mark position embedding as non-trainable param 
        self.register_buffer('pos_emb', sincos_position_embedding(num_img_patches, vision_dim))
        
    def forward(self, image_patch_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # Calculate similarity between text and patch features
        # image_patch_features: (batch_size, num_patches, embedding_dim)
        # text_features: (batch_size, num_tokens, embedding_dim)
        
        similarity = torch.einsum('bij,bkj->bik', text_features, image_patch_features)
        
        # Apply temperature scaling and softmax
        attention = F.softmax(similarity / self.temperature.clamp(0, 100), dim=-1)
        
        # add position embedding to image patch features 
        pe_image_patch_features = image_patch_features + self.pos_emb
        # Get text-aware visual features by combining patch features according to attention (with position embedding)
        text_aware_features = torch.einsum('bik,bkj->bij', attention, pe_image_patch_features)
        
        return text_aware_features
    
    
class AttentionPooling(nn.Module):
    """Attention pooling layer to compress sequence of tokens into a single token"""
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, num_layers: int = 2, num_readouts: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_readouts = num_readouts
        self.intermediate_dim = self.output_dim // self.num_readouts
        assert self.intermediate_dim * self.num_readouts == self.output_dim, "Output dim must be divisible by num readouts"
        self.query = nn.Parameter(torch.randn(1, num_readouts, self.intermediate_dim))
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                kv_input_dim=input_dim, 
                q_dim=self.intermediate_dim, 
                mlp_dim=self.output_dim, 
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
        the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.

        mask will be true at start of text, end of text, and padding tokens
        """
        # mask: [batch, seq_len], True for tokens to attend to
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
            
        for layer in self.blocks:
            query = layer(query, x, mask)
        
        query = self.layer_norm(query)
        return query.reshape(batch_size, -1)  # dimension: [batch, output_dim]
    
    
class ModelConfig:
    """Configuration class for VLA_BERT_DINO model parameters"""
    def __init__(
        self,
        clip_model=None,  # Keep for backwards compatibility but not used
        text_pooling_output_dim=512,
        vision_pooling_output_dim=512,
        pooling_heads=8,
        pooling_layers=4,
        num_readouts=1,
        action_dim=7,
        history_length=20,
    ):
        self.clip_model = clip_model
        self.text_pooling_output_dim = text_pooling_output_dim
        self.vision_pooling_output_dim = vision_pooling_output_dim
        self.pooling_heads = pooling_heads
        self.pooling_layers = pooling_layers
        self.num_readouts = num_readouts
        self.action_dim = action_dim
        self.history_length = history_length