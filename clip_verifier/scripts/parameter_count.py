import torch
import torch.nn as nn
from model import TextAwareVisualExtraction, AttentionPooling, ModelConfig

def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def analyze_verifier_parameters(history_length=20, action_dim=7, use_transformer=False):
    """Analyze parameter counts for the VLA_BERT_DINO verifier"""
    
    # Model configuration
    model_config = ModelConfig(
        text_pooling_output_dim=768,
        vision_pooling_output_dim=768,
        pooling_heads=8,
        pooling_layers=4,
        num_readouts=4,
        action_dim=action_dim,
        history_length=history_length,
    )
    
    print("=== VLA_BERT_DINO Verifier Parameter Analysis ===")
    print(f"Configuration:")
    print(f"  - History length: {history_length}")
    print(f"  - Action dimension: {action_dim}")
    print(f"  - Use transformer: {use_transformer}")
    print(f"  - Text pooling output dim: {model_config.text_pooling_output_dim}")
    print(f"  - Vision pooling output dim: {model_config.vision_pooling_output_dim}")
    print(f"  - Pooling heads: {model_config.pooling_heads}")
    print(f"  - Pooling layers: {model_config.pooling_layers}")
    print(f"  - Num readouts: {model_config.num_readouts}")
    print()
    
    # Create a mock model to analyze components
    class MockBERT(nn.Module):
        def __init__(self):
            super().__init__()
            # BERT base has ~110M parameters
            self.embedding = nn.Embedding(30522, 768)  # vocab_size, hidden_size
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1),
                num_layers=12
            )
            self.pooler = nn.Linear(768, 768)
    
    class MockDINO(nn.Module):
        def __init__(self):
            super().__init__()
            # DINO v2 ViT-B/14 has ~86M parameters
            self.patch_embed = nn.Linear(3 * 14 * 14, 768)  # patch_size=14
            self.pos_embed = nn.Parameter(torch.randn(1, 256, 768))  # 256 patches for 224x224 image
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1)
                for _ in range(12)
            ])
    
    # Create components
    bert = MockBERT()
    dino = MockDINO()
    
    # Text-aware visual extraction
    num_img_patches = 256  # For 224x224 image with 14x14 patches
    text_aware_extraction = TextAwareVisualExtraction(
        num_img_patches=num_img_patches,
        vision_dim=768
    )
    
    # Pooling components
    text_pooling = AttentionPooling(
        input_dim=768,
        output_dim=model_config.text_pooling_output_dim,
        num_heads=model_config.pooling_heads,
        num_layers=model_config.pooling_layers,
        num_readouts=model_config.num_readouts
    )
    
    agentview_pooling = AttentionPooling(
        input_dim=768,
        output_dim=model_config.vision_pooling_output_dim,
        num_heads=model_config.pooling_heads,
        num_layers=model_config.pooling_layers,
        num_readouts=model_config.num_readouts
    )
    
    handview_pooling = AttentionPooling(
        input_dim=768,
        output_dim=model_config.vision_pooling_output_dim,
        num_heads=model_config.pooling_heads,
        num_layers=model_config.pooling_layers,
        num_readouts=model_config.num_readouts
    )
    
    # Fusion and projection layers
    f_t_dim = model_config.text_pooling_output_dim + model_config.vision_pooling_output_dim
    input_projection = nn.Linear(f_t_dim, model_config.vision_pooling_output_dim)
    
    image_fusion_proj = nn.Sequential(
        nn.Linear(2 * model_config.vision_pooling_output_dim, 2 * model_config.vision_pooling_output_dim),
        nn.ReLU(),
        nn.Linear(2 * model_config.vision_pooling_output_dim, model_config.vision_pooling_output_dim)
    )
    
    # Action trajectory processing
    if use_transformer:
        single_step_action_encoder = nn.Linear(action_dim, model_config.vision_pooling_output_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config.vision_pooling_output_dim,
            nhead=8,
            dim_feedforward=model_config.vision_pooling_output_dim * 2,
            dropout=0.1
        )
        trajectory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        action_components = [single_step_action_encoder, trajectory_encoder]
    else:
        complex_action_encoder = nn.Sequential(
            nn.Linear(history_length * action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, model_config.vision_pooling_output_dim)
        )
        action_components = [complex_action_encoder]
    
    # Logit scale parameter
    logit_scale = nn.Parameter(torch.tensor(2.6592))
    
    # Calculate parameter counts
    print("Parameter Counts:")
    print("-" * 50)
    
    # BERT (frozen during inference)
    bert_total, bert_trainable = count_parameters(bert)
    print(f"BERT (frozen): {bert_total:,} parameters (0 trainable)")
    
    # DINO (frozen during inference)
    dino_total, dino_trainable = count_parameters(dino)
    print(f"DINO (frozen): {dino_total:,} parameters (0 trainable)")
    
    # Text-aware visual extraction
    text_aware_total, text_aware_trainable = count_parameters(text_aware_extraction)
    print(f"Text-aware visual extraction: {text_aware_total:,} parameters ({text_aware_trainable:,} trainable)")
    
    # Pooling components
    text_pooling_total, text_pooling_trainable = count_parameters(text_pooling)
    agentview_pooling_total, agentview_pooling_trainable = count_parameters(agentview_pooling)
    handview_pooling_total, handview_pooling_trainable = count_parameters(handview_pooling)
    
    print(f"Text pooling: {text_pooling_total:,} parameters ({text_pooling_trainable:,} trainable)")
    print(f"Agentview pooling: {agentview_pooling_total:,} parameters ({agentview_pooling_trainable:,} trainable)")
    print(f"Handview pooling: {handview_pooling_total:,} parameters ({handview_pooling_trainable:,} trainable)")
    
    # Fusion and projection
    input_proj_total, input_proj_trainable = count_parameters(input_projection)
    fusion_total, fusion_trainable = count_parameters(image_fusion_proj)
    
    print(f"Input projection: {input_proj_total:,} parameters ({input_proj_trainable:,} trainable)")
    print(f"Image fusion projection: {fusion_total:,} parameters ({fusion_trainable:,} trainable)")
    
    # Action trajectory processing
    action_total = 0
    action_trainable = 0
    for component in action_components:
        total, trainable = count_parameters(component)
        action_total += total
        action_trainable += trainable
    
    action_type = "Transformer" if use_transformer else "MLP"
    print(f"Action trajectory ({action_type}): {action_total:,} parameters ({action_trainable:,} trainable)")
    
    # Logit scale
    logit_scale_total = logit_scale.numel()
    logit_scale_trainable = logit_scale.numel() if logit_scale.requires_grad else 0
    print(f"Logit scale: {logit_scale_total:,} parameters ({logit_scale_trainable:,} trainable)")
    
    # Calculate totals
    frozen_params = bert_total + dino_total
    trainable_params = (text_aware_trainable + text_pooling_trainable + agentview_pooling_trainable + 
                       handview_pooling_trainable + input_proj_trainable + fusion_trainable + 
                       action_trainable + logit_scale_trainable)
    total_params = frozen_params + trainable_params
    
    print("-" * 50)
    print(f"FROZEN PARAMETERS (BERT + DINO): {frozen_params:,}")
    print(f"TRAINABLE PARAMETERS: {trainable_params:,}")
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print()
    
    # Memory usage estimation (assuming float32)
    frozen_memory_mb = frozen_params * 4 / (1024 * 1024)  # 4 bytes per parameter
    trainable_memory_mb = trainable_params * 4 / (1024 * 1024)
    total_memory_mb = total_params * 4 / (1024 * 1024)
    
    print("Memory Usage (float32):")
    print(f"Frozen parameters: {frozen_memory_mb:.2f} MB")
    print(f"Trainable parameters: {trainable_memory_mb:.2f} MB")
    print(f"Total memory: {total_memory_mb:.2f} MB")
    
    return {
        'frozen_params': frozen_params,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'frozen_memory_mb': frozen_memory_mb,
        'trainable_memory_mb': trainable_memory_mb,
        'total_memory_mb': total_memory_mb
    }

if __name__ == "__main__":
    print("=== MLP Version ===")
    mlp_results = analyze_verifier_parameters(history_length=10, action_dim=7, use_transformer=False)
    
    print("\n" + "="*60 + "\n")
    
    print("=== Transformer Version ===")
    transformer_results = analyze_verifier_parameters(history_length=10, action_dim=7, use_transformer=True) 