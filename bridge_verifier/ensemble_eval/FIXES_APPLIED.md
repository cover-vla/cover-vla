# Fixes Applied to efficient_ensemble_merged.py

This document details all fixes applied to ensure `efficient_ensemble_merged.py` matches the training script (`finetune_trajectory_bridge_ddp.py`) exactly.

## Summary of Issues and Fixes

### Issue 1: TextAwareVisualExtraction Missing Arguments
**Error**: `TypeError: TextAwareVisualExtraction.__init__() missing 1 required positional argument: 'vision_dim'`

**Root Cause**: `TextAwareVisualExtraction` requires `num_img_patches` and `vision_dim` arguments.

**Training Script** (lines 213-216):
```python
self.text_aware_visual_extraction = TextAwareVisualExtraction(
    num_img_patches=self.num_img_patches,
    vision_dim=vision_dim,
)
```

**Fix Applied**:
```python
# Get dimensions from model structure
vision_dim = self.siglip_model.visual.trunk.num_features
visual_patch_size = self.siglip_model.visual.trunk.patch_embed.proj.kernel_size[0]
image_size = self.siglip_model.visual.image_size[0] if hasattr(self.siglip_model.visual, 'image_size') else 224
num_img_patches = (image_size // visual_patch_size) ** 2

# Initialize with correct parameters
text_aware = TextAwareVisualExtraction(
    num_img_patches=num_img_patches,
    vision_dim=vision_dim
)
```

---

### Issue 2: Vision/Text Pooling Using Wrong Module
**Error**: State dict mismatch - expected `AttentionPooling` keys but got `Sequential` structure

**Root Cause**: Used `nn.Sequential` instead of `AttentionPooling`

**Training Script** (lines 218-231):
```python
self.text_pooling = AttentionPooling(
    text_dim, 
    text_pooling_output_dim,  # 512
    pooling_heads,             # 8
    pooling_layers,            # 4
    num_readouts=self.num_readouts,  # 1
)        
self.vision_poolings = AttentionPooling(
    vision_dim,
    vision_pooling_output_dim,  # 512
    pooling_heads,               # 8
    pooling_layers,              # 4
    num_readouts=self.num_readouts  # 1
)
```

**Fix Applied**:
```python
vision_pooling = AttentionPooling(
    input_dim=vision_dim,
    output_dim=512,
    num_heads=8,
    num_layers=4,
    num_readouts=1
)

text_pooling = AttentionPooling(
    input_dim=text_dim,
    output_dim=512,
    num_heads=8,
    num_layers=4,
    num_readouts=1
)
```

---

### Issue 3: TransformerEncoder Wrong Feedforward Dimension
**Error**: Size mismatch - expected `dim_feedforward=1024` but got default `2048`

**Root Cause**: PyTorch's default `dim_feedforward` is `2048`, but training uses `512 * 2 = 1024`

**Training Script** (lines 246-253):
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=vision_pooling_output_dim,  # 512
    nhead=8,
    dim_feedforward=vision_pooling_output_dim * 2,  # 1024
    dropout=0.1
)
self.trajectory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
```

**Fix Applied**:
```python
encoder_layer = torch.nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=1024,  # 512 * 2, matching training
    batch_first=False,
    dropout=0.1
)
trajectory_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
```

---

### Issue 4: MLP Encoder Wrong Architecture
**Error**: Missing `LayerNorm` and `Dropout` layers

**Root Cause**: Used simple MLP instead of the training architecture

**Training Script** (lines 256-262):
```python
self.complex_action_encoder = nn.Sequential(
    nn.Linear(self.history_length * self.action_dim, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, vision_pooling_output_dim)  # 512
)
```

**Fix Applied**:
```python
complex_encoder = torch.nn.Sequential(
    torch.nn.Linear(self.history_length * self.action_dim, 512),
    torch.nn.LayerNorm(512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(512, 512)
)
```

---

## Complete Component Configuration

After all fixes, the components match the training script exactly:

### 1. TextAwareVisualExtraction
- `num_img_patches`: Calculated from image_size / patch_size
- `vision_dim`: From `siglip_model.visual.trunk.num_features`

### 2. Vision Pooling (AttentionPooling)
- `input_dim`: `vision_dim`
- `output_dim`: `512`
- `num_heads`: `8`
- `num_layers`: `4`
- `num_readouts`: `1`

### 3. Text Pooling (AttentionPooling)
- `input_dim`: `text_dim`
- `output_dim`: `512`
- `num_heads`: `8`
- `num_layers`: `4`
- `num_readouts`: `1`

### 4. Input Projection
- `Linear(1024, 512)` where 1024 = text_output + vision_output

### 5. Transformer Path (if use_transformer=True)
- **Single Step Encoder**: `Linear(action_dim, 512)`
- **Trajectory Encoder**: `TransformerEncoder`
  - `d_model`: `512`
  - `nhead`: `8`
  - `dim_feedforward`: `1024` (512 * 2)
  - `num_layers`: `4`
  - `batch_first`: `False`
  - `dropout`: `0.1`

### 6. MLP Path (if use_transformer=False)
- **Complex Action Encoder**: `Sequential`
  - `Linear(history_length * action_dim, 512)`
  - `LayerNorm(512)`
  - `ReLU()`
  - `Dropout(0.1)`
  - `Linear(512, 512)`

---

## Verification

To verify all components match:

```bash
cd /root/vla-clip/bridge_verifier
conda activate vla-clip
python3 ensemble_eval/verify_components.py
```

This will:
1. Load the SigLIP model
2. Create test instances of each component
3. Compare state dict keys with training model
4. Report any mismatches

---

## Key Takeaways

1. **Always check training script** for exact module configurations
2. **PyTorch defaults** may not match training (e.g., `dim_feedforward`)
3. **Custom modules** like `AttentionPooling` cannot be replaced with `Sequential`
4. **Dimension calculations** must match training exactly (image size, patches, etc.)

---

**Status**: ✅ All fixes applied, architecture matches training script exactly
**Date**: October 2025


