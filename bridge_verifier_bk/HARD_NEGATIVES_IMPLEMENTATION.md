# Hard Negatives Implementation for VLA-CLIP Bridge Training

## Overview

This implementation incorporates hard negative samples into the VLA-CLIP Bridge training pipeline using weighted InfoNCE loss. The system supports both positive instructions (original + up to 39 rephrases) and hard negatives for improved contrastive learning.

## Key Components

### 1. Data Processing (`augment_bridge_dataset_with_hard_negatives.py`)

**New Features:**
- Replaces `rephrases_json` with `curated_hard_negatives.json` as input
- Skips first 2 timesteps per episode (non-meaningful actions)
- Stops processing when reaching `max_sample_id` from hard negatives data
- Creates normalized dataset format version `3.0_with_hard_negatives`

**Data Structure:**
```json
{
  "action_histories": {...},
  "instructions": {...},
  "samples": [
    {
      "sample_id": 0,
      "action_history_id": "action_123",
      "agent_view_image_file": "456.jpg",
      "positives": ["instr_1", "instr_2", ...],
      "hard_negatives": [
        {
          "instruction_id": "instr_5",
          "positive_instruction_id": "instr_1",
          "similarity": 0.95,
          "error": 0.23
        }
      ],
      "episode_id": 10,
      "timestep": 3
    }
  ]
}
```

**Usage:**
```bash
python augment_bridge_dataset_with_hard_negatives.py \
  --builder_dir /path/to/bridge_dataset \
  --output_path bridge_with_hard_negatives.json \
  --history_length 10 \
  --hard_negatives_json curated_hard_negatives.json \
  --images_folder bridge_images
```

### 2. Training (`finetune_trajectory_bridge_ddp_with_hard_negatives.py`)

**New Features:**
- `BridgeDatasetWithHardNegatives` class for loading hard negatives data
- `VLA_CLIP_Bridge_HardNegatives` model (same architecture, different dataset handling)
- `weighted_infonce_loss_with_hard_negatives()` function
- Custom `collate_hard_negatives_batch()` function
- Support for variable numbers of positives/negatives per sample

**Weighted InfoNCE Loss:**
- Combines in-batch negatives with hard negatives in denominator
- Uses similarity-based weighting for hard negatives
- Numerically stable implementation with log-sum-exp trick
- Configurable alpha parameter for hard negative weighting

**Loss Formula:**
```
L_i = log(∑_j exp(s_ij)) - s_ii
```
where `s_ij` includes:
- Positive logit: `s_ii = scale * (action_i · (image_i + text_i))`
- In-batch negatives: `s_ij = scale * (action_i · (image_j + text_j))` for j≠i
- Hard negatives: `s_ik = scale * (action_i · (image_i + hard_neg_k)) + log(α * weight_k)`

**Usage:**
```bash
python finetune_trajectory_bridge_ddp_with_hard_negatives.py \
  --augmented_dataset bridge_with_hard_negatives.json \
  --images_folder bridge_images \
  --history_length 10 \
  --batch_size 32 \
  --lr 1e-6 \
  --epochs 50 \
  --hard_negative_alpha 1.0 \
  --use_wandb
```

## Compatibility

### Dataset Formats Supported:
1. **3.0_with_hard_negatives**: New format with positives and hard negatives
2. **2.0_normalized**: Previous normalized format (fallback)
3. **1.0_legacy**: Original legacy format (fallback)

### Scale Compatibility:
- **Small datasets**: Works with 2 samples (current `curated_hard_negatives.json`)
- **Full dataset**: Designed to handle 1,383,034 samples efficiently
- **Memory optimization**: Deduplicates action histories and instructions

## Key Implementation Details

### 1. Action History Skipping
- Skips first 2 timesteps per episode (`t=0, t=1`)
- Reasoning: Early actions are typically non-meaningful (robot initialization)

### 2. Sample ID Management
- Global `sample_id_counter` ensures consistent indexing
- Stops processing when reaching `max_sample_id` from hard negatives data
- Compatible with both small test sets and full datasets

### 3. Hard Negatives Processing
- Variable number of hard negatives per positive instruction
- Similarity-based weighting using provided similarity scores
- Handles empty hard negatives gracefully

### 4. Batch Processing
- Custom collate function handles variable-length sequences
- Maps positives to original samples for image/action reuse
- Efficient GPU memory usage with proper batching

### 5. Loss Computation
- Numerically stable with log-sum-exp trick
- Configurable alpha parameter for hard negative importance
- Proper gradient flow through all components

## Performance Considerations

### Memory Efficiency:
- Deduplicates action histories (compression ratio: ~2-3x)
- Lazy loading of data in dataset class
- DDP sharding across multiple GPUs

### Training Stability:
- Gradient clipping (max_norm=1.0)
- Linear warmup scheduler
- Proper initialization and numerical stability

### Monitoring:
- Hard negative accuracy tracking
- Positive/negative logit statistics
- GPU memory usage monitoring
- Comprehensive wandb logging

## Usage Examples

### Small Dataset Testing:
```bash
# Generate dataset with existing hard negatives (2 samples)
python augment_bridge_dataset_with_hard_negatives.py \
  --max_episodes 1 \
  --hard_negatives_json curated_hard_negatives.json

# Train with small dataset
python finetune_trajectory_bridge_ddp_with_hard_negatives.py \
  --batch_size 2 \
  --epochs 5 \
  --world_size 1
```

### Full Dataset Training:
```bash
# Generate full dataset (will process until max_sample_id from hard negatives)
python augment_bridge_dataset_with_hard_negatives.py \
  --hard_negatives_json curated_hard_negatives_full.json

# Multi-GPU training
python finetune_trajectory_bridge_ddp_with_hard_negatives.py \
  --batch_size 64 \
  --epochs 100 \
  --world_size 8 \
  --hard_negative_alpha 2.0
```

## Future Extensions

1. **Dynamic Hard Negative Mining**: Update hard negatives during training
2. **Curriculum Learning**: Gradually increase hard negative difficulty
3. **Multi-modal Hard Negatives**: Include visual hard negatives
4. **Adaptive Alpha**: Learn the hard negative weighting parameter

## Notes

- The implementation maintains backward compatibility with existing datasets
- All hard negatives data is expected to be pre-computed and provided in the JSON file
- The system gracefully handles samples without hard negatives
- DDP training is fully supported with proper synchronization
