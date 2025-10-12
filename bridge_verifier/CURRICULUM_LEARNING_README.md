# Curriculum Learning for VLA-CLIP Bridge Training

## Overview

This curriculum learning approach addresses the distribution shift problem between collected Bridge dataset and policy rollout data by gradually introducing policy-in-the-loop data while maintaining Bridge dataset dominance.

## Key Features

### 1. **Bridge Dataset Dominance**
- Minimum 80% Bridge data throughout training
- Ensures unbiased learning from expert demonstrations
- Maintains high-quality positive examples

### 2. **Gradual Policy Introduction**
- **Epochs 1-10**: 100% Bridge data (pure expert foundation)
- **Epochs 11-25**: 100% → 70% Bridge, 0% → 30% Policy (gradual introduction)
- **Epochs 26-50**: 70% → 50% Bridge, 30% → 50% Policy (maintain dominance)

### 3. **Semantically Meaningful Hard Negative Filtering**
- **Same Task, Different Execution**: High semantic similarity (>0.8) + high action error (>0.3)
  - Example: "Pick up the red cup" (successful) vs "Pick up the red cup" (failed attempt)
- **Different Tasks**: Low semantic similarity (<0.4) + different episodes
  - Example: "Pick up the red cup" vs "Put the blue bowl on the table"
- **Avoids**: Bad grammar instructions or completely unrelated actions

## Usage

### Basic Training
```bash
./run_curriculum_training.sh
```

### Custom Configuration
```bash
python finetune_trajectory_bridge_curriculum_ddp.py \
    --bridge_dataset /path/to/bridge_dataset.json \
    --policy_dataset /path/to/policy_dataset.json \
    --images_folder /path/to/images \
    --history_length 10 \
    --epochs 50 \
    --bridge_dominance_ratio 0.8 \
    --max_policy_ratio 0.5 \
    --world_size 2
```

## Dataset Requirements

### Bridge Dataset (Primary)
- Pure Bridge V2 dataset with expert demonstrations
- Contains original action sequences and instructions
- No policy-generated data (unbiased)

### Policy Dataset (Secondary, Optional)
- Policy-in-the-loop generated data
- May contain biased hard negatives
- Used for distribution matching

## Curriculum Schedule Details

```python
def get_curriculum_schedule(epoch, total_epochs):
    progress = epoch / total_epochs
    
    if progress < 0.2:      # First 20% of training
        return {'bridge': 1.0, 'policy': 0.0}
    elif progress < 0.5:    # Next 30% of training  
        bridge_ratio = 1.0 - 0.3 * ((progress - 0.2) / 0.3)
        return {'bridge': bridge_ratio, 'policy': 1.0 - bridge_ratio}
    else:                   # Remaining 50% of training
        bridge_ratio = max(0.5, 0.7 - 0.2 * ((progress - 0.5) / 0.5))
        return {'bridge': bridge_ratio, 'policy': 1.0 - bridge_ratio}
```

## Hard Negative Filtering Strategy

We select semantically meaningful hard negatives that represent actual confusions:

```python
def filter_hard_negatives(hard_negatives, current_sample):
    filtered = []
    
    for neg in hard_negatives:
        similarity = neg['similarity']
        error = neg['error']
        
        if similarity > 0.8 and error > 0.3:
            # Same task, different execution (good instruction + bad actions)
            filtered.append({
                'instruction': neg['instruction'],
                'confidence': similarity * error,
                'type': 'same_task_different_execution'
            })
        elif similarity < 0.4 and neg['episode_id'] != current_sample['episode_id']:
            # Different task entirely
            filtered.append({
                'instruction': neg['instruction'], 
                'confidence': 1.0 - similarity,
                'type': 'different_task'
            })
    
    # Take top 3 most meaningful negatives
    return sorted(filtered, key=lambda x: x['confidence'], reverse=True)[:3]
```

### Examples of Good Hard Negatives:
- ✅ "Pick up the red cup" (successful) vs "Pick up the red cup" (failed)
- ✅ "Move to the table" vs "Put the bowl on the table" (similar but different tasks)

### Examples of Bad Hard Negatives:
- ❌ "Pick up the red cup" vs "Drive the car" (completely unrelated)
- ❌ "Pick up red cup" vs "Pick up the blue cup" (successful alternatives)

## Benefits

1. **Unbiased Learning**: Starts with pure expert demonstrations
2. **Distribution Matching**: Gradually adapts to policy rollout distribution
3. **Conservative Negatives**: Avoids using successful alternatives as negatives
4. **Bridge Dominance**: Maintains expert data majority throughout training
5. **Flexible**: Can run Bridge-only by omitting policy dataset

## Monitoring

The training script logs:
- Current Bridge/Policy data ratios
- Training and validation losses
- Curriculum schedule progress
- Model performance metrics

## Example Output

```
Epoch 5: Bridge=100%, Policy=0%   Train Loss: 2.45, Val Loss: 2.38
Epoch 15: Bridge=85%, Policy=15%  Train Loss: 1.89, Val Loss: 1.92
Epoch 30: Bridge=65%, Policy=35%  Train Loss: 1.45, Val Loss: 1.51
Epoch 50: Bridge=50%, Policy=50%  Train Loss: 1.23, Val Loss: 1.34
```

This approach provides a principled way to balance unbiased learning with distribution matching while avoiding the pitfalls of using successful trajectories as hard negatives.
