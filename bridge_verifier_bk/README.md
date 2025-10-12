# Bridge Verifier - VLA-CLIP
## Overview

The Bridge Verifier implements a contrastive learning approach that aligns visual observations with action histories. The system leverages the Bridge V2 robotics dataset and trains models using distributed data parallel (DDP) for efficient multi-GPU training.

## Project Structure

```
bridge_verifier/
├── augment_bridge_dataset.py          # Dataset preprocessing and augmentation
├── finetune_trajectory_bridge_ddp.py  # Main distributed training script 
├── instruction_mapping.json           # Instruction rephrasing mappings 
└── README.md                          # This documentation file
```

## Key Components

### 1. Dataset Augmentation (`augment_bridge_dataset.py`)
- **Purpose**: Preprocesses Bridge V2 dataset with action histories and instruction rephrasing
- **Features**:
  - Extracts action trajectories with configurable history length
  - Normalizes and deduplicates instructions
  - Saves agent view images as JPG files
  - Memory-efficient processing with action history reuse
  - Supports instruction rephrasing for data augmentation
  - Handles both normalized and legacy data formats

### 2. Distributed Training (`finetune_trajectory_bridge_ddp.py`)
- **Purpose**: Main training script using PyTorch Distributed Data Parallel
- **Architecture**: VLA-CLIP model with contrastive learning
- **Key Classes**:
  - `BridgeDataset`: Custom dataset class with DDP sharding support
  - `VLA_CLIP_Bridge`: Main model combining CLIP visual/text encoders with action encoders
- **Features**:
  - Multi-GPU distributed training with NCCL backend
  - Configurable transformer or MLP encoders for action histories
  - Contrastive loss between visual observations and action sequences
  - Top-k accuracy metrics for both image-to-action and action-to-image retrieval
  - Wandb integration for experiment tracking
  - Gradient clipping and linear warmup scheduling
  - Memory-efficient loading with DDP sharding
  - Comprehensive GPU memory and training metrics

### 3. Instruction Mapping (`instruction_mapping.json`)
- **Purpose**: Contains rephrased versions of original Bridge V2 instructions
- **Usage**: Enables data augmentation through instruction diversity
- **Content**: Mapping from original instructions to multiple rephrasings

## Complete Setup Instructions

### Prerequisites
```bash
# Disable auto tmux and install system dependencies
touch ~/.no_auto_tmux
sudo apt update && sudo apt install unzip git-lfs -y

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init
source ~/.bashrc

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install
rm awscliv2.zip && rm -rf aws
```

### Environment Setup
```bash
# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate environment
conda create -n vla-clip python=3.10 -y
conda activate vla-clip

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install core dependencies
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
pip install numpy bitsandbytes wandb openai tqdm ijson
pip install timm==0.9.10 tokenizers==0.19.1
pip install torch>=2.2.0 torchvision>=0.16.0
pip install transformers==4.40.1 h5py
pip install git+https://github.com/openai/CLIP.git
```

### Configure AWS and Git
```bash
# Configure AWS credentials (use your own credentials)
aws configure set aws_access_key_id YOUR_ACCESS_KEY
aws configure set aws_secret_access_key YOUR_SECRET_KEY
aws configure set default.region us-east-1

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@domain.com"

# Install Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

### Dataset Preparation

1. **Clone repository and download pre-processed data**:
```bash
git clone https://github.com/jackyk02/vla-clip
cd vla-clip

# Download pre-augmented dataset (recommended)
aws s3 cp s3://bridge-data-bucket/vla-clip/bridge_dataset_with_rephrases.json ./
aws s3 cp s3://bridge-data-bucket/vla-clip/1.3m_imgs.zip ./
unzip 1.3m_imgs.zip
```

2. **Optional: Download raw Bridge V2 dataset for custom preprocessing**:
```bash
aws s3 cp s3://bridge-data-bucket/bridge-data /root/tensorflow_datasets/bridge_dataset/1.0.0 --recursive
```

3. **Optional: Generate custom augmented dataset**:
```bash
cd bridge_verifier
python augment_bridge_dataset.py --max_episodes 10
```

### Authentication Setup
```bash
# Login to Weights & Biases for experiment tracking
wandb login
```

## Usage

### Training

#### Quick Start (Recommended)
```bash
cd vla-clip/bridge_verifier
python finetune_trajectory_bridge_ddp.py \
    --epochs 20 \
    --batch_size 16384 \
    --lr 8e-4 \
    --history_length 10 \
    --augmented_dataset bridge_dataset_with_rephrases.json \
    --images_folder bridge_dataset_with_rephrases_images \
    --save_name bridge_rephrases \
    --use_transformer \
    --use_wandb \
    --world_size 4 \
    --warmup_epochs 1 \
    --train_log_freq 10 \
    --eval_log_freq 100 \
    --validation_split 0.1
```

#### Data Generation Example
```bash
cd bridge_verifier
python augment_bridge_dataset.py \
    --builder_dir /root/bridge_dataset/1.0.0 \
    --output_path custom_bridge_dataset.json \
    --history_length 10 \
    --rephrases_json instruction_mapping.json \
    --max_episodes 1000 \
    --images_folder custom_bridge_images
```

#### Inference Example (TODO)
```bash
cd /root/vla-clip/bridge_verifier

# Download the model checkpoint
aws s3 cp s3://bridge-data-bucket/rephrase/bridge_rephrases_epoch_20.pt ./bridge_rephrases_epoch_20.pt

python vla_clip_inference_bridge.py \
  --model_path bridge_rephrases_epoch_20.pt \
  --bridge_dataset 10episodes.json \
  --images_folder 10episodes_imgs \
  --history_length 10 \
  --use_transformer \
  --num_samples 50 \
  --action_pool_size 20
```

### Training Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 50 | Number of training epochs |
| `--batch_size` | int | 64 | Batch size per GPU |
| `--lr` | float | 1e-6 | Learning rate |
| `--validation_split` | float | 0.1 | Fraction for validation |
| `--warmup_epochs` | int | 10 | Linear warmup period |

### Model Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--history_length` | int | **Required** | Action history length (must match dataset) |
| `--action_dim` | int | None | Action dimension (auto-inferred if None) |
| `--use_transformer` | flag | False | Use transformer instead of MLP for actions |

### Dataset and Paths
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--augmented_dataset` | str | **Required** | Path to augmented Bridge dataset JSON |
| `--images_folder` | str | **Required** | Path to agent view images directory |
| `--checkpoint_dir` | str | bridge_trajectory_checkpoints_ddp | Checkpoint save directory |
| `--save_name` | str | None | Model name (auto-generated if None) |
| `--resume` | str | None | Checkpoint path to resume from |

### Distributed Training
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--world_size` | int | All GPUs | Number of GPUs to use |
| `--port` | int | 12355 | Communication port for DDP |

### Logging and Monitoring
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use_wandb` | flag | False | Enable Weights & Biases tracking |
| `--train_log_freq` | int | 50 | Training log frequency (steps) |
| `--eval_log_freq` | int | 500 | Evaluation frequency (steps) |

## Dataset Format

The augmented dataset follows this optimized structure:

```json
{
  "action_histories": [
    {
      "id": "action_hist_0",
      "actions": [[x1, y1, z1, rx1, ry1, rz1, g1], ...]
    }
  ],
  "instructions": [
    {
      "id": "inst_0",
      "text": "pick up the red block"
    }
  ],
  "samples": [
    {
      "instruction_id": "inst_0",
      "action_history_id": "action_hist_0",
      "image_path": "agent_view_episode_1_step_10.jpg"
    }
  ]
}
```

## Performance Metrics

The training system tracks comprehensive metrics:

### Training Metrics
- **Contrastive Loss**: Main training objective using InfoNCE loss
- **Learning Rate**: Current learning rate with warmup scheduling
- **Gradient Norm**: L2 norm of gradients for stability monitoring
- **Training Throughput**: Samples/second across all GPUs

### Evaluation Metrics
- **Image-to-Action Top-k Accuracy**: Visual query → action retrieval (k=1,5,10)
- **Action-to-Image Top-k Accuracy**: Action query → visual retrieval (k=1,5,10)
- **Validation Loss**: Contrastive loss on held-out data