#!/bin/bash

# Exit on error
set -e

echo "Starting setup process..."

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 completed successfully"
    else
        echo "✗ Error during $1"
        exit 1
    fi
}

# Reward environment setup
echo "Setting up Reward environment..."
source $HOME/miniconda3/etc/profile.d/conda.sh

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

if ! conda env list | grep -qE "^\s*monkey-verifier\s"; then
    $HOME/miniconda3/bin/conda create -n monkey-verifier python=3.10 -y
else
    echo "Conda environment 'monkey-verifier' already exists. Skipping creation."
fi

conda activate monkey-verifier
cd "$dir_path/../monkey-verifier/llava_setup/LLaVA"

pip install --upgrade pip
pip install -e .
pip install ninja

# LLaVA-RLHF dependencies
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed==0.9.3 peft==0.4.0 transformers==4.31.0 bitsandbytes==0.41.0
pip install datasets==3.6.0 wandb==0.20.1 numpy==1.26.4 json_numpy openpyxl==3.1.5 tensorflow==2.19.0
check_status "LLaVA-RLHF dependencies"

# Setup directories and download model
cd "$dir_path/../"
git clone https://huggingface.co/robomonkey-vla/monkey-verifier-7b model_dir