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

# Initialize conda in the current shell
echo "Initializing conda..."
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

echo "Creating sglang-vla environment..."
# Create and activate environment
if ! conda env list | grep -qE "^\s*sglang-vla\s"; then
    $HOME/miniconda3/bin/conda create -n sglang-vla python=3.9 -y
else
    echo "Conda environment 'sglang-vla' already exists. Skipping creation."
fi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sglang-vla

cd "$dir_path/../sglang-vla"
pip install --upgrade pip
pip install -e "python[all]"
pip install json_numpy
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
pip install timm===0.9.10 openpyxl==3.1.5
pip install fastapi uvicorn
pip install tensorflow==2.19.0
pip install transformers==4.51.3
pip install vllm==0.6.4.post1
check_status "SGLang setup"
