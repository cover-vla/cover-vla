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

# Create and activate environment
if ! conda env list | grep -qE "^\s*simpler_env\s"; then
    echo "Creating simpler_env environment..."
    $HOME/miniconda3/bin/conda create -n simpler_env python=3.10 -y
else
    echo "Conda environment 'simpler_env' already exists. Skipping creation."
fi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env

cd "$dir_path/../SimplerEnv"
sudo apt-get install -y libvulkan1 cudnn9-cuda-12 libx11-6
pip install --upgrade pip
pip install numpy==1.24.4
pip install -r requirements_full_install.txt --use-deprecated=legacy-resolver
pip install tensorflow-probability==0.24.0
check_status "SimplerEnv base setup"

# Additional SimplerEnv dependencies
pip install -e ./ManiSkill2_real2sim/
pip install -e .
pip install pandas==2.3.0 openpyxl==3.1.5 matplotlib==3.10.3 mediapy==1.2.0
pip install tensorflow[and-cuda]==2.18.0
pip install "git+https://github.com/nathanrooy/simulated-annealing@293e2b0ad88f81668e98ae104ee204d41b8b34f5"
pip install flax==0.8.1
pip install dlimp "git+https://github.com/kvablack/dlimp@d08da3852c149548aaa8551186d619d87375df08"
pip install distrax==0.1.5 flax==0.8.1 transformers==4.40.1 wandb==0.20.1 mediapy==1.2.0 tf_keras==2.19.0 einops==0.8.1
pip install chex==0.1.2
check_status "SimplerEnv additional dependencies"

# OpenVLA-mini setup
echo "Setting up OpenVLA-mini..."
cd "$dir_path/../openvla-mini"
pip install -e .
pip install -r requirements-min.txt
pip install draccus==0.8.0 rich==14.0.0 accelerate==1.8.0 tensorflow==2.15.0 tensorflow-graphics==2021.12.3 jsonlines==4.0.0 robosuite==1.4.0 bddl==3.5.0 easydict==1.13 gym==0.23.0 PyOpenGL-accelerate==3.1.9 scikit-learn==1.7.0
pip install ml_dtypes==0.5.0
check_status "OpenVLA-mini setup"

# Environment variables
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Additional packages
pip install fastapi uvicorn json_numpy flax==0.8.1
check_status "Additional packages installation"
