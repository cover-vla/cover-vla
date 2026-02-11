#!/bin/bash

# Exit on error
set -e

echo "Starting setup process with uv..."

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 completed successfully"
    else
        echo "✗ Error during $1"
        exit 1
    fi
}

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

# Set up base paths for easier navigation
VLA_CLIP_ROOT="$(cd "$dir_path/../.." && pwd)"
echo "VLA-CLIP root: $VLA_CLIP_ROOT"

# Install system dependencies first
echo "Installing system dependencies..."
sudo apt-get install -y libvulkan1 libx11-6
check_status "System dependencies installation"

# Create virtual environment
cd "$VLA_CLIP_ROOT"
VENV_PATH="$VLA_CLIP_ROOT/.venv_cover"

echo "Creating virtual environment 'cover'..."
uv venv "$VENV_PATH" --python 3.10
check_status "Virtual environment creation"

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Install setuptools first (needed for pkg_resources which sapien requires)
echo "Installing setuptools..."
uv pip install 'setuptools<70.0.0'
check_status "Setuptools installation"

# Install all dependencies from requirements.txt
echo "Installing all dependencies from requirements.txt..."
echo "This may take a while as uv resolves all dependencies..."
uv pip install -r "$VLA_CLIP_ROOT/requirements.txt"
check_status "Dependencies installation"

# Install SimplerEnv without dependencies (it has dlimp which conflicts with TF 2.18)
echo "Installing SimplerEnv packages without dependencies..."
cd "$VLA_CLIP_ROOT/CoVer_VLA/SimplerEnv"
uv pip install --no-deps -e ./ManiSkill2_real2sim/
uv pip install --no-deps -e .
check_status "SimplerEnv installation"

# Install local editable packages
echo "Installing local packages..."
cd "$VLA_CLIP_ROOT/lerobot_custom"
uv pip install -e ".[pi0]"
check_status "LeRobot installation with PI0 support"

cd "$VLA_CLIP_ROOT/bridge_verifier"
uv pip install -e .
check_status "Bridge Verifier installation"

cd "$VLA_CLIP_ROOT/CoVer_VLA/inference"
uv pip install -e .
check_status "Inference package installation"

# Ensure transformers and torch are at correct versions (may be downgraded by dependencies)
echo "Ensuring correct transformers and PyTorch versions..."
uv pip install 'transformers==4.48.3' 'torch>=2.4.0' torchvision 'numpy>=1.26.4,<2.0' --upgrade
check_status "Final dependency version check"

# Set environment variables
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Set PYTHONPATH to include inference root
INFERENCE_ROOT="$VLA_CLIP_ROOT/CoVer_VLA/inference"
export PYTHONPATH="$INFERENCE_ROOT:$PYTHONPATH"
echo "PYTHONPATH updated to include: $INFERENCE_ROOT"


echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo "Your environment 'cover' is ready to use."
echo ""
echo "To activate the environment (run this in your shell):"
echo "  source $VLA_CLIP_ROOT/.venv_cover/bin/activate"
echo ""
echo "Set environment variables:"
echo "  export MUJOCO_GL=osmesa"
echo "  export PYOPENGL_PLATFORM=osmesa"
echo ""
echo "To run experiments:"
echo "  cd $INFERENCE_ROOT/experiments/robot/simpler/bashes"
echo "  ./test_pi.sh"
echo ""
