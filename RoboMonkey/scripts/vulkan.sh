#!/bin/bash

# Setup Vulkan dependencies for SIMPLER
# Reference: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting


sudo apt-get install libvulkan1 vulkan-utils xvfb -y
sudo mkdir -p /usr/share/vulkan/icd.d/ && echo '{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155"
    }
}' | sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json