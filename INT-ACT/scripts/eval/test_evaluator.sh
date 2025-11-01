#!/bin/bash

# Trap Ctrl+C and clean up all child processes
trap "echo 'Ctrl+C received, killing server...'; kill $SERVER_PID; exit 1" SIGINT

CONFIG_NAMES=("test_simpler.yaml")

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    singularity exec --nv \
        --bind /usr/share/nvidia \
		--bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
	    --bind /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
        --overlay /scratch/zf540/pi0/pi_overlay.ext3:ro \
        --overlay /scratch/work/public/singularity/vulkan-1.4.309-cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sqf:ro \
        /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
        /bin/bash -c "source ~/set_path.sh; \
            export PATH='/ext3/uv:$PATH'; \
            source ./.venv/bin/activate; \
            uv run src/agent/run.py \
            --config_path config/experiment/simpler/${CONFIG_NAME} \
            --use_bf16 False \
            --eval_cfg.role server" &
    SERVER_PID=\$!

    singularity exec --nv \
        --bind /usr/share/nvidia \
		--bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
	    --bind /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
        --overlay /scratch/zf540/pi0/pi_overlay.ext3:ro \
        --overlay /scratch/work/public/singularity/vulkan-1.4.309-cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sqf:ro \
        /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
        /bin/bash -c "~/set_path.sh; \
            export PATH='/ext3/uv:$PATH'; \
            source ./src/experiments/envs/simpler_ms3/.venv/bin/activate; \
            python src/agent/run.py \
            --config_path config/experiment/simpler/${CONFIG_NAME} \
            --use_bf16 False \
            --eval_cfg.role client"
    kill $SERVER_PID

done