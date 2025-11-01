#!/bin/bash

#SBATCH --job-name=ev_baselinepi0_lang1_bridge_simpler
#SBATCH --output=log/slurm/eval/simpler/%x_%j.out
#SBATCH --error=log/slurm/eval/simpler/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint="a100|h100"
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --account=pr_109_tandon_advanced


# Trap Ctrl+C and clean up all child processes
trap "echo 'Ctrl+C received, killing server...'; kill $SERVER_PID; exit 1" SIGINT

CONFIG_NAMES=("pi0_baseline_bridge_ev_lang1.yaml")
SEEDS=(42 7 314)

# Function to check if a port is available
is_port_in_use() {
    ss -tuln | grep ":$1" > /dev/null
    return $?
}

find_available_port() {
    local port
    for port in $(shuf -i 10000-65500 -n 200); do
        if ! ss -tuln | grep ":$port" > /dev/null; then
            echo $port
            return 0
        fi
    done
    # Fallback to a default port if no random port is found (unlikely)
    echo 5000
    return 1
}


# set all the paths to environment variables
source ./set_path.sh

for SEED in "${SEEDS[@]}"; do
    echo "Running with seed $SEED"

    for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
        echo "Running with config $CONFIG_NAME"
        # pull the pretrained_model_gradient_step_cnt list via Python+PyYAML with a custom !include handler
        STEP_COUNTS=( $(python3 - <<EOF
import yaml, os
from yaml.loader import SafeLoader

# handle !include by loading the referenced file
def include_constructor(loader, node):
    base = os.path.dirname(os.path.abspath("config/experiment/simpler/${CONFIG_NAME}"))
    rel = loader.construct_scalar(node)
    with open(os.path.join(base, rel)) as f:
        return yaml.load(f, Loader=SafeLoader)

SafeLoader.add_constructor('!include', include_constructor)

cfg_path = "config/experiment/simpler/${CONFIG_NAME}"
with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=SafeLoader)

for s in cfg["eval_cfg"]["pretrained_model_gradient_step_cnt"]:
    print(s)
EOF
) )
        BATCH_SIZE=4
        TOTAL=${#STEP_COUNTS[@]}
        for (( i=0; i<TOTAL; i+=BATCH_SIZE )); do
            CHUNK=( "${STEP_COUNTS[@]:i:BATCH_SIZE}" )
            # spawn one server+client pair per step count, each on its own port
            SERVER_PIDS=()
            CLIENT_PIDS=()
            for STEP in "${CHUNK[@]}"; do
                # Find a random available port instead of incrementing
                PORT=$(find_available_port)
                echo "Selected random port $PORT for step $STEP"

                echo "Launching server on port $PORT for step $STEP"

                # start server for this STEP
                singularity exec --nv \
                    --bind /usr/share/nvidia \
                    --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
                    --bind /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
                    --overlay ${OVERLAY_EXT3}:ro \
                    --overlay /scratch/work/public/singularity/vulkan-1.4.309-cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sqf:ro \
                    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                    /bin/bash -c "source ./set_path.sh; \
                        export PATH='/ext3/uv:$PATH'; \
                        source ./.venv/bin/activate; \
                        python src/agent/run.py \
                        --config_path config/experiment/simpler/${CONFIG_NAME} \
                        --seed ${SEED} \
                        --use_bf16 False \
                        --eval_cfg.port ${PORT} \
                        --eval_cfg.pretrained_model_gradient_step_cnt=\"[${STEP}]\" \
                        --use_wandb False \
                        --eval_cfg.role server" &
                SERVER_PIDS+=($!)

                echo "Launching client on port $PORT for step $STEP"
                # give server a moment to bind
                sleep 2

                # start client for this STEP
                singularity exec --nv \
                    --bind /usr/share/nvidia \
                    --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
                    --bind /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
                    --overlay ${OVERLAY_EXT3}:ro \
                    --overlay /scratch/work/public/singularity/vulkan-1.4.309-cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sqf:ro \
                    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                    /bin/bash -c "source ./set_path.sh; \
                        export PATH='/ext3/uv:$PATH'; \
                        source ./src/experiments/envs/simpler/.venv/bin/activate; \
                        python src/agent/run.py \
                        --config_path config/experiment/simpler/${CONFIG_NAME} \
                        --seed ${SEED} \
                        --use_bf16 False \
                        --eval_cfg.port ${PORT} \
                        --eval_cfg.pretrained_model_gradient_step_cnt=\"[${STEP}]\" \
                        --use_wandb False \
                        --eval_cfg.role client" &
                CLIENT_PIDS+=($!)

            done

            # wait for this batch of clients
            wait "${CLIENT_PIDS[@]}"
            # then kill this batch of servers
            for pid in "${SERVER_PIDS[@]}"; do kill $pid; done
        done

        # # Run the gather_data_to_wandb.py script with the same config file
        # singularity exec \
        #     --overlay ${OVERLAY_EXT3}:ro \
        #     --overlay /scratch/work/public/singularity/vulkan-1.4.309-cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sqf:ro \
        #     /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
        #     /bin/bash -c "source ./set_path.sh; \
        #                 export PATH='/ext3/uv:$PATH'; \
        #                 source ./.venv/bin/activate; \
        #                 uv run scripts/eval/gather_data_to_wandb.py \
        #                 --config_path config/experiment/simpler/${CONFIG_NAME}"
    done
done
