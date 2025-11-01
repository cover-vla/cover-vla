To see how we evaluate a model, please refer to [ev_pi0_bridge_simpler.sh](../slurms/eval_scripts/simpler/ev_pi0_bridge_simpler.sh).

INT-ACT essentially is an extension of SimplerEnv, so theoretically, you can just clone our forked [SimplerEnv](https://github.com/juexZZ/SimplerEnv/tree/fea52fbb9e0da8a2e4e7e5a155b8e5b7f9dd5b87) and [Maniskill2](https://github.com/juexZZ/ManiSkill2_real2sim/tree/eeb04c788feafdf08f4565bcda34370e5a555325) and run the evaluation like you would with SimplerEnv. This README provides a more detailed explanation of how we evaluate the models presented in our paper.

Similar to training scripts, evaluation scripts are designed to be run on an SLURM cluster. Here, we will break down the scripts so you can reuse them on your cluster or local machine without SLURM.


## SLURM Script Breakdown
Eval scripts are, in general, much more complicated. Maniskill2, which SimplerEnv is based on, does not natively support GPU parallelism for simulation, so you can only evaluate one scene at a time, resulting in a slow evaluation. So we had to implement these gimmicks to parallelize the evaluation process. 

We have tried Maniskill3-based SimplerEnv, which natively supports GPU parallelism, but we found two issues: 
1. Discrepancy in metrics (task success rate, etc) between Maniskill2 and Maniskill3 is pretty significant. 
2. Maniskill3 has some unresolved memory leak issues

So, for now, we will stick with Maniskill2.

### Directives
#!/bin/bash
```bash
#SBATCH --job-name=ev_pi0_bridge_simpler
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

trap "echo 'Ctrl+C received, killing server...'; kill $SERVER_PID; exit 1" SIGINT
```
The shebang and directives are similar to the training scripts. Note that we use significantly fewer compute resources. 

The `trap` command is a relic for debugging. Feel free to completely remove it if you don't intend to run the script interactively.

Similar to training, **when doing local training**, you can remove all of these except the shebang.

### Config Files and Random Seed
```bash
CONFIG_NAMES=("pi0_finetune_bridge_ev.yaml")
SEEDS=(42 7 314)
```
This section defines which config files and random seeds to use. As you may have noticed, in theory, you can put multiple config files and the script will run them sequentially (e.g., `CONFIG_NAMES=("config1.yaml" "config2.yaml")`). However, we only use one config file in this example.

Random seeds here will affect the random seed used in all RNG-based operations in `torch`, `numpy`, etc that we can think of. This is for reproducibility.

### Idle Port Finding
```bash
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
```
Our server-client architecture for inference uses `websocket` under the hood. To avoid race conditions between parallel evaluations, we need to find available ports for each client-server pair. The `is_port_in_use` function is a relic and can be removed. `find_available_port` is the function that's actually being used.

**Even when running locally**, it's recommended to keep this section.

### Environment Variables
```bash
# set all the paths to environment variables
source ./set_path.sh
```
Set the environment variables for paths. Same as the training script.

### Iterate Over Random Seeds and Config Files
```bash
for SEED in "${SEEDS[@]}"; do
    echo "Running with seed $SEED"

    for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
        echo "Running with config $CONFIG_NAME"
```
We iterate over the random seeds and config files defined earlier. These are evaluated sequentially. No parallelism happening.

### Iterate Over Checkpoint Steps
```bash
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
```
Here we are doing something very gimmicky. The `pretrained_model_gradient_step_cnt` is a list of steps to evaluate the model at, defined in `config/experiment/simpler/${CONFIG_NAME}`. We use Python to read the config file and extract the steps, which are then stored in `STEP_COUNTS`, so that we can iterate over them later.

### Eval Loop
```bash
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
```
In our experience, on one 80GB A100/H100, we can evaluate 4 pairs of server+client in parallel, so we set `BATCH_SIZE=4`. You can adjust this number to match your GPU's memory.

Then, we iterate over configs and random seed combinations, launching 4 pairs of server+client simultaneously, each pair using a checkpoint at a specific gradient step.

### Launch Server and Client
```bash
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
    done
done
```
Here we launch the server and client pair. We use `SERVER_PIDS+=($!)` and `CLIENT_PIDS+=($!)` to store the PIDs of the launched processes.

Later on, we wait for all clients to finish with `wait "${CLIENT_PIDS[@]}"`, and then kill the servers with `kill $pid`. This is because the server is designed to run indefinitely, so we need to kill it after the client finishes.

There is another gimmick here. We use `--eval_cfg.pretrained_model_gradient_step_cnt=\"[${STEP}]\"` to pass the specific gradient step to the server and client. You may recall that we previously extracted a list of steps from the config file. This is actually overwriting that list with a single step, so that the server and client only evaluate the model at that specific step. That list is already extracted, so we can safely overwrite it.

## Evaluate Trained and Third-Party/Non-LeRobot Models
### Trained Models
$\pi_0$ are fine-tuned and trained by us using LeRobot. You can evaluate them using the provided evaluation script. You can also refer to the [config file](../config/experiment/simpler/pi0_finetune_bridge_ev.yaml) for the evaluation settings.


You can change `eval_cfg.pretrained_model_gradient_step_cnt`, which is a list of steps to evaluate the model at.


### Third-Party/Non-LeRobot Models
To evaluate third-party models, such as Magma, Octo, SpatialVLA, etc., or any models you develop without LeRobot, there are some works to do. 

1. **Create LeRobot Compatible Placeholder**: See [Magma Folder](../src/model/magma) to see how we create a placeholder for Magma. You will need to create a `configuration_{your_model}.py` which defines the model's configuration, and a `modeling_{your_model}.py` which defines the model's architecture. Of course, these files don't need to hold anything substantial, because the actual modeling/configuration is done in a third-party package (potentially of your own), like Magma, Octo, etc.

2. **Install the Third-Party Model**: You will need to install the third-party model. Please follow [inference server installation section of the main README](README.md#octo-and-magma-install-inference-server-policy-environment)

3. **Modify the entry point**: Please modify the  [entry point script](../src/agent/run.py) accordingly so you can launch your own model's evaluation.

4. **Create an experiment config file**: You will need to create a config file under `config/experiment/simpler/` that defines the evaluation settings for your model. You can use the existing [config files](../config/experiment/simpler/magma_bridge_ev_lang1.yaml) as a reference. For `eval_cfg.pretrained_model_gradient_step_cnt`, our approach is to set an arbitrary number, such as 15130, which is the number of steps we trained $\pi_0$ on BridgeV2. This is because many third-party models do not have a gradient step count, so we just use a placeholder.
