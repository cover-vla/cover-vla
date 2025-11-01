To see how we fine-tune the $\pi_0$ model, please refer to [pi_finetune_bridge.sh](../slurms/train_scripts/pi0_finetune_bridge.sh).

Our scripts are all designed to be run on a SLURM cluster. Here, we will break down the scripts so you can reuse them on your cluster or local machine without SLURM.

Although not thoroughly tested, we believe that by removing Singularity-related commands as recommended in the following section, you can run the script directly on local workstations to perform training and fine-tuning. We use almost identical scripts for training from scratch and fine-tuning, with only differences in respective configuration files.

## SLURM Script Breakdown
We will use the aforementioned script as an example to explain the components of an SLURM script.
### Directives
```bash
#!/bin/bash

#SBATCH --job-name=pi0_finetune_bridge
#SBATCH --output=log/slurm/train/%x.out
#SBATCH --error=log/slurm/train/%x.err
#SBATCH --time=44:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100|h100"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=440G
#SBATCH --account=pr_109_tandon_advanced
#SBATCH --requeue
```
This is the header of the SLURM script. `#!/bin/bash` is the so-called shebang, indicating that the script would be run in a bash shell. 

The `#SBATCH` lines specify various parameters for the job and should be self-explanatory. For example, you can see that we fine-tune the models on 4 A100/H100 GPUs.

**When doing local training**, you can remove all of these except the shebang.

### Environment Variables
```bash
# set all the paths to environment variables
source ./set_path.sh
```
This line sets all kinds of environment variables, which are mostly about paths to store models, logs, data, etc. The main README.md file has a section on how to set up the environment variables, so please refer to that.

### OMP Threads
```bash
# CPU check
TOTAL_CORES=$(nproc)
echo "TOTAL_CORES=$TOTAL_CORES"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

# Compute OMP_NUM_THREADS (avoid division by zero)
OMP_THREADS=$((TOTAL_CORES / NUM_GPU))

# Ensure OMP_NUM_THREADS is at least 1
OMP_THREADS=$((OMP_THREADS > 0 ? OMP_THREADS : 1))
echo "OMP_NUM_THREADS=$OMP_THREADS"
```
This section computes the number of OpenMP threads to use in multi-GPU training. Technically, it should help with performance, but empirically, we found no detectable improvement. You can remove this section if you want.

### Multi-GPU Setup
```bash
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)
```
This section sets up the master address and port for distributed training. The `MASTER_ADDR` is set to the first node in the SLURM job, and `MASTER_PORT` is set to a free port on that node.

**We only test our code on a single node with multiple GPUs.** Multi-node training is not supported in our codebase.

### Training Command
```bash
echo "Job restart count: $SLURM_RESTART_COUNT"

singularity exec --nv \
--overlay ${OVERLAY_EXT3}:ro \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "source ./set_path.sh; \
    export PATH='/ext3/uv:$PATH'; \
    source ./.venv/bin/activate; \
    export OMP_NUM_THREADS=$OMP_THREADS; \
    uv run torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPU \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --max-restarts=0 \
    --standalone \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/agent/run.py \
    --config_path config/train/pi0_finetune_bridge.yaml" || scontrol requeue $SLURM_JOB_ID
```
This is the main command that runs the training with Singularity.

**If you are not using singularity** (e.g., running locally), you can remove the `singularity exec` line and just run the command inside the quotes directly in your terminal. That is, remove everything before `"source ...`. If you do use Singularity, please refer to your cluster documentation to understand Singularity commands.

Everything after `torchrun` is standard distributed training arguments. You can use ChatGPT to help you understand them.

`--config_path config/train/pi0_finetune_bridge.yaml` specifies the configuration file for the training. We will explain the configuration file in the next section.

You may notice the `|| scontrol requeue $SLURM_JOB_ID` at the end of the command. Together with the `#SBATCH --requeue` directive and `echo "Job restart count: $SLURM_RESTART_COUNT"`,
this allows the job to be automatically resubmitted if it fails. This is to handle an unresolved bug that happens almost whimsically on our cluster. You can remove this part if you don't need it.

## Configuration Files Breakdown
We use `draccus` to manage configurations. 

### Training Configuration
[pi0_finetune_bridge.yaml](../config/train/pi0_finetune_bridge.yaml) is the configuration file for fine-tuning the $\pi_0$ model. 

We use a `yaml` file in [config/train](../config/train) to define the training configuration. The dataclass that defines the configuration can be found in [src/agent/configuration_pipeline.py](../src/agent/configuration_pipeline.py), especially the `TrainPipelineConfig` class.

Things like local batch size, global batch size, gradient accumulation steps, and wandb logging are defined there. The naming should be self-explanatory, but if you have any questions, feel free to post an issue on GitHub.

### Model Configuration
We use a `json` file in [config/model](../config/train) to define the model configuration. This is to follow the LeRobot/HuggingFace convention. The dataclass that defines the model configuration can be found in their corresponding [LeRobot Repo](https://huggingface.co/lerobot/pi0/blob/main/config.json). 

The model configuration is loaded by the training configuration and can be changed there as well. Interestingly, the LeRobot convention is that the model configuration defines learning rate, weight decay, and some other related hyperparameters.

#### Placeholder Configuration for Non-LeRobot Models
As you may have noticed, we have some placeholders for models that do not follow the LeRobot convention, such as Magma, SpatialVLA, Octo, and such. More details about them will be provided in the evaluation documentation.
