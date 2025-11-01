#!/bin/bash

#SBATCH --job-name=test_rlds_dataset
#SBATCH --output=log/slurm/dataset/%x.out
#SBATCH --error=log/slurm/dataset/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=45
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zf540@nyu.edu
#SBATCH --account=pr_109_tandon_advanced

module purge
# increase limit on number of files opened in parallel to 20k --> conversion opens up to 1k temporary files in /tmp to store dataset during conversion
ulimit -n 20000

# dataset: bridge_dataset, or fractal20220817_data
singularity exec \
--overlay ../pi_overlay.ext3:ro \
/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "source ~/.bashrc;
    export PATH='/ext3/uv:$PATH'; \
    source ./.venv/bin/activate; \
    uv run python scripts/dataset/test_rlds_dataset.py \
    --config_path config/train/pi0_finetune_taco.yaml"
