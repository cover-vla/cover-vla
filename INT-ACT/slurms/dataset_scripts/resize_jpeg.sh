#!/bin/bash

#SBATCH --job-name=resize_jpeg
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

singularity exec \
--overlay ../pi_overlay.ext3:ro \
/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "export PATH='/ext3/uv:$PATH'; \
    source ./.venv/bin/activate; \
    uv run python scripts/dataset/modify_rlds_dataset.py \
        --dataset=libero_90_openvla_processed \
        --data_dir=/scratch/zf540/pi0/open-pi-zero/data/temp/modified_libero_rlds/libero_90_rlds \
        --target_dir=$VLA_DATA_DIR/resize_224 \
        --mods=resize_and_jpeg_encode \
        --n_workers=40 \
        --max_episodes_in_memory=200"