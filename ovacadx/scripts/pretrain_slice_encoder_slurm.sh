#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

cd /gpfs/work4/0/tese0618/ovacadx || exit
export HOME=/gpfs/work4/0/tese0618/
export PYTHONPATH="${PYTHONPATH}:/gpfs/work4/0/tese0618/ovacadx"

mkdir -p wandb/$SLURM_JOBID

# make sure to set WANDB_API_KEY in environment variables or add it below as
# -e WANDB_API_KEY=<your_wandb_api_key>
# do not publish your API key to public repositories!
srun apptainer exec --nv "/gpfs/work4/0/tese0618/misc_v5.sif" \
    -e WANDB_DIR="/gpfs/work4/0/tese0618/ovacadx/wandb/$SLURM_JOBID" \
    -e WANDB_CONFIG_DIR="/gpfs/work4/0/tese0618/ovacadx/wandb/$SLURM_JOBID" \
    -e WANDB_CACHE_DIR="/gpfs/work4/0/tese0618/ovacadx/wandb/$SLURM_JOBID" \
    -e WANDB_START_METHOD="thread" \
    -- wandb login \
    -- python3 "experiments/pretraining/pretrain_slice_encoder.py" \
        --data_path "/gpfs/work4/0/tese0618/datasets/AbdomenCT-1K-Processed" \
        --backbone "vit_t_16" \
        --drop_last_batch \
        --pin_memory