#!/bin/bash

#SBATCH --job-name=register-segmentation
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --account=project_2005092
#SBATCH --partition=gpusmall

# If run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    sbatch "$0" "$@"
    exit
fi

module use /appl/local/csc/modulefiles
module load pytorch/2.5

source /scratch/project_2011109/erik/llm-descriptor-classifier/venv11/bin/activate

MODEL="$1"
DATASET="$2"

# Check if input and output files are provided
if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "Usage: sbatch slurm.sh model dataset"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p slurm-logs

# Run the Python script using srun
srun python 1_train.py "$MODEL" "$DATASET"