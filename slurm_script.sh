#!/bin/bash
#SBATCH --job-name=debug-tests
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/stud/work/phahn/dev/logs/%A_%x_%a.out
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --array=0-26
date;hostname;pwd
source activate dal-toolbox
cd /mnt/stud/work/phahn/dev/RayWorkerBug/

output_dir=/mnt/stud/work/phahn/dev/output/{$SLURM_ARRAY_TASK_ID}_{$SLURM_JOB_ID}

# Run the deep learning script with the current hyperparameters
python -u main.py \
    output_dir=$output_dir
