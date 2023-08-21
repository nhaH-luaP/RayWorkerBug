#!/bin/bash
#SBATCH --job-name=debug-tests
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --output=/mnt/stud/work/phahn/dev/logs/%A_%x_%a.out
#SBATCH --array=0-99
hostname

source activate rayworkerbug
cd /mnt/stud/work/phahn/dev/RayWorkerBug/

output_dir=/mnt/stud/work/phahn/dev/output/${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}/

# Run the deep learning script
srun python -u main.py \
    output_dir=$output_dir
