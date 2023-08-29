#!/bin/bash
#SBATCH --job-name=ray_debug
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --output=/mnt/stud/home/phahn/dev/logs/%A_%x_%a.out
#SBATCH --array=0-99
hostname

source ~/envs/rayworkerbug/bin/activate

cd ~/dev/RayWorkerBug/

output_dir=/mnt/stud/home/phahn/dev/output/${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}/

# Run the deep learning script
srun python -u main.py \
    output_dir=$output_dir \
    num_cpus=16 \
    num_gpus=1 \
    num_cpus_per_trial=16 \
    num_gpus_per_trial=1
