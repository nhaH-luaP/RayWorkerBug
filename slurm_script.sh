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

# Stuff that needs to happen before
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

IDX = $SLURM_ARRAY_TASK_ID

PORT=9000 + $IDX
NMP=9200 + $IDX
OMP=9400 + $IDX
RCSP=9600 + $IDX
RSP=9800 + $IDX
MINWP=10000 + $IDX*5000
MAXWP=10000 + ($IDX+1)*5000



# Start Ray Cluster with individual ports
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" \
        --port=$PORT \
        --node-manager-port=$NMP \
        --object-manager-port=$OMP \
        --ray-client-server-port=$RCSP \
        --redis-shard-ports=$RSP \
        --min-worker-port=$MINWP \
        --max-worker-port=$MAXWP \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &

# Run the deep learning script
srun python -u main.py \
    output_dir=$output_dir
