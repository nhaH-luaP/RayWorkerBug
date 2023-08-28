# RayWorkerBug
Demonstration of a bug when running an array of jobs on a Slurm-Cluster, each containing a ray optimization process.

## How to reproduce

- Create a fresh virtual environment with ```python -m venvs ~/envs/rayworkerbug```
- Change output/log directories inside the ``slurm_script.sh`` according to your system
- Run experiment with ```sbatch slurm_script.sh```

## Common Error Message

```
core_worker.cc:201: Failed to register worker 01000000ffffffffffffffffffffffffffffffffffffffffffffffff to Raylet. IOError: [RayletClient] Unable to register worker with raylet. No such file or directory
```
