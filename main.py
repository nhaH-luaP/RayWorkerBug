import os
import json
import time

import hydra
import torch
import ray
import ray.tune as tune

import ray.air as air
import numpy as np

from omegaconf import OmegaConf
from ray.tune.search.optuna import OptunaSearch

def train(config):
    time.sleep(5)
    return {"val_metric" : torch.rand((1)).item()}


@hydra.main(version_base=None, config_path=".", config_name="cfg")
def main(args):
    print(OmegaConf.to_yaml(args))
    os.makedirs(args.output_dir, exist_ok=True)

    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
    num_gpus = torch.cuda.device_count()
    print('>>> Num CPUS:',num_cpus,' Num GPUS:',num_gpus)
    
    ray.init(
        num_cpus=num_cpus, 
        num_gpus=num_gpus, 
        include_dashboard=False,
        _temp_dir=args.output_dir
        )

    search_space = {"param": tune.uniform(0, .1)}
    objective = tune.with_resources(train, resources={'cpu': args.cpus_per_trial, 'gpu': args.gpus_per_trial})
    tune_config = tune.TuneConfig(num_samples=args.num_opt_samples, metric="val_metric", mode="min")
    run_config = air.RunConfig(storage_path=args.output_dir)
    tuner = tune.Tuner(objective, param_space=search_space, tune_config=tune_config, run_config=run_config)
    
    results = tuner.fit()

    print('>>>>> FINISHED')


if __name__ == '__main__':
    main()
