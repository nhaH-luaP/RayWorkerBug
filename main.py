import os
import json
import time

import hydra

import ray
import ray.tune as tune
import ray.air as air

import numpy as np

def train(config):
    time.sleep(5)
    return {"val_metric" : np.random.rand((1)).item()}


@hydra.main(version_base=None, config_path=".", config_name="cfg")
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Ray
    ray.init(
        num_cpus=args.num_cpus, 
        num_gpus=args.num_gpus, 
        include_dashboard=False,
        _temp_dir=args.output_dir
        )

    # Settings for optimization
    search_space = {"param": tune.uniform(0, .1)}
    objective = tune.with_resources(train, resources={'cpu': args.num_cpus_per_trial, 'gpu': args.num_gpus_per_trial})
    tune_config = tune.TuneConfig(num_samples=args.num_opt_samples, metric="val_metric", mode="min")
    run_config = air.RunConfig(storage_path=args.output_dir)
    tuner = tune.Tuner(objective, param_space=search_space, tune_config=tune_config, run_config=run_config)
    
    # Run optimization
    results = tuner.fit()


if __name__ == '__main__':
    main()
