# Hyperparameter optimization on final datasets obtained through DAL
import os
import json
import time

import hydra
import torch
import ray
import ray.tune as tune

from omegaconf import OmegaConf
from ray.tune.search.optuna import OptunaSearch

def train(config):
    time.sleep(10)
    return {"val_metric" : torch.rand((1)).item()}


@hydra.main(version_base=None, config_path=".", config_name="cfg")
def main(args):
    print(OmegaConf.to_yaml(args))
    os.makedirs(args.output_dir, exist_ok=True)

    # Start hyperparameter search
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', args.num_cpus))
    num_gpus = torch.cuda.device_count()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

    search_space = {
        "lr": tune.loguniform(1e-5, .1), 
        "weight_decay": tune.loguniform(1e-5, .1)
        }
    
    objective = tune.with_resources(train, resources={'cpu': args.num_cpus, 'gpu': args.num_gpus})
    objective = tune.with_parameters(objective)

    search_alg = OptunaSearch(points_to_evaluate=[{'lr': args.lr, 'weight_decay': args.weight_decay}])
    tune_config = tune.TuneConfig(search_alg=search_alg, num_samples=args.num_opt_samples, metric="val_metric", mode="min")

    tuner = tune.Tuner(objective, param_space=search_space, tune_config=tune_config)
    results = tuner.fit()

    print('Best Hyperparameters: {}'.format(results.get_best_result(metric='val_metric', mode='min').config))
    print('Saving results.')
    history = {
        'empty':None 
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(history, f)


if __name__ == '__main__':
    main()
