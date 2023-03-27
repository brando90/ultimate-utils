"""
Main Idea:
- create sweep with a sweep config & get sweep_id for the agents (note, this creates a sweep in wandb's website)
- create agent to run a setting of hps by giving it the sweep_id (that mataches the sweep in the wandb website)
- keep running agents with sweep_id until you're done

note:
    - Each individual training session with a specific set of hyperparameters in a sweep is considered a wandb run.

ref:
    - read: https://docs.wandb.ai/guides/sweeps
"""

import wandb
from pprint import pprint
import math
import torch

sweep_config: dict = {
    "project": "playground",
    "entity": "your_wanbd_username",
    "name": "my-ultimate-sweep",
    "metric":
        {"name": "train_loss",
         "goal": "minimize"}
    ,
    "method": "random",
    "parameters": None,  # not set yet
}

parameters = {
    'optimizer': {
        'values': ['adam', 'adafactor']}
    ,
    'scheduler': {
        'values': ['cosine', 'none']}  # todo, think how to do
    ,
    'lr': {
        "distribution": "log_uniform_values",
        "min": 1e-2,
        "max": 0.9}
    ,
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
    }
    ,
    # it's often the case that some hps we don't want to vary in the run e.g. num_its
    'num_its': {'value': 5}
}
sweep_config['parameters'] = parameters
pprint(sweep_config)
entity = sweep_config['entity']
project = sweep_config['project']

# create sweep in wandb's website & get sweep_id to create agents that run a single agent with a set of hps
sweep_id = wandb.sweep(sweep_config)
print(f'{sweep_id=}')
print(f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")

def my_train_func():
    # read the current value of parameter "a" from wandb.config
    # I don't think we need the group since the sweep name is already the group
    run = wandb.init(config=sweep_config)
    print(f'{run=}')
    pprint(f'{wandb.config=}')
    lr = wandb.config.lr
    num_its = wandb.config.num_its

    train_loss: float = 8.0 + torch.rand(1).item()
    for i in range(num_its):
        # get a random update step from the range [0.0, 1.0] using torch
        update_step: float = lr * torch.rand(1).item()
        wandb.log({"lr": lr, "train_loss": train_loss - update_step})
    print(f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    run.finish()


# run the sweep, The cell below will launch an agent that runs train 5 times, usingly the randomly-generated hyperparameter values returned by the Sweep Controller.
wandb.agent(sweep_id, function=my_train_func, count=5)
print(f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
