from pathlib import Path

from pprint import pprint
import torch

import yaml
import wandb

# Load YAML file into Python dictionary
path: Path = Path(
    '/Users/brandomiranda/ultimate-utils/tutorials_for_myself/my_wandb_uu/my_wandb_sweeps_uu/sweep_in_python_yaml_config/sweep_config.yaml').expanduser()
with open(path, 'r') as f:
    sweep_config = yaml.safe_load(f)
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
# wandb.get_sweep_url()
