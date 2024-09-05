"""
Logical flow for running a sweep:
1. Define the sweep configuration in a YAML file and load it in Python as a dict.
2. Initialize the sweep in Python which create it on your project/eneity in wandb platform and get the sweep_id.
3. Finally, once the sweep_id is acquired, execute the sweep using the desired number of agents in python.

ref
    - youtube video: https://www.youtube.com/watch?v=9zrmUIlScdY
    - https://chat.openai.com/share/fbf98147-3987-4d75-b7c5-52b67a1048a6
"""
import yaml
import wandb
import torch
from pathlib import Path
from pprint import pprint

# Define the training function
def train_model():
    # Initialize a new wandb run
    run = wandb.init()

    # Retrieve learning rate and number of iterations from the sweep configuration
    lr = wandb.config.lr
    num_iterations = wandb.config.num_its

    # Simulate the training process
    train_loss = 8.0 + torch.rand(1).item()

    for i in range(num_iterations):
        # Update the training loss and log the results
        update_step = lr * torch.rand(1).item()
        train_loss -= update_step
        wandb.log({"lr": lr, "train_loss": train_loss})

    # Finish the current run
    run.finish()


if __name__ == '__main__':
    # -- 1. Define the sweep configuration in a YAML file and load it in Python as a dict.
    path2sweep_config = '~/ultimate-utils/tutorials_for_myself/my_wandb_uu/my_wandb_sweeps_uu/sweep_in_python_yaml_config/sweep_config.yaml'
    config_path = Path(path2sweep_config).expanduser()
    with open(config_path, 'r') as file:
        sweep_config = yaml.safe_load(file)

    # -- 2. Initialize the sweep in Python which create it on your project/eneity in wandb platform and get the sweep_id.
    sweep_id = wandb.sweep(sweep_config, entity=sweep_config['entity'], project=sweep_config['project'])
    print(f"Sweep URL: https://wandb.ai/{sweep_config['entity']}/{sweep_config['project']}/sweeps/{sweep_id}")

    # -- 3. Finally, once the sweep_id is acquired, execute the sweep using the desired number of agents in python.
    wandb.agent(sweep_id, function=train_model, count=5)

