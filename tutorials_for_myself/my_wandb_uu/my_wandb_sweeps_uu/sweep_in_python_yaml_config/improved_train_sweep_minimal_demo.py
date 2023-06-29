"""
Procedure for executing the sweep:
1. Define the sweep configuration in a YAML file and import it into python.
2. Initialize the sweep on the wandb platform via python and retrieve the sweep_id.
3. Once the sweep_id is acquired, execute the sweep using the desired number of agents.

ref: https://chat.openai.com/share/fbf98147-3987-4d75-b7c5-52b67a1048a6
"""

import yaml
import wandb
import torch
from pathlib import Path
from pprint import pprint

# Load YAML file into Python dictionary
config_path = Path(
    '~/ultimate-utils/tutorials_for_myself/my_wandb_uu/my_wandb_sweeps_uu/sweep_in_python_yaml_config/sweep_config.yaml').expanduser()

with open(config_path, 'r') as file:
    sweep_config = yaml.safe_load(file)

pprint(sweep_config)

# Retrieve entity and project details from the configuration
entity = sweep_config['entity']
project = sweep_config['project']

# Initialize sweep on wandb's platform & obtain sweep_id to create agents
sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
print(f'Sweep ID: {sweep_id}')
print(f"Sweep URL: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
wandb.get_sweep_url()


# Define the training function
def train_model():
    # Initialize a new wandb run
    run = wandb.init()
    wandb.get_sweep_url()

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


# Execute the sweep with a specified number of agents
wandb.agent(sweep_id, function=train_model, count=5)

print(f"Sweep URL: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
wandb.get_sweep_url()
