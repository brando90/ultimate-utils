from pathlib import Path

import wandb
import yaml

def get_sweep_config(path2sweep_config: str) -> dict:
    """ Get sweep config from path """
    config_path = Path(path2sweep_config).expanduser()
    with open(config_path, 'r') as file:
        sweep_config = yaml.safe_load(file)
    return sweep_config


def exec_run_for_wandb_sweep(path2sweep_config: str,
                             train: callable,
                             count: int = 5,
                             ) -> str:  # str but not sure https://chat.openai.com/share/4ef4748c-1796-4c5f-a4b7-be39dfb33cc4
    """
    Run standard sweep from config file.

    e.g.
        path2sweep_config = '~/ultimate-utils/tutorials_for_myself/my_wandb_uu/my_wandb_sweeps_uu/sweep_in_python_yaml_config/sweep_config.yaml'

    Important remark:
        - run = wandb.init() and run.finish() is run inside the train function.
    """
    # -- 1. Define the sweep configuration in a YAML file and load it in Python as a dict.
    sweep_config: dict = get_sweep_config(path2sweep_config)

    # -- 2. Initialize the sweep in Python which create it on your project/eneity in wandb platform and get the sweep_id.
    sweep_id = wandb.sweep(sweep_config, entity=sweep_config['entity'], project=sweep_config['project'])
    print(f'{wandb.get_sweep_url()}')
    # from uutils.wandb_uu.common import _print_sweep_url
    # _print_sweep_url(sweep_config, sweep_id)

    # -- 3. Finally, once the sweep_id is acquired, execute the sweep using the desired number of agents in python.
    wandb.agent(sweep_id, function=train, count=count)  # train in charge of doing run = wandb.init() and run.finish()
    return sweep_id
