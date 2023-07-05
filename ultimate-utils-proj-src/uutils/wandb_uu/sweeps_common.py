from pathlib import Path
from typing import Optional

import wandb
import yaml


def get_sweep_config(path2sweep_config: str) -> dict:
    """ Get sweep config from path """
    config_path = Path(path2sweep_config).expanduser()
    with open(config_path, 'r') as file:
        sweep_config = yaml.safe_load(file)
    return sweep_config


def exec_run_for_wandb_sweep(path2sweep_config: str,
                             function: callable,
                             ) -> str:  # str but not sure https://chat.openai.com/share/4ef4748c-1796-4c5f-a4b7-be39dfb33cc4
    """
    Run standard sweep from config file. Given correctly set train func., it will run a sweep in the standard way.
    Note, if entity and project are None, then wandb might try to infer them and the call might fail. If you want to
    do a debug mode, set wandb.init(mode='dryrun') else to log to the wandb plataform use 'online' (ref: https://chat.openai.com/share/c5f26f70-37be-4143-95f9-408c92c59669 unverified).
    You need to code the mode in your train file correctly yourself e.g., train = lambda : train(args) or put mode in
    the wandb_config but note that mode is given to init so you'd need to read that field from a file and not from
    wandb.config (since you haven't initialized wandb yet).

    e.g.
        path2sweep_config = '~/ultimate-utils/tutorials_for_myself/my_wandb_uu/my_wandb_sweeps_uu/sweep_in_python_yaml_config/sweep_config.yaml'

    Important remark:
        - run = wandb.init() and run.finish() is run inside the train function.
    """
    # -- 1. Define the sweep configuration in a YAML file and load it in Python as a dict.
    sweep_config: dict = get_sweep_config(path2sweep_config)

    # -- 2. Initialize the sweep in Python which create it on your project/eneity in wandb platform and get the sweep_id.
    sweep_id = wandb.sweep(sweep_config, entity=sweep_config.get('entity'), project=sweep_config.get('project'))
    print(f'{wandb.get_sweep_url()}')
    # from uutils.wandb_uu.common import _print_sweep_url
    # _print_sweep_url(sweep_config, sweep_id)

    # -- 3. Finally, once the sweep_id is acquired, execute the sweep using the desired number of agents in python.
    function = lambda: function(sweep_id)
    wandb.agent(sweep_id, function=function, count=sweep_config.get('run_cap'))  # train does wandb.init(), run.finish()
    return sweep_id


def main_example_run_train_debug_sweep():
    pass


if __name__ == '__main__':
    import time

    start_time = time.time()
    main_example_run_train_debug_sweep()
    print(f"The main function executed in {time.time() - start_time} seconds.\a")
