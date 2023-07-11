from argparse import Namespace
from pathlib import Path
from typing import Union

import wandb
import yaml
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

import uutils

from pdb import set_trace as st


# def dict_to_namespace(data: dict):
#     if isinstance(data, dict):
#         return Namespace(**{k: dict_to_namespace(v) for k, v in data.items()})
#     elif isinstance(data, list):
#         return [dict_to_namespace(v) for v in data]
#     else:
#         return data

def get_sweep_url_from_run(run: Run) -> str:
    """ https://stackoverflow.com/questions/75852199/how-do-i-print-the-wandb-sweep-url-in-python/76624367#76624367 """
    return run.get_sweep_url()


def get_sweep_url_from_config(sweep_config: dict, sweep_id: str) -> str:
    sweep_url = f"Sweep URL: https://wandb.ai/{sweep_config['entity']}/{sweep_config['project']}/sweeps/{sweep_id}"
    return sweep_url


def get_sweep_url_from_entity_project_sweep_id(entity: str, project: str, sweep_id: str) -> str:
    """

    https://wandb.ai/{username}/{project}/sweeps/{sweep_id}
    """
    api = wandb.Api()
    sweep = api.sweep(f'{entity}/{project}/{sweep_id}')
    return sweep.url


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
    print(f'wandb sweep url (uutils): {get_sweep_url_from_config(sweep_config, sweep_id)}')

    # -- 3. Finally, once the sweep_id is acquired, execute the sweep using the desired number of agents in python.
    wandb.agent(sweep_id, function=function, count=sweep_config.get('run_cap'))  # train does wandb.init(), run.finish()
    return sweep_id


def setup_wandb_for_train_with_hf_trainer(args: Namespace,
                                          ) -> tuple[wandb.Config, Union[Run, RunDisabled, None]]:
    """
    Set up wandb for the train function that uses hf trainer. If report_to is none then wandb is disabled o.w. if
    report_to is wandb then we set the init to online to log to wandb platform. Always uses config to create the
    run config. It uses wandb.config for a sweep or a debug config (via args.path2debug_config) for report_to none runs.
    """
    report_to = args.report_to
    mode = 'disabled' if report_to == 'none' else 'online'  # no 'dryrun' since wandb logging is already tested by hf
    print(f'{mode=}')
    run: Union[Run, RunDisabled, None] = wandb.init(mode=mode)
    print(f'{run=}')
    # - discover what type of run your doing (no wandb or sweep with wandb)
    print(f'{report_to=}')
    if report_to == 'none':
        # - use debug config from file
        config: wandb.Config = wandb.Config()
        config.update(vars(args))
        config_dict: dict = get_sweep_config(args.path2config)
        config.update(config_dict)
    else:  # then load the debug config
        # https://docs.wandb.ai/ref/python/run?_gl=1*80ki1e*_ga*MTYwMTE3MDYzNS4xNjUyMjI2MTE1*_ga_JH1SJHJQXJ*MTY4ODU5NDI0NS4zMDAuMS4xNjg4NTk1MDg3LjU5LjAuMA..
        print(f'{run.get_sweep_url()=}')
        # - use the sweep config sent from wandb in wandb.config
        config: wandb.Config = wandb.config
        config.update(vars(args))
    return config, run


# - examples & tests

def train_demo(args: Namespace):
    import torch

    # - init run, if report_to is wandb then: 1. sweep use online args merges with sweep config, else report_to is none and wandb is disabled
    config, run = setup_wandb_for_train_with_hf_trainer(args)
    print(f'{config=}')
    uutils.pprint_any_dict(config)

    # Simulate the training process
    num_its = config.get('num_its')  # usually obtained from args or config
    lr = config.get('lr')  # usually obtained from args or config
    train_loss = 8.0 + torch.rand(1).item()
    for i in range(num_its):
        train_loss -= lr * torch.rand(1).item()
        run.log({"lr": lr, "train_loss": train_loss})

    # Finish the current run
    run.finish()


def main_example_run_train_debug_sweep_mode_for_hf_trainer():
    """
python -m pdb -c continue /Users/brandomiranda/ultimate-utils/ultimate-utils-proj-src/uutils/wandb_uu/sweeps_common.py --report_to none
python -m pdb -c continue /Users/brandomiranda/ultimate-utils/ultimate-utils-proj-src/uutils/wandb_uu/sweeps_common.py --report_to wandb
    """
    from uutils.hf_uu.hf_argparse.common import get_simple_args

    # - get most basic hf args args
    args: Namespace = get_simple_args()  # just report_to, path2sweep_config, path2debug_seep
    print(args)

    # - run train
    report_to = args.report_to
    if report_to == "none":
        train: callable = train_demo
        train(args)
    elif report_to == "wandb":
        path2sweep_config = args.path2sweep_config
        train = lambda: train_demo(args)
        exec_run_for_wandb_sweep(path2sweep_config, train)
    else:
        raise ValueError(f'Invaid hf report_to option: {report_to=}.')


if __name__ == '__main__':
    import time

    start_time = time.time()
    main_example_run_train_debug_sweep_mode_for_hf_trainer()
    print(f"The main function executed in {time.time() - start_time} seconds.\a")
