"""

"""
import argparse
from argparse import Namespace
from typing import Union

import transformers

from transformers import HfArgumentParser
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from uutils.hf_uu.common import report_to2wandb_init_mode


# -- simple args

def get_simple_args() -> Namespace:
    """
    Get simple args for experiment. The idea is that all arguments are in the config file (for the sweep) so it's only
    needed to give the config path.

    todo: fix wandb so that only 1 config file has to be maintain for real expts vs debug: https://community.wandb.ai/t/generating-only-a-local-concrete-set-of-values-for-a-sweep-locally-without-logging-remotely/4692
    """
    # - great terminal argument parser
    parser = argparse.ArgumentParser()

    # -- create argument options
    parser.add_argument('--report_to', type=str, default='none', help='')
    parser.add_argument('--path2config', type=str,
                        default='~/ultimate-utils/ultimate-utils-proj-src/uutils/wandb_uu/sweep_configs/debug_config.yaml',
                        help='Its recommended to avoid running a random default. Example params: '
                             '--path2config ~/ultimate-utils/ultimate-utils-proj-src/uutils/wandb_uu/sweep_configs/sweep_config.yaml'
                             '--path2config ~/ultimate-utils/ultimate-utils-proj-src/uutils/wandb_uu/sweep_configs/debug_config.yaml'
                        )

    # -- parse args and return args namespace obj
    args: Namespace = parser.parse_args()
    return args


# --

def wandb_sweep_config_2_sys_argv_args_str(config: dict) -> list[str]:
    """Make a sweep config into a string of args the way they are given in the terminal.
    Replaces sys.argv list of strings "--{arg_name} str(v)" with the arg vals from the config.
    This is so that the input to the train script is still an HF argument tuple object (as if it was called from
    the terminal) but overwrites it with the args/opts given from the sweep config file.
    """
    args: list[str] = [item for pair in [[f'--{arg_name}', str(v)] for arg_name, v in config.items()] for item in pair]
    return args


def setup_wandb_for_train_hf_trainer_with_parser(parser: HfArgumentParser,
                                                 ) -> tuple[tuple, Union[Run, RunDisabled, None]]:
    """
    Set up wandb for the train function that uses hf trainer. If report_to is none then wandb is disabled o.w. if
    report_to is wandb then we set the init to online to log to wandb platform.

    Note:
        - for now your parser needs to have 3 dataclasses due to this line in the code:
            e.g. model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        - note this could be a context manager. Do nothing for now. Just call run.finish() in train.
    """
    import wandb
    # - get the report_to (again) from args to init wandb for your hf trainer
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    report_to = training_args.report_to
    mode = report_to2wandb_init_mode(report_to)
    run: Union[Run, RunDisabled, None] = wandb.init(mode)
    # - discover what type of run your doing (no wandb or sweep with wandb)
    if isinstance(wandb.config, wandb.Config):  # then you are in a sweep!
        print(f'{wandb.get_sweep_url()}')
        # - overwrite the args using the config
        config = wandb.config
        args = wandb_sweep_config_2_sys_argv_args_str(config)
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)
    else:  # then load the debug config or use the defaults from args
        # I realized that if I'm already using the arg parse then the defaults could be there and no need for default config when using hf wandb
        # config: dict = get_sweep_config(training_args.path2debug_config)
        # args = wandb_sweep_config_2_sys_argv_args_str(config)
        # args = parser.parse_args_into_dataclasses(args)
        pass
    return (model_args, data_args, training_args), run


# - examples & tests

def train_demo(parser: HfArgumentParser):
    import torch

    # - init run, if report_to is wandb then 1. sweep use online args merges with sweep config, else report_to is none and wandb is disabled
    args, run = setup_wandb_for_train_hf_trainer_with_parser(parser)
    model_args, data_args, training_args = args
    print(model_args, data_args, training_args)

    # Simulate the training process
    num_its = 5  # usually obtained from args or config
    lr = 0.1  # usually obtained from args or config
    train_loss = 8.0 + torch.rand(1).item()
    for i in range(num_its):
        train_loss -= lr * torch.rand(1).item()
        run.log({"lr": lr, "train_loss": train_loss})

    # Finish the current run
    run.finish()


def main_example_run_train_debug_sweep_mode_for_hf_trainer(train: callable = train_demo):
    """

python ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/hf_argparse/common.py --report_to none

python ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/hf_argparse/common.py --report_to wandb
    """
    from uutils.wandb_uu.sweeps_common import exec_run_for_wandb_sweep
    from uutils.hf_uu.hf_argparse.falcon_uu_training_args import ModelArguments, DataArguments, TrainingArguments

    # - get simple args, just report_to, path2sweep_config, path2debug_seep
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # - run train
    report_to = training_args.report_to
    train = train_demo
    if report_to == "none":
        train(parser)
    elif report_to == "wandb":
        path2sweep_config = training_args.path2sweep_config
        train = lambda: train(parser)
        exec_run_for_wandb_sweep(path2sweep_config, train)
    else:
        raise ValueError(f'Invaid hf report_to option: {report_to=}.')


if __name__ == '__main__':
    import time

    start_time = time.time()
    main_example_run_train_debug_sweep_mode_for_hf_trainer()
    print(f"The main function executed in {time.time() - start_time} seconds.\a")
