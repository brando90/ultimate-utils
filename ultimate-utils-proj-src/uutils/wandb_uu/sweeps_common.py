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


def wandb_sweep_config_2_sys_argv_args_str(config: dict) -> list[str]:
    """Make a sweep config into a string of args the way they are given in the terminal.
    Replaces sys.argv list of strings "--{arg_name} str(v)" with the arg vals from the config.
    This is so that the input to the train script is still an HF argument tuple object (as if it was called from
    the terminal) but overwrites it with the args/opts given from the sweep config file.
    """
    args: list[str] = [item for pair in [[f'--{arg_name}', str(v)] for arg_name, v in config.items()] for item in pair]
    return args


def exec_run_for_wandb_sweep(path2sweep_config: str,
                             function: callable,
                             pass_sweep_id: bool = False
                             ) -> None:  # str but not sure https://chat.openai.com/share/4ef4748c-1796-4c5f-a4b7-be39dfb33cc4
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
    if pass_sweep_id:
        function = lambda: function(sweep_id)
    wandb.agent(sweep_id, function=function,
                count=sweep_config.get('run_cap'))  # train does wandb.init() & run.finish()
    # return sweep_id  # not sure if I should be returning this


def setup_and_run_train(parser,
                        mode: str,
                        train: callable,
                        sweep_id: Optional[str] = None,
                        ):
    # if sweep get args from wandb.config else use cmd args (e.g. default args)
    if sweep_id is None:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()  # default args is to parse sys.argv
        run = wandb.init(mode=mode)
        train(args=(model_args, data_args, training_args), run=run)
    else:  # run sweep
        assert mode == 'online'
        run = wandb.init(mode=mode)
        # print(f'{wandb.get_sweep_url()=}')
        sweep_config = wandb.config
        args: list[str] = wandb_sweep_config_2_sys_argv_args_str(sweep_config)
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            args)  # default args is to parse sys.argv
        train(args, run)


# - examples & tests

def train_demo(args: tuple, run):
    import torch

    # usually here in the wandb demos
    # # Initialize a new wandb run
    # run = wandb.init(mode=mode)
    # # print(f'{wandb.get_sweep_url()=}')

    # unpack args
    model_args, data_args, training_args = args

    # unpack args/config
    num_its = training_args.num_its
    lr = training_args.lr

    # Simulate the training process
    train_loss = 8.0 + torch.rand(1).item()
    for i in range(num_its):
        update_step = lr * torch.rand(1).item()
        train_loss -= update_step
        wandb.log({"lr": lr, "train_loss": train_loss})

    # Finish the current run
    run.finish()

def main_example_run_train_debug_sweep_mode_for_hf_trainer(train: callable = train_demo):
    """

    idea:
    - get path2sweep_config from argparse args.
    - decide if it's debug or not from report_to


    if report_to = "none" => mode=dryrun and entity & project are None. Call agent(,count=1)
    if report_to = "wandb" => mode="online", set entity, proj from config file. Call agent(, count=run_cap)

    --
    (HF trainingargs, wandb.init)
    (report_to, mode)
    Yes, makes sense
    ("none", "disabled") yes == debug no wandb
    ("wandb", "dryrun") yes == debug & test wanbd logging

    ("wandb", "online") yes == usually means run real expt and log to wandb platform.
    No, doesn't make sense
    ("none", "dryrun") no issue, but won't log to wandb locally anyway since hf trainer wasn't instructed to do so.
    """
    from transformers import HfArgumentParser
    from uutils.hf_uu.hf_argparse.falcon_uu import ModelArguments, DataArguments, TrainingArguments

    # - run sweep or debug
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    path2sweep_config: str = training_args.path2sweep_config
    sweep_config: dict = get_sweep_config(path2sweep_config)

    # note these if stmts could've just been done with report_to hf train args opt.
    mode, report_to = sweep_config.get('mode'), sweep_config.get('report_to')
    if mode == 'online':
        # run a standard sweep. The train or setup_and_run_train func. make sure wandb.config is set correctly in args
        assert report_to == 'wandb'
        setup_and_run_train = lambda sweep_id: setup_and_run_train(parser, mode, train, sweep_id)
        exec_run_for_wandb_sweep(path2sweep_config, function=setup_and_run_train, pass_sweep_id=True)
    elif mode == 'dryrun':
        raise ValueError(f'dryrun for hf trainer not needed since its already tested if the wandb logging works')
    elif mode == 'disabled':
        assert report_to == 'none'
        setup_and_run_train(parser, mode, train, pass_sweep_id = False)


if __name__ == '__main__':
    import time

    start_time = time.time()
    main_example_run_train_debug_sweep_mode_for_hf_trainer()
    print(f"The main function executed in {time.time() - start_time} seconds.\a")
