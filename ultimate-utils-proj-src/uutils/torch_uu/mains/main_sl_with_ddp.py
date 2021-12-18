"""
Main script to set up meta-learning experiments
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
# import torch.optim as optim

from argparse import Namespace

from uutils import args_hardcoded_in_script, report_times
from uutils.argparse_uu.common import setup_args_for_experiment
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint, parse_args_standard_sl
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.agents.supervised_learning import ClassificationSLAgent
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.dataloaders.helpers import get_sl_dataloader
from uutils.torch_uu.distributed import set_sharing_strategy, print_process_info, set_devices, setup_process, \
    move_to_ddp, cleanup, print_dist
from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_first_time
from uutils.torch_uu.training.supervised_learning import train_agent_fit_single_batch, \
    train_agent_fixed_number_of_iterations, train_agent_fixed_number_of_epochs, train_agent_fit_until_convergence, \
    train_agent_fit_until_perfect_train_accuracy


def manual_load(args) -> Namespace:
    """
    Warning: hardcoding the args can make it harder to reproduce later in a main.sh script with the
    arguments to the experiment.
    """
    raise ValueError(f'Not implemented')


def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    # -- parse args from terminal
    args: Namespace = parse_args_standard_sl()
    args.wandb_project = 'playground'  # needed to log to wandb properly

    # - debug args
    args.experiment_name = f'debug'
    args.run_name = f'debug (Adafactor) : {args.jobid=}'
    args.force_log = True
    # args.log_to_wandb = True
    args.log_to_wandb = False

    # - real args
    # args.experiment_name = f'Real experiment name (Real)'
    # args.run_name = f'Real experiment run name: {args.jobid=}'
    # args.force_log = False
    # args.log_to_wandb = True
    # #args.log_to_wandb = False

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        args: Namespace = manual_load(args)
    else:
        # NOP: since we are using args from terminal
        pass
    # -- Setup up remaining stuff for experiment
    args: Namespace = setup_args_for_experiment(args, num_workers=4)
    return args


def main():
    """
    train tree_nn in parallel
    Note: end-to-end ddp example on mnist: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    :return:
    """
    # - load the args from either terminal, ckpt, manual etc.
    args: Namespace = load_args()
    [print(f'{k, v}') for k, v in vars(args).items()]

    # - parallel train
    if not args.parallel:  # serial
        print('RUNNING SERIALLY')
        args.world_size = 1
        train(rank=-1, args=args)
    else:
        print(f"{torch.cuda.device_count()=}")
        args.world_size = torch.cuda.device_count()
        # args.world_size = mp.cpu_count() - 1  # 1 process is main, the rest are (parallel) trainers
        set_sharing_strategy()
        mp.spawn(fn=train, args=(args,), nprocs=args.world_size)


def train(rank, args):
    print_process_info(rank, flush=True)
    args.rank = rank  # have each process save the rank
    set_devices(args)  # args.device = rank or .device
    setup_process(args, rank, master_port=args.master_port, world_size=args.world_size)
    print(f'setup process done for rank={rank}')

    # create the (ddp) model, opt & scheduler
    get_and_create_model_opt_scheduler_first_time(args)
    print_dist(f"{args.model=}\n{args.opt=}\n{args.scheduler=}", args.rank)

    # create the dataloaders
    args.dataloaders: dict = get_sl_dataloader(args)

    # Agent does everything, proving, training, evaluate etc.
    agent: Agent = ClassificationSLAgent(args, args.model)

    # -- Start Training Loop
    print_dist('====> about to start train loop', args.rank)
    if args.training_mode == 'iterations':
        train_agent_fixed_number_of_iterations(args, agent, args.dataloaders, args.opt, args.scheduler)
    elif args.training_mode == 'epochs':
        train_agent_fixed_number_of_epochs(args, agent, args.dataloaders, args.opt, args.scheduler)
    elif args.training_mode == 'fit_single_batch':
        train_agent_fit_single_batch(args, agent, args.dataloaders, args.opt, args.scheduler)
    elif args.training_mode == 'fit_until_convergence':
        train_agent_fit_until_convergence(args, agent, args.dataloaders, args.opt, args.scheduler)
    else:
        raise ValueError(f'Invalid training_mode value, got: {args.training_mode}')

    # -- Clean Up Distributed Processes
    print(f'\n----> about to cleanup worker with rank {rank}')
    cleanup(rank)
    print(f'clean up done successfully! {rank}')


# -- Run experiment

if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
