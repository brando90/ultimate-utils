# import pydevd_pycharm
# pydevd_pycharm.settrace('9.59.196.91', port=22, stdoutToServer=True, stderrToServer=True)
import os
import sys

from socket import gethostname

from datetime import datetime

# pprint(sys.path)
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import uutils.torch

from torch.optim.lr_scheduler import ReduceLROnPlateau

import transformers
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule

import numpy as np
from numpy import random

from pathlib import Path

from uutils import save_args, load_cluster_jobids_to, set_system_wide_force_flush
from uutils.torch import print_dict_of_dataloaders_dataset_types, print_dataloaders_info
from uutils.torch.distributed import setup_process, cleanup, set_sharing_strategy, move_to_ddp, find_free_port, \
    clean_end_with_sigsegv_hack, is_lead_worker, is_running_serially, \
    print_process_info, set_devices

from data_pkg.data_preparation import get_dataloaders, get_simply_type_lambda_calc_dataloader_from_folder

from models.tree_gen_simple import get_tree_gen_simple

from agent import SynthAgent

# import radam

from pdb import set_trace as st

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # experimental setup
    parser.add_argument('--reproduce_10K', action='store_true', default=False ,
                        help='Unset this if you want to run'
                            'your own data set. This is not really meant'
                            'to be used if you are trying to reproduce '
                            'the simply type lambda cal experiments on the '
                            '10K dataset.')
    parser.add_argument('--debug', action='store_true', help='if debug')
    parser.add_argument('--serial', action='store_true', help='if running serially')

    parser.add_argument('--split', type=str, default='train', help=' train, val, test')
    parser.add_argument('--data_set_path', type=str, default='~/data/simply_type_lambda_calc/dataset10000/',
                        help='path to data set splits')

    parser.add_argument('--log_root', type=str, default=Path('~/data/logs/').expanduser())

    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--num_its', type=int, default=-1)
    parser.add_argument('--training_mode', type=str, default='iterations')
    parser.add_argument('--no_validation', action='store_true', help='no validation is performed')
    parser.add_argument('--save_model_epochs', type=int, default=10, help='the number of epochs between model savings')
    parser.add_argument('--num_workers', type=int, default=-1, help='the number of data_lib-loading threads (when running serially')

    # term encoder
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=1)

    # tactic label classifier
    parser.add_argument('--criterion', type=str, help='loss criterion', default=nn.CrossEntropyLoss())

    # optimization
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_warmup_steps', type=int, default=-1)
    # parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--l2', type=float, default=0.0)
    # parser.add_argument('--lr_reduce_steps', default=3, type=int,
    #                     help='the number of steps before reducing the learning rate \
    #                     (only applicable when no_validation == True)')

    parser.add_argument('--lr_reduce_patience', type=int, default=10)

    # parser.add_argument('--resume', type=str, help='the model checkpoint to resume')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--always_use_deterministic_algorithms', action='store_true', help='tries to make pytorch fully determinsitic')

    args = parser.parse_args()
    args.validation = not args.no_validation

    # distributed params
    args.rank = -1  # should be written by each worker with their rank, if not we are running serially
    args.master_port = find_free_port()

    # determinism
    print('----- TRYING TO MAKE CODE DETERMINISTIC')
    # uutils.torch.make_code_deterministic(args.seed)

    print(f'{args.seed=}')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {args.device}')

    load_cluster_jobids_to(args)
    args.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.log_root = args.log_root / f'logs_{args.current_time}_jobid_{args.jobid}'
    args.log_root.mkdir(parents=True, exist_ok=True)
    args.tb_dir = args.log_root / 'tb'
    args.tb_dir.mkdir(parents=True, exist_ok=True)

    # annealing learning rate is only applicable when data_lib set is train TODO figure out later
    # if (not args.no_validation) and (args.lr_reduce_steps is not None):
    #     print('--lr_reduce_steps is applicable only when no_validation == True', 'ERROR')

    # get device name if possible
    try:
        args.gpu_name = torch.cuda.get_device_name(0)
        print(f'\nargs.gpu_name = {args.gpu_name}\n')
    except:
        args.gpu_name = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.PID = str(os.getpid())
    if torch.cuda.is_available():
        args.nccl = torch.cuda.nccl.version()
    return args

def load_checkpoint(args, optimizer, model):
    # todo - decide a way to make this standard in the way I do things and put in my standard utils as always
    start_epoch = 0
    if args.resume != None:
        # log(f'loading model checkpoint from {args.resume}')
        if args.device.type == 'cpu':
            checkpoint = torch.load(args.resume, map_location='cpu')
        else:  # use GPU
            checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch_num'] + 1
        model.to(args.device)
    return start_epoch

def main_distributed():
    """
    train tree_nn in parallel
    Note: end-to-end ddp example on mnist: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    :return:
    """
    # parse the options
    args = parse_args()
    [print(f'{k,v}') for k, v in vars(args).items()]

    # parallel train
    if torch.cuda.is_available():
        args.world_size = torch.cuda.device_count()
        print(f"{torch.cuda.device_count()=}")
    elif args.serial:
        args.world_size = 1
        print('RUNNING SERIALLY')
    else:
        # args.world_size = mp.cpu_count() - 1  # 1 process is main, the rest are (parallel) trainers
        args.world_size = 4

    # spawn the distributed training code
    print(f'\n{args.world_size=}')

    if args.serial:
        print('RUNNING SERIALLY')
        train(rank=-1, args=args)
    else:
        print('\nABOUT TO SPAWN WORKERS')
        set_sharing_strategy()
        mp.spawn(fn=train, args=(args,), nprocs=args.world_size)

def train(rank, args):
    print_process_info(rank, flush=True)
    args.rank = rank  # have each process save the rank
    set_devices(args)  # basically args.gpu = rank if not debugging/serially
    setup_process(args, rank, master_port=args.master_port, world_size=args.world_size)
    print(f'setup process done for rank={rank}')

    # create the dataloaders
    dataloaders = get_simply_type_lambda_calc_dataloader_from_folder(args)

    # create the model
    mdl = get_tree_gen_simple(args, dataloaders)
    mdl = move_to_ddp(rank, args, mdl)

    # start_epoch = load_checkpoint(args, optimizer, tactic_predictor)
    start_epoch = 0
    start_it = 0

    # create the optimizer
    optimizer = Adafactor(mdl.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    # optimizer = torch.optim.Adam(mdl.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    # optimizer = radam.RAdam(mdl.parameters(), lr=args.learning_rate)
    print(f'{optimizer=}')

    # decay/anneal learning rate wrt epochs
    scheduler = None
    # scheduler = ReduceLROnPlateau(optimizer, patience=args.lr_reduce_patience, verbose=True)  # temporary
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=False)
    scheduler = AdafactorSchedule(optimizer)
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps)
    print(f'{scheduler=}')

    # Agent does everything, proving, training, evaluate etc.
    agent = SynthAgent(args, mdl, optimizer, dataloaders, scheduler)

    # - save args
    if agent.is_lead_worker():
        save_args(args)

    # -- Start Training Loop
    agent.log('====> about to start train loop')
    if not args.reproduce_10K:  # real experiment
        agent.log('-- running experiment')
        # agent.main_train_loop_based_on_fixed_number_of_epochs(args, start_epoch)
        # agent.main_train_loop_until_convergence(args, start_it)
        agent.main_train_loop_based_on_fixed_iterations(args, start_it)
        # agent.train_single_batch()
    else:  # reproduction
        agent.log('-- running reproduction')
        optimizer = Adafactor(mdl.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        scheduler = AdafactorSchedule(optimizer)
        agent = SynthAgent(args, mdl, optimizer, dataloaders, scheduler)
        agent.main_train_loop_until_convergence(args, start_it)

    # -- Clean Up Distributed Processes
    print(f'\n----> about to cleanup worker with rank {rank}')
    cleanup(rank)
    print(f'clean up done successfully! {rank}')
    print(f'{args.batch_size=}')
    print(f'{args.embed_dim=}')
    print(f'{args.num_layers=}')

if __name__ == '__main__':
    import time
    start = time.time()
    print(f'\n{gethostname()=}')
    print(f'{torch.cuda.device_count()=},{torch.cuda.is_available()=}')
    print(f'-----> device = {torch.device("cuda" if torch.cuda.is_available() else "cpu")}\n')
    main_distributed()
    duration_secs = time.time() - start
    print(f"Success, time passed: hours:{duration_secs / (60 ** 2)}, minutes={duration_secs / 60}, seconds={duration_secs}")
