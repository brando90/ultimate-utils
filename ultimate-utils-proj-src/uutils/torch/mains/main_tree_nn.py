#!/home/miranda9/miniconda3/envs/automl-meta-learning/bin/python

"""
main file for train models with SL for prediticing hashes

python main_training_sl_predict_hash.py --no_validation --exp_id test1

"""
from pdb import set_trace as st
import os
import sys

from socket import gethostname

from datetime import datetime

# pprint(sys.path)
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from data_lib.dag.dag_dataloader import get_dataloader_from_dag_files, get_dataloader_from_dag_files_syn_sem, test_dag_syn_sem
from data_lib.dataloader_hash_dataset import get_distributed_dataloader_from_feat_file
from progressbar import ProgressBar
from uutils import save_opts, load_cluster_jobids_to, set_system_wide_force_flush
from uutils.torch import print_dict_of_dataloaders_dataset_types
from uutils.torch.distributed import setup_process, cleanup, set_sharing_strategy, move_to_ddp, find_free_port, \
    clean_end_with_sigsegv_hack, is_lead_worker, is_running_serially, \
    print_process_info, set_devices, is_running_parallel
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ML4Coq
from agents import HashPredictorAgent
from embeddings_zoo.tactic_predictor import get_tree_nn_tactic_predictor_hash_predictor, \
    get_tree_nn_tactic_predictor_hash_predictor_dag_dataset

import numpy as np
from numpy import random

from pathlib import Path


sys.stdout.flush()  # no delay in print statements

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument('--tac_grammar', type=str, default=Path('/ml4coq-proj-src/embeddings_zoo/tactic_decoders/tactics.ebnf').expanduser())

    # experimental setup
    parser.add_argument('--debug', action='store_true', help='if debug')
    parser.add_argument('--serial', action='store_true', help='if running serially')
    # parser.add_argument('--include_synthetic', action='store_true')
    parser.add_argument('--exp_id', type=str, default='exp id:')
    # - data set arguments
    # dag
    # we only need the path to dag_dat_prep since that contains the paths to the train, val, test splits
    parser.add_argument('--path2dataprep', type=str, default=Path("~/data/lasse_datasets_coq/split_dag_data_prep.pt").expanduser())
    parser.add_argument('--path2vocabs', type=str, default=Path('~/data/lasse_datasets_coq/dag_counters.pt').expanduser())
    parser.add_argument('--path2hash2idx', type=str, default=Path('~/data/lasse_datasets_coq/dag_hash2index.pt').expanduser())

    parser.add_argument('--mode', type=str, default='syntactic')
    parser.add_argument('--split', type=str, default='train', help='dont train on val or test. This flag is mainly '
                                                                   'for the options "train" and "test_debug".')

    parser.add_argument('--log_root', type=str, default=Path('~/data/logs/').expanduser())

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--resume_path', type=str, default='', help='the path for model checkpoint to resume')
    parser.add_argument('--no_validation', action='store_true', help='no validation is performed')
    # parser.add_argument('--save_model_epochs', type=int, default=10, help='the number of epochs between model savings')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of data_lib-loading threads (when running serially')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt_freq_in_hours', type=int, default=6)
    parser.add_argument('--filter', type=str)

    # term encoder
    parser.add_argument('--term_encoder_embedding_dim', type=int, default=256)

    # tactic label classifier
    # parser.add_argument('--size_limit', type=int, default=50)
    # parser.add_argument('--embedding_dim_decoder', type=int, default=256, help='dimension of the grammar embeddings')
    # parser.add_argument('--symbol_dim', type=int, default=256, help='dimension of the terminal/nonteaarminal symbol embeddings')
    # parser.add_argument('--hidden_dim', type=int, default=256, help='dimension of the LSTM controller')
    # parser.add_argument('--teacher_forcing', type=float, default=1.0)
    parser.add_argument('--criterion', type=str, help='loss criterion', default=nn.CrossEntropyLoss())

    # optimization
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--lr_reduce_patience', type=int, default=10)
    parser.add_argument('--lr_reduce_steps', default=3, type=int,
                        help='the number of steps before reducing the learning rate \
                        (only applicable when no_validation == True)')

    # debugging flags
    parser.add_argument('--train_set_length', type=int, default=236_436)

    opts = parser.parse_args()

    # to have checkpointing work every 6 hours.
    opts.start = time.time()
    opts.next_time_to_ckpt_in_hours = 0.0

    opts.validation = not opts.no_validation

    # distributed params
    opts.rank = -1  # should be written by each worker with their rank, if not we are running serially
    opts.master_port = find_free_port()

    # TODO IMPROVE THIS
    torch.manual_seed(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {opts.device}')

    load_cluster_jobids_to(opts)
    opts.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    opts.log_root = opts.log_root / f'logs_{opts.current_time}_jobid_{opts.jobid}'
    opts.log_root.mkdir(parents=True, exist_ok=True)
    opts.tb_dir = opts.log_root / 'tb'
    opts.tb_dir.mkdir(parents=True, exist_ok=True)

    # annealing learning rate is only applicable when data_lib set is train TODO figure out later
    if (not opts.no_validation) and (opts.lr_reduce_steps is not None):
        print('--lr_reduce_steps is applicable only when no_validation == True', 'ERROR')

    # get device name if possible
    try:
        opts.gpu_name = torch.cuda.get_device_name(0)
        print(f'\nopts.gpu_name = {opts.gpu_name}\n')
    except:
        opts.gpu_name = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opts.PID = str(os.getpid())
    if torch.cuda.is_available():
        opts.nccl = torch.cuda.nccl.version()
    return opts

def load_checkpoint(opts, optimizer, model):
    start_epoch = 0
    if opts.resume_path != '':  # if the path is not empty then there is a model checkpoint
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.resume, map_location='cpu')
        else:  # use GPU
            checkpoint = torch.load(opts.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['n_epoch'] + 1
        model.to(opts.device)
    if is_running_parallel(opts.rank):  # halt all processes until everyone has loaded checkpoint
        torch.distributed.barrier()
    return start_epoch

def main_distributed():
    """
    train tree_nn in parallel

    Note: end-to-end ddp example on mnist: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    :return:
    """
    # parse the options
    opts = parse_args()
    [print(f'{k,v}') for k, v in vars(opts).items()]

    # parallel train
    if torch.cuda.is_available():
        opts.world_size = torch.cuda.device_count()
        print(f"{torch.cuda.device_count()=}")
    elif opts.serial:
        opts.world_size = 1
        print('RUNNING SERIALLY')
    else:
        # opts.world_size = mp.cpu_count() - 1  # 1 process is main, the rest are (parallel) trainers
        opts.world_size = 4

    # spawn the distributed training code
    print(f'\n{opts.world_size=}')
    save_opts(opts)
    if opts.serial:
        print('RUNNING SERIALLY')
        train(rank=-1, opts=opts)
    else:
        print('\nABOUT TO SPAWN WORKERS')
        # mp.set_sharing_strategy('file_system')
        set_sharing_strategy()
        print('done setting sharing strategy...next mp.spawn')
        mp.spawn(fn=train, args=(opts,), nprocs=opts.world_size)
        print('done mp.spwan')

def train(rank, opts):
    print_process_info(rank, flush=True)
    # have each process save the rank
    opts.rank = rank
    set_devices(opts)  # basically opts.gpu = rank if not debugging/serially
    # setup up worker
    setup_process(opts, rank, master_port=opts.master_port, world_size=opts.world_size)
    print(f'setup process done for rank={rank}')

    # create the dataloaders
    # dataloaders = get_distributed_dataloader_from_feat_file(opts, rank, opts.world_size)
    # dataloaders = get_dataloader_from_dag_files(opts, opts.rank, opts.world_size)
    dataloaders = get_dataloader_from_dag_files_syn_sem(opts, opts.rank, opts.world_size, opts.mode)
    if is_lead_worker(rank):
        print_dict_of_dataloaders_dataset_types(dataloaders)
        # [print_dataloaders_info(opts, dataloaders, split) for split, _ in dataloaders.items()]

    # create the model (TE + Decoder)
    # tactic_predictor = get_tree_nn_tactic_predictor_hash_predictor(opts)
    tactic_predictor = get_tree_nn_tactic_predictor_hash_predictor_dag_dataset(opts)
    tactic_predictor = move_to_ddp(rank, opts, tactic_predictor)

    # TODO figure out dist checkpointing later
    # start_epoch = load_checkpoint(opts, optimizer, tactic_predictor)
    start_epoch = 0

    # create the optimizer
    # optimizer = torch.optim.SGD(tactic_predictor.parameters(), lr=opts.learning_rate, momentum=opts.momentum, weight_decay=opts.l2)
    optimizer = torch.optim.Adam(tactic_predictor.parameters(), lr=opts.learning_rate, weight_decay=opts.l2)

    # Agent does everything, proving, training, evaluate etc.
    agent = HashPredictorAgent(tactic_predictor, optimizer, dataloaders, opts)

    # decay/anneal learning rate wrt epochs
    if opts.no_validation:  # only train set
        # use a fix step scheduler, decays the learning rate of each parameter group by gamma every step_size epochs.
        # scheduler = StepLR(optimizer, step_size=opts.lr_reduce_steps, gamma=0.1)
        # for now if only train lets overfit to train_loss
        scheduler = ReduceLROnPlateau(optimizer, patience=opts.lr_reduce_patience, verbose=True)  # temporary
    else:  # decay learning rate when validation loss got stuck
        # scheduler that decays if stuck on a plateu
        scheduler = ReduceLROnPlateau(optimizer, patience=opts.lr_reduce_patience, verbose=True)

    # -- Start Training Loop
    agent.log('====> about to start train loop')
    # train_loop(opts, agent, scheduler, start_epoch)
    train_loop_test(opts, agent, scheduler, start_epoch)
    # agent.train_single_batch()

    # -- Clean Up Distributed Processes
    print(f'\n----> about to cleanup worker with rank {rank}')
    cleanup(rank)
    print(f'clean up done successfully! {rank}')

def train_loop(opts, agent, scheduler, start_epoch):
    agent.log('Starting training...')
    for n_epoch in range(start_epoch, start_epoch + opts.num_epochs):
        opts.num_epoch = n_epoch
        # training
        train_loss, train_acc = agent.train(n_epoch)
        agent.log_train_stats(n_epoch, train_loss, train_acc, val_iterations=0, val_ckpt=True)
        agent.save(n_epoch)

        # validation
        # if opts.validation:
        #     val_loss, val_acc = agent.valid(n_epoch)
        # anneal the learning rate
        # if opts.no_validation:
        #     # scheduler.step()  # fix step scheduler
        #     scheduler.step(train_loss)
        # else:  # use the validation loss to anneal
        #     scheduler.step(val_loss)
    agent.log('-------> done training!')

def train_loop_test(opts, agent, scheduler, start_epoch):
    set_system_wide_force_flush()
    agent.print_dataloaders_info(opts.split)
    agent.log('Starting training...')
    opts.it = 0
    opts.n_epoch = 0
    # train_loss, train_acc = agent.predict_tactic_hashes(prediction_mode='train')
    while True:
        train_loss, train_acc = agent.train_test_debug(opts.split)
        agent.save_every_x(opts.n_epoch)  # saves ckpt every 6 hours
        scheduler.step(train_loss)

        # if train_acc >= 1.0 and train_loss <= 0.01:
        if train_acc >= 1.0:
            agent.log_train_stats(opts.it, train_loss, train_acc, val_ckpt=True)
            agent.save(opts.n_epoch)
            opts.n_epoch += 1
            break  # halt once both the accuracy is high enough AND train loss is low enough
        opts.n_epoch += 1

    agent.log(f'-------> done training at {opts.n_epoch=} and {opts.it=}')

if __name__ == '__main__':
    import time
    # start script
    start = time.time()
    print(f'-----> device = {torch.device("cuda" if torch.cuda.is_available() else "cpu")}\n')
    main_distributed()
    duration_secs = time.time() - start
    print(f"time passed: hours:{duration_secs / (60 ** 2)}, minutes={duration_secs / 60}, seconds={duration_secs}")
    print(f'\n---> hostname: {gethostname()}')
    print('DONE!!!\a')
