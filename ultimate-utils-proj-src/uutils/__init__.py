"""
Utils class with useful helper functions

utils: https://www.quora.com/What-do-utils-files-tend-to-be-in-computer-programming-documentation
"""
import json
import pickle
import subprocess
import time

import math
from datetime import datetime
from pprint import pprint

import dill
import networkx as nx
import numpy as np
import random
import pandas as pd

import time
import logging
import argparse

import os
import shutil
import sys

import pathlib
from pathlib import Path

from pdb import set_trace as st

import types

from pandas import DataFrame

from socket import gethostname

from lark import Lark, tree, Tree, Token

from collections import deque

from argparse import Namespace

from typing import Union, Any, Optional, Match

import progressbar

from uutils.logging_uu.wandb_logging.common import setup_wand
from uutils.torch_uu.distributed import find_free_port


def hello():
    import uutils
    print(f'\nhello from uutils __init__.py in:\n{uutils}\n')


def print_pids():
    import torch.multiprocessing as mp

    print('running main()')
    print(f'current process: {mp.current_process()}')
    print(f'pid: {os.getpid()}')


# - getting args for expts

def setup_args_for_experiment(args: Namespace,
                              num_workers: int = 0,
                              use_tb: bool = False  # not needed anymore since wandb exists
                              ) -> Namespace:
    """
    Sets up programming details for experiment to run but it should not affect the machine learning results.
    The pretty print & save the args in alphabetically sorted order.

    Note:
        - This function assume the arguments for the experiment have been given already (e.g. through the terminal
        or manually hardcoded in the main/run script).

    Example/Recommended pattern to use:
    1. From terminal (e.g. by running a bash main.sh script that runs python main_experiment.py):
        args = parse_basic_meta_learning_args_from_terminal()
        args = uutils.setup_args_for_experiment(args)
        main(args)
        wandb.finish() if is_lead_worker(args.rank) and args.log_to_wandb else None
        print("Done with experiment, success!\a")

    2. From script (also called, manual or hardcoded)
        args = parse_basic_meta_learning_args_from_terminal()
        args = manual_values_for_run(args)  # e.g. args.batch_size = 64, or ckpt
        args = uutils.setup_args_for_experiment(args)
        main(args)
        wandb.finish() if is_lead_worker(args.rank) and args.log_to_wandb else None
        print("Done with experiment, success!\a")

    Things it sets up:
        - 1. makes sure it has intial index (or use the one provided by user/checkpoint
        - gets rank, port for DDP
        - sets determinism or not
        - device name
        - dirs & filenames for logging
        - cluster job id stuff
        - pid, githash
        - best_val_loss as -infinity
        - wandb be stuff based on your options
    todo:
        - decide where to put annealing, likely in the parse args from terminal and set the default there. Likely
        no annealing as default, or some rate that is done 5 times during entire training time or something like
        that is based on something that tends to work.
    """
    import torch
    import logging
    import uutils
    from uutils.logger import Logger as uuLogger

    # - 0. to make sure epochs or iterations is explicit, set it up in the argparse arguments
    assert args.training_mode in ['epochs', 'iterations']
    # - 1. set the iteration/epoch number to start training from
    if args.training_mode == 'iterations':
        # set the training iteration to start from beginning or from specified value (e.g. from ckpt iteration index).
        args.it = 0 if not hasattr(args, 'it') else args.it
        assert args.it >= 0, f'Iteration to train has to be start at zero or above but got: {args.it}'
        args.epoch_num = -1
    elif args.training_mode == 'epochs':
        # set the training epoch number to start from beginning or from specified value e.g. from ckpt epoch_num index.
        args.epoch_num = 0 if not hasattr(args, 'epoch_num') else args.epoch_num
        assert args.epoch_num >= 0, f'Epoch number to train has to be start at zero or above but got: {args.epoch_num}'
        args.it = -1
    else:
        raise ValueError(f'Invalid training mode: {args.training_mode}')
    # - annealing learning rate...
    # if (not args.no_validation) and (args.lr_reduce_steps is not None):
    #     print('--lr_reduce_steps is applicable only when no_validation == True', 'ERROR')

    # NOTE: this should be done outside cuz flags have to be declared first then parsed, args = parser.parse_args()
    if hasattr(args, 'no_validation'):
        args.validation = not args.no_validation

    # - distributed params
    args.rank = -1  # should be written by each worker with their rank, if not we are running serially
    args.master_port = find_free_port()

    # - determinism
    if hasattr(args, 'always_use_deterministic_algorithms'):
        if args.always_use_deterministic_algorithms:
            uutils.torch_uu.make_code_deterministic(args.seed)
            logging.warning(f'Seed being ignored, seed value: {args.seed=}')

    # - get device name
    print(f'{args.seed=}')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {args.device}')

    # - get cluster info (including hostname)
    load_cluster_jobids_to(args)

    # - get log_root
    # usually in options: parser.add_argument('--log_root', type=str, default=Path('~/data/logs/').expanduser())
    args.log_root: Path = Path('~/data/logs/').expanduser() if not hasattr(args, 'log_root') else args.log_root
    args.log_root: Path = Path(args.log_root).expanduser() if isinstance(args.log_root, str) else args.log_root
    args.log_root: Path = args.log_root.expanduser()
    args.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.log_root = args.log_root / f'logs_{args.current_time}_jobid_{args.jobid}'
    args.log_root.mkdir(parents=True, exist_ok=True)
    # create tb in log_root
    if use_tb:
        from torch.utils.tensorboard import SummaryWriter
        args.tb_dir = args.log_root / 'tb'
        args.tb_dir.mkdir(parents=True, exist_ok=True)
        args.tb = SummaryWriter(log_dir=args.tb_dir)

    # - setup expanded path to checkpoint
    if hasattr(args, 'path_to_checkpoint'):
        # str -> Path(path_to_checkpoint).expand()
        if isinstance(args.path_to_checkpoint, str):
            args.path_to_checkpoint = Path(args.path_to_checkpoint).expanduser()
        elif isinstance(args.path_to_checkpoint, Path):
            args.path_to_checkpoint.expanduser()
        elif args.path_to_checkpoint is None:  # do nothing since the args defualt is None
            pass
        else:
            raise ValueError(f'Path to checkpoint is not of the right type: {type(args.path_to_checkpoint)=},'
                             f'with value: {args.path_to_checkpoint=}')

    # - get device name if possible
    try:
        args.gpu_name = torch.cuda.get_device_name(0)
    except:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.gpu_name = args.device
    print(f'\nargs.gpu_name = {args.gpu_name}\n')  # print gpu_name if available else cpu

    # - save PID
    args.PID = str(os.getpid())
    if torch.cuda.is_available():
        args.nccl = torch.cuda.nccl.version()

    # - email option (warning, not recommended! very unreliable, especially from cluster to cluster)
    # logging.warning('uutils email is not recommended! very unreliable, especially from cluster to cluster)
    # args.mail_user = 'brando.science@gmail.com'
    # args.pw_path = Path('~/pw_app.config.json').expanduser()

    # - try to get githash, might fail and return error string in field but whatever
    args.githash: str = try_to_get_git_revision_hash_short()
    args.githash_short: str = try_to_get_git_revision_hash_short()
    args.githash_long: str = try_to_get_git_revision_hash_long()

    # - get my logger, its set at the agent level
    args.logger: uuLogger = uutils.logger.Logger(args)

    # - best val loss
    args.best_val_loss: float = float('inf')

    # - wandb
    if hasattr(args, 'log_to_wandb'):
        setup_wand(args)
    else:
        pass

    # - for debugging
    # args.environ = [str(f'{env_var_name}={env_valaue}, ') for env_var_name, env_valaue in os.environ.items()]

    # - to avoid pytorch multiprocessing issues with CUDA: https://pytorch.org/docs/stable/data.html
    args.pin_memory = False
    args.num_workers = num_workers

    # - return
    uutils.print_args(args)
    uutils.save_args(args)
    return args


def _parse_basic_meta_learning_args_from_terminal() -> Namespace:
    """
    Parse the arguments from the terminal so that the experiment runs as the user specified there.
    Example/Recommended pattern to use:
        See setup_args_for_experiment(...) to avoid copy pasting example/only mantain it in one place.
    Note:
        - Strongly recommended to see setup_args_for_experiment(...)
    """
    import argparse
    # import torch.nn as nn

    parser = argparse.ArgumentParser()

    # experimental setup
    parser.add_argument('--debug', action='store_true', help='if debug')
    parser.add_argument('--force_log', action='store_true', help='to force logging')
    parser.add_argument('--serial', action='store_true', help='if running serially')
    parser.add_argument('--args_hardcoded_in_script', action='store_true',
                        help='set to true if the args will be set from the script manually'
                             'e.g. by hardcoding them in the script.')

    parser.add_argument('--split', type=str, default='train', help=' train, val, test')
    # this is the name used in synth agent, parser.add_argument('--data_set_path', type=str, default='', help='path to data set splits')
    parser.add_argument('--data_path', type=str, default='VALUE SET IN MAIN Meta-L SCRIPT',
                        help='path to data set splits')

    parser.add_argument('--log_root', type=str, default=Path('~/data/logs/').expanduser())

    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--num_its', type=int, default=-1)
    parser.add_argument('--training_mode', type=str, default='iterations')
    # parser.add_argument('--no_validation', action='store_true', help='no validation is performed')
    # parser.add_argument('--embed_dim', type=int, default=256)
    # parser.add_argument('--nhead', type=int, default=8)
    # parser.add_argument('--num_layers', type=int, default=1)
    # parser.add_argument('--criterion', type=str, help='loss criterion', default=nn.CrossEntropyLoss())

    # - optimization
    # parser.add_argument('--optimizer', type=str, default='Adam')
    # parser.add_argument('--learning_rate', type=float, default=1e-5)
    # parser.add_argument('--num_warmup_steps', type=int, default=-1)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--l2', type=float, default=0.0)
    # parser.add_argument('--lr_reduce_steps', default=3, type=int,
    #                     help='the number of steps before reducing the learning rate \
    #                     (only applicable when no_validation == True)')
    # parser.add_argument('--lr_reduce_patience', type=int, default=10)

    # - miscellaneous
    parser.add_argument('--path_to_checkpoint', type=str, default=None,
                        help='the model checkpoint path to resume from.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--always_use_deterministic_algorithms', action='store_true',
                        help='tries to make pytorch fully determinsitic')
    # for now default is 4 since meta-learning code is not parallizable right now.
    parser.add_argument('--num_workers', type=int, default=4,
                        help='the number of data_lib-loading threads (when running serially')

    # - sims/dists computations
    parser.add_argument('--sim_compute_parallel', action='store_true', help='compute sim or dist in parallel.')
    parser.add_argument('--metrics_as_dist', action='store_true', help='')
    parser.add_argument('--show_layerwise_sims', action='store_true', help='show sim/dist values per layer too')

    # - wandb
    parser.add_argument('--log_to_wandb', action='store_true', help='store to weights and biases')
    parser.add_argument('--wandb_project', type=str, default='meta-learning-playground')
    parser.add_argument('--wandb_entity', type=str, default='brando')
    parser.add_argument('--wandb_group', type=str, default='experiment_debug', help='helps grouping experiment runs')
    # parser.add_argument('--wandb_log_freq', type=int, default=10)
    # parser.add_argument('--wandb_ckpt_freq', type=int, default=100)
    # parser.add_argument('--wanbd_mdl_watch_log_freq', type=int, default=-1)

    # - parse arguments
    args = parser.parse_args()
    # - load cluster ids so that wandb can use it later for naming runs, experiments, etc.
    load_cluster_jobids_to(args)
    return args


def parse_args() -> Namespace:
    """
        Parses command line arguments
    """
    parser = argparse.ArgumentParser(description="Pretraining Experiment")
    parser.add_argument(
        "data", help="path to dataset", metavar="DIR",
    )
    parser.add_argument(
        "--exp-dir", type=str, help="directory for the experiment results"
    )
    parser.add_argument(
        "--num-classes", type=int, help="number of classes in dataset", metavar="N"
    )
    parser.add_argument(
        "--seed", type=int, help="seed for deterministic experimenting", default=61820
    )
    parser.add_argument(
        "--epochs", type=int, help="number of epochs of training", metavar="E"
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        help="learning rate of the optimizer",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", type=float, help="momentum of the optimizer", metavar="M"
    )
    parser.add_argument(
        "--weight-decay", type=float, help="weight decay of the optimizer", metavar="WD"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="size of batch for training and validation",
        metavar="B",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        help="url used to setup distributed training",
        default="localhost:61820",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="number of GPUs (processes) used for model training",
        default=1,
        dest="num_gpus",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        help="frequency of model, dataloader, optimizer state",
        default=1,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="path to checkpoint to be loaded. If input is empty, then the model trains from scratch.",
        default="",
    )

    return parser.parse_args()


def args_hardcoded_in_script(args: Namespace) -> bool:
    """
    Return true if we are using (hardcoded) manual args.

    Npte: We detect if to use manual args if manual_loads_name is not None.
    None means, do not use manual args.
    """
    # - if manual_loads_name is not given then params aren't manually given
    if not hasattr(args, 'manual_loads_name'):
        return False
    else:
        # - args are hardcoded if name is not None
        args_hardcoded: bool = args.manual_loads_name != 'None'
        return args_hardcoded


def get_args_from_checkpoint_pickle_file(args: Namespace) -> Namespace:
    """
    Get the args from the checkpoint when the args is inside the pickle file saved by the torch.save, dill, etc.
    """
    import torch
    ckpt: dict = torch.load(args.path_2_init_sl, map_location=torch.device('cpu'))
    return ckpt['args']


def get_args_from_checkpoint_json_file(path: str, filename: str, mode='r') -> Namespace:
    """
    Load the arguments from a checkpoint when the args is saved in a json file e.g. filename = args.json.
    """
    args: dict = load_json(path=path, filename=filename, mode=mode)
    args: Namespace = dict2namespace(args)
    return args


def make_args_from_metalearning_checkpoint(args: Namespace,
                                           path2args: str,
                                           filename: str = "args.json",
                                           precedence_to_args_checkpoint: bool = True,
                                           it: Optional[int] = 0,
                                           ) -> Namespace:
    """
    Get args form metalearning and merge it with current args. Default precedence is to the checkpoint
    since you want to keep training from the checkpoint value.

    Note:
        - you could get the it or epoch_num from the ckpt pickle file here too and set it in args but I decided
        to not do this here and do it when processing the actualy pickle file to seperate what the code is doing
        and make it less confusing (in one place you process the args.json and the other you process the pickled
        file with the actual model, opt, meta_learner etc).
    """
    path2args: Path = Path(path2args).expanduser() if isinstance(path2args, str) else path2args.expanduser()
    args_ckpt: Namespace = get_args_from_checkpoint_json_file(path=path2args, filename=filename)
    args_ckpt: Namespace = map_args_fields_from_string_to_usable_value(args_ckpt)
    # args.num_its = args.train_iters
    # - for forward compatibility, but ideally getting the args and the checkpoint will be all in one place in the future
    args.training_mode = 'iterations'
    args.it = it
    # - updater args has precedence
    if precedence_to_args_checkpoint:
        args: Namespace = merge_args(starting_args=args, updater_args=args_ckpt)
    else:
        args: Namespace = merge_args(starting_args=args_ckpt, updater_args=args)
    # - always overwrite the path to the checkpoint with the one given by the user
    # (since relitive paths aren't saved properly since they are usually saved as expanded paths)
    args.path_to_checkpoint: Path = path2args
    args.log_root: Path = Path('~/data/logs/')
    return args


def map_args_fields_from_string_to_usable_value(args_ckpt: Namespace) -> Namespace:
    """
    Since in some previous code the args where saved as a .json file with the field values as strings, you
    need to convert them to their actualy usable values for experiments e.g. string int to actual int.

    Note:
        - future checkpointing make sure to make args a pickable object and save it in the pickle checkpoint.
    """
    new_args: Namespace = Namespace()
    dict_args: dict = vars(args_ckpt)
    for field, value in dict_args.items():
        # if not isinstance(field, str):
        #     value_processed = field
        if value == 'True':
            value_processed: bool = True
        elif value == 'False':
            value_processed: bool = False
        elif value == 'None':
            value_processed = None
        elif value.isnumeric():  # non-negative int
            value_processed: int = int(value)
        elif is_negative_int(value):
            value_processed: int = int(value)
        elif is_float(value):
            value_processed: float = float(value)
        else:
            value_processed: str = str(value)
        setattr(new_args, field, value_processed)
    return new_args


# -

def try_to_get_git_revision_hash_short():
    """ ref: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script """
    try:
        git_hash: str = str(_get_git_revision_short_hash())
        return git_hash
    except Exception as e:
        print(f'(Not critical), unable to retrieve githash for reason: {e}')
        return f'{e}'


def try_to_get_git_revision_hash_long():
    """ ref: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script """
    try:
        git_hash: str = str(_get_git_revision_hash())
        return git_hash
    except Exception as e:
        print(f'(Not critical), unable to retrieve githash for reason: {e}')
        return f'{e}'


def _get_git_revision_hash():
    """ ref: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def _get_git_revision_short_hash():
    """ ref: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script """
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def cat_file(path2filename: Union[str, Path]):
    """prints/displays file contents. Do path / filename or the like outside of this function. ~ is alright to use. """
    expanduser(path2filename)
    with open(path2filename, 'r') as f:
        print(f.read())


def get_good_progressbar(max_value: Union[int, progressbar.UnknownLength, None] = None) -> progressbar.ProgressBar:
    """
    Example output:

    100% (100 of 100) |#####| Elapsed Time: 0:00:10 |  Time:  0:00:10 |    9.8 it/

    For unknown length e.g:
        bar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
        for i in range(20):
            time.sleep(0.1)
            bar.update(i)

    reference:
        - https://progressbar-2.readthedocs.io/en/latest/
        - https://github.com/WoLpH/python-progressbar/discussions/253
        - https://stackoverflow.com/questions/30834730/how-to-print-iterations-per-second
        - https://github.com/tqdm/tqdm/discussions/1211
    :rtype: object
    :return:
    """
    widgets = [
        progressbar.Percentage(),
        ' ', progressbar.SimpleProgress(format=f'({progressbar.SimpleProgress.DEFAULT_FORMAT})'),
        ' ', progressbar.Bar(),
        ' ', progressbar.Timer(), ' |',
        ' ', progressbar.ETA(), ' |',
        ' ', progressbar.AdaptiveTransferSpeed(unit='it'),
    ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_value)
    return bar


def get_good_progressbar_tdqm():
    # https://github.com/tqdm/tqdm/discussions/1211
    pass


def get_logger(name, log_path, log_filename, rank=0):
    """
        Initializes and returns a standard logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if rank != 0:
        return logger
    # Setup file & console handler
    file_handler = logging.FileHandler(os.path.join(log_path, log_filename + ".log"))
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Attach handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def remove_folders_recursively(path):
    print("WARNING: HAS NOT BEEN TESTED")
    path.expanduser()
    try:
        shutil.rmtree(str(path))
    except OSError:
        # do nothing if removing fails
        pass


def oslist_for_path(path):
    return [f for f in path.iterdir() if f.is_dir()]


def _make_and_check_dir(path):
    """
    NOT NEEDED use:

    mkdir(parents=True, exist_ok=True) see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
    """
    path = os.path.expanduser(path)
    path.mkdir(
        parents=True, exit_ok=True
    )  # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.


def create_folder(path2folder) -> None:
    """
    Creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.

    :param path2folder:
    :return:
    """
    if isinstance(path2folder, str):
        path2folder = Path(path2folder).expanduser()
    assert isinstance(path2folder, Path)
    path2folder.mkdir(parents=True, exist_ok=True)


def timeSince(start):
    """
    How much time has passed since the time "start"

    :param float start: the number representing start (usually time.time())
    """
    now = time.time()
    s = now - start
    ## compute how long it took in hours
    h = s / 3600
    ## compute numbers only for displaying how long it took
    m = math.floor(s / 60)  # compute amount of whole integer minutes it took
    s -= m * 60  # compute how much time remaining time was spent in seconds
    ##
    msg = f"time passed: hours:{h}, minutes={m}, seconds={s}"
    return msg, h


def report_times(start: float) -> str:
    import time
    duration_secs = time.time() - start
    msg = f"time passed: hours:{duration_secs / (60 ** 2)}, minutes={duration_secs / 60}, seconds={duration_secs}"
    return msg


def is_NaN(value):
    """
    Checks is value is problematic by checking if the value:
    is not finite, is infinite or is already NaN
    """
    return not np.isfinite(value) or np.isinf(value) or np.isnan(value)


def make_args_pickable(args: Namespace) -> Namespace:
    """
    Returns a copy of the args namespace but with unpickable objects as strings.

    note: implementation not tested against deep copying.
    ref:
        - https://stackoverflow.com/questions/70128335/what-is-the-proper-way-to-make-an-object-with-unpickable-fields-pickable
        - pycharm halting all the time issues: https://stackoverflow.com/questions/70761481/how-to-stop-pycharms-break-stop-halt-feature-on-handled-exceptions-i-e-only-b
        - stop progressbar from printing progress when checking if it's pickable: https://stackoverflow.com/questions/70762899/how-does-one-stop-progressbar-from-printing-eta-progress-when-checking-if-the
    """
    pickable_args = argparse.Namespace()
    # - go through fields in args, if they are not pickable make it a string else leave as it
    # The vars() function returns the __dict__ attribute of the given object.
    for field in vars(args):
        # print(f'-----{field}')
        field_val: Any = getattr(args, field)
        # - if current field value is not pickable, make it pickable by casting to string
        from uutils.logger import Logger
        # if not dill.pickles(field_val):
        #     field_val: str = str(field_val)

        # - remove bellow once is_picklable works on pycharm
        if callable(field_val):
            field_val: str = str(field_val)
        elif isinstance(field_val, Logger):
            field_val: str = str(field_val)
        elif field == 'scheduler':
            field_val: str = str(field_val)
        elif field == 'dataloaders':
            field_val: str = str(field_val)
        elif field == 'model':
            field_val: str = str(field_val)
        elif field == 'bar':
            field_val: str = str(field_val)
        # this is at the end so that progressbar ETA print happens only in an emergency,
        # but the right way to fix this is to have pycharm only halt on unhandled exceptions
        elif not dill.pickles(field_val):
            field_val: str = str(field_val)

        # -
        elif not is_picklable(field_val):
            field_val: str = str(field_val)
        # - after this line the invariant is that it should be pickable, so set it in the new args obj
        setattr(pickable_args, field, field_val)
        # print('f-----')
    return pickable_args


def make_opts_pickable(opts):
    """ Makes a namespace pickable """
    return make_args_pickable(opts)


def is_picklable(obj: Any) -> bool:
    """
    Checks if somehting is pickable.

    Ref:
        - https://stackoverflow.com/questions/70128335/what-is-the-proper-way-to-make-an-object-with-unpickable-fields-pickable
        - pycharm halting all the time issue: https://stackoverflow.com/questions/70761481/how-to-stop-pycharms-break-stop-halt-feature-on-handled-exceptions-i-e-only-b
    """
    import pickle
    try:
        pickle.dumps(obj)
    except:
        return False
    return True


def xor(a: Any, b: Any) -> bool:
    """
    Returns xor of a and b. Only one can be true but not both.

    ref: https://stackoverflow.com/a/432948/1601580
    """
    assert (True + True + True + False) == 3, 'Semantics of python changed'  # guard against change semantics of python.
    xor_bool: bool = (bool(a) + bool(b) == 1)
    return xor_bool


##

def make_and_check_dir2(path):
    """
        tries to make dir/file, if it exists already does nothing else creates it.
    """
    try:
        os.makedirs(path)
    except OSError:
        pass


####

"""
Greater than 4 I get this error:

ValueError: Seed must be between 0 and 2**32 - 1
"""


def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system. 
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed


def seed_everything(seed=42):
    """
    https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch
    """
    import torch
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_hostname_mit():
    from socket import gethostname

    hostname = gethostname()
    if "polestar-old" in hostname or hostname == "gpu-16" or hostname == "gpu-17":
        return "polestar-old"
    elif "openmind" in hostname:
        return "OM"
    else:
        return hostname


def make_dirpath_current_datetime_hostname(path=None, comment=""):
    """Creates directory name for tensorboard experiments.

    Keyword Arguments:
        path {str} -- [path to the runs directory] (default: {None})
        comment {str} -- [comment to add at the end of the file of the experiment] (default: {''})

    Returns:
        [PosixPath] -- [nice object interface for manipulating paths easily]
    """
    # makedirpath with current date time and hostname
    import socket
    import os
    from datetime import datetime

    # check if path is a PosixPath object
    if type(path) != pathlib.PosixPath and path is not None:
        path = Path(path)
    # get current time
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    # make runs logdir path
    runs_log_dir = os.path.join(
        "runs", current_time + "_" + socket.gethostname() + comment
    )
    # append path to front of runs_log_dir
    log_dir = Path(runs_log_dir)
    if path is not None:
        log_dir = path / runs_log_dir
    return log_dir


def host_local_machine(local_hosts=None):
    """ Returns True if its a recognized local host
    
    Keyword Arguments:
        local_hosts {list str} -- List of namaes of local hosts (default: {None})
    
    Returns:
        [bool] -- True if its a recognized local host False otherwise.
    """
    from socket import gethostname

    if local_hosts is None:
        local_hosts = [
            "Sarahs-iMac.local",
            "Brandos-MacBook-Pro.local",
            "Beatrizs-iMac.local",
            "C02YQ0GQLVCJ",
        ]
    hostname = gethostname()
    if hostname in local_hosts:
        return True
    else:  # not known local host
        return False


def my_print(*args, filepath="~/my_stdout.txt"):
    """Modified print statement that prints to terminal/scree AND to a given file (or default).

    Note: import it as follows:

    from utils.utils import my_print as print

    to overwrite builtin print function

    Keyword Arguments:
        filepath {str} -- where to save contents of printing (default: {'~/my_stdout.txt'})
    """
    # https://stackoverflow.com/questions/61084916/how-does-one-make-an-already-opened-file-readable-e-g-sys-stdout
    import sys
    from builtins import print as builtin_print
    filepath = Path(filepath).expanduser()
    # do normal print
    builtin_print(*args, file=sys.__stdout__)  # prints to terminal
    # open my stdout file in update mode
    with open(filepath, "a+") as f:
        # save the content we are trying to print
        builtin_print(*args, file=f)  # saves to file


def collect_content_from_file(filepath):
    filepath = Path(filepath).expanduser()
    contents = ""
    with open(filepath, "r") as f:
        for line in f.readlines():
            contents = contents + line
    return contents


# -

def unique_name_from_str(string: str, last_idx: int = 12) -> str:
    """
    Generates a unique id name
    refs:
    - md5: https://stackoverflow.com/questions/22974499/generate-id-from-string-in-python
    - sha3: https://stackoverflow.com/questions/47601592/safest-way-to-generate-a-unique-hash
    (- guid/uiid: https://stackoverflow.com/questions/534839/how-to-create-a-guid-uuid-in-python?noredirect=1&lq=1)
    """
    import hashlib
    m = hashlib.md5()
    string = string.encode('utf-8')
    m.update(string)
    unqiue_name: str = str(int(m.hexdigest(), 16))[0:last_idx]
    return unqiue_name


# - cluster stuff

def print_args(args: Namespace, sort_keys: bool = True):
    """
    Note:
        - compared with pprint_any_dict this does not convert values into strings and does not create a
        dictionary.
            - If you want a dictionary with the true values pprint(vars(args))

    ref:
        - https://stackoverflow.com/questions/24728933/sort-dictionary-alphabetically-when-the-key-is-a-string-name
    """
    assert isinstance(args, Namespace), f'Error: args has to be of type Namespace but got {type(args)}'
    dict_args: dict = vars(args)
    if sort_keys:
        sorted_names: list = sorted(dict_args.keys(), key=lambda x: x)
    [print(f'{k, dict_args[k]}') for k in sorted_names]
    # pprint_any_dict(dict_args)
    # pprint(vars(args))


def pprint_args(args: Namespace):
    print_args(args)


def pprint_dict(dic):
    pprint_any_dict(dic)


def pprint_any_dict(dic: dict, indent: Optional[int] = None):
    """
    This pretty prints a json.

    Note: this is not the same as pprint.

    Warning:
        - if indent is an int then the values will become strings.

    todo: how to have pprint do indent and keep value without making it into a string.
    """
    import json

    if indent:
        # make all keys strings recursively with their naitve str function
        dic = to_json(dic)
        print(json.dumps(dic, indent=4, sort_keys=True))  # only this one works...idk why
    else:
        pprint(dict)
    # return pretty_dic


def pprint_namespace(ns):
    """ pretty prints a namespace """
    pprint_any_dict(ns)


def _to_json_dict_with_strings(dictionary) -> dict:
    """
    Convert dict to dict with leafs only being strings. So it recursively makes keys to strings
    if they are not dictionaries.

    Use case:
        - saving dictionary of tensors (convert the tensors to strins!)
        - saving arguments from script (e.g. argparse) for it to be pretty
    """
    # base case: if the input is not a dict make it into a string and return it
    # if type(dictionary) != dict or type(dictionary) != list:
    if type(dictionary) != dict:
        return str(dictionary)
    # recurse into all the values that are dictionaries
    d = {k: _to_json_dict_with_strings(v) for k, v in dictionary.items()}
    return d


def to_json(dic) -> dict:
    if type(dic) is dict:
        dic = dict(dic)
    else:
        dic = dic.__dict__
    return _to_json_dict_with_strings(dic)


def save_to_json_pretty(data: Any, path2filename: Union[str, Path], mode='w', indent=4, sort_keys=True,
                        force: bool = True):
    """

    force: this argument when true forces anything that isn't jsonable into a string so that you force it to save as
    json data anyway. e.g. objs, function, tensorboard etc. are made into ANY string representation they have & saved.
    This is likely useful when you have param args floating around carying pointers/refs to arbitrary data but you
    want to save it anyway.
    """
    import json

    expanduser(path2filename)

    data = to_json(data) if force else data
    with open(path2filename, mode) as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)


def expanduser(path: Union[str, Path]):
    if not isinstance(path, Path):
        path: Path = Path(path).expanduser()
    path.expanduser()
    assert not '~' in str(path), f'Path username was not expanded properly see path: {path=}'
    # return path

# def save_to_json():
#     if not isinstance(path2filename, Path):
#         path2filename: Path = Path(path2filename).expanduser()
#     path2filename.expanduser()
#
#     with open(path2filename, mode) as f:
#         json.dump(to_json(dic), f, indent=indent, sort_keys=sort_keys)

def save_args_to_sorted_json(args, dirpath):
    with open(dirpath / 'args.json', 'w+') as argsfile:
        args_data = {key: str(value) for (key, value) in args.__dict__.items()}
        json.dump(args_data, argsfile, indent=4, sort_keys=True)


def save_opts_to_sorted_json(opts, dirpath):
    save_args_to_sorted_json(opts, dirpath)


def save_git_hash_if_possible_in_args(args, path_to_repo_root):
    """
    :param args:
    :param path_to_repo_root: e.g. '~/automl-meta-learning/'
    :return:
    """
    try:
        githash = subprocess.check_output(
            ['git', '-C', str(Path(path_to_repo_root).expanduser()), 'rev-parse', 'HEAD'])
        args.githash = githash
        print(f'githash: {githash} \n"')
    except:
        args.githash = -1


##

def to_table(df):
    """
    Need to pass the rows, columns data etc to table and create axis to create the table I want.
    But honestly the to latex function is much better.
    :param df:
    :return:
    """
    from pandas.plotting import table
    # table()
    # df = df.astype(str)
    table(data=df.values)
    table(df)
    df.plot(table=True)


def to_latex_is_rapid_learning_real(df: DataFrame):
    # put the |c|...|c| for each column
    column_format = '|'
    num_columns = len(df.columns) + 1  # +1 because the "index" df is the rows
    for _ in range(num_columns):
        column_format = column_format + 'c|'
    latex = df.to_latex(column_format=column_format)
    # replace \toprule, \midrule, \bottomrule with \hline (looks better plus don't need to import \usepackage{booktabs}
    rules = ['\\toprule', '\\midrule', '\\bottomrule']
    latex = latex.replace(rules[0], '\\hline')
    latex = latex.replace(rules[1], '\\hline')
    latex = latex.replace(rules[2], '\\hline')
    latex = latex.replace('+-', ' $\\pm$ ')
    return latex


##

def create_logs_dir_and_load(opts):
    from uutils import load_cluster_jobids_to
    from datetime import datetime

    load_cluster_jobids_to(opts)
    # opts.log_root = Path('~/data/logs/').expanduser()
    # create and load in args path to current experiments from log folder path
    opts.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    opts.log_experiment_dir = opts.log_root / f'logs_{opts.current_time}_jobid_{opts.jobid}'
    opts.log_experiment_dir.mkdir(parents=True, exist_ok=True)
    # create and load in args path to checkpoint (same as experiment log path)
    opts.checkpoint_dir = opts.log_experiment_dir


def save_opts(opts: Namespace, args_filename: str = 'opts.json'):
    """ Saves opts, crucially in sorted order. """
    # save opts that was used for experiment
    with open(opts.log_root / args_filename, 'w') as argsfile:
        # in case some things can't be saved to json e.g. tb object, torch_uu.Tensors, etc.
        args_data = {key: str(value) for key, value in opts.__dict__.items()}
        json.dump(args_data, argsfile, indent=4, sort_keys=True)


def save_args(args: Namespace, args_filename: str = 'args.json'):
    """ Saves args, crucially in sorted order. """
    save_opts(args, args_filename)


def load_cluster_jobids_to(args):
    import os

    # Get Get job number of cluster
    args.jobid = -1
    args.slurm_jobid, args.slurm_array_task_id = -1, -1
    args.condor_jobid = -1
    args.hostname = str(gethostname())
    if "SLURM_JOBID" in os.environ:
        args.slurm_jobid = int(os.environ["SLURM_JOBID"])
        args.jobid = args.slurm_jobid
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        args.slurm_array_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    # - note this is set manually in the .sub file so you might have a different name e.g. MY_CONDOR_JOB_ID
    if "CONDOR_JOB_ID" in os.environ:
        args.condor_jobid = int(os.environ["CONDOR_JOB_ID"])
        args.jobid = args.condor_jobid

    if "PBS_JOBID" in os.environ:
        # args.num_workers = 8
        try:
            args.jobid = int(os.environ["PBS_JOBID"])
        except:
            args.jobid = os.environ["PBS_JOBID"]
    if 'dgx' in str(gethostname()):
        args.jobid = f'{args.jobid}_pid_{os.getpid()}'


def set_system_wide_force_flush():
    """
    Force flushes the entire print function everywhere.

    https://stackoverflow.com/questions/230751/how-to-flush-output-of-print-function
    :return:
    """
    import builtins
    import functools
    print2 = functools.partial(print, flush=True)
    builtins.print = print2


# graph stuff

def draw_nx(g, labels=None):
    import matplotlib.pyplot as plt
    if labels is not None:
        g = nx.relabel_nodes(g, labels)
    pos = nx.kamada_kawai_layout(g)
    nx.draw(g, pos, with_labels=True)
    plt.show()


def draw_nx_attributes_as_labels(g, attribute):
    # import pylab
    import matplotlib.pyplot as plt
    import networkx as nx
    labels = nx.get_node_attributes(g, attribute)
    pos = nx.kamada_kawai_layout(g)
    nx.draw(g, pos, labels=labels, with_labels=True)
    # nx.draw(g, labels=labels)
    # pylab.show()
    plt.show()


def draw_nx_with_pygraphviz(g, path2file=None, save_file=False):
    attribute_name = None
    draw_nx_with_pygraphviz_attribtes_as_labels(g, attribute_name, path2file, save_file)


def draw_nx_with_pygraphviz_attribtes_as_labels(g, attribute_name, path2file=None, save_file=False):
    import pygraphviz as pgv
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # https://stackoverflow.com/questions/15345192/draw-more-information-on-graph-nodes-using-pygraphviz
    # https://stackoverflow.com/a/67442702/1601580

    if path2file is None:
        path2file = './example.png'
        path2file = Path(path2file).expanduser()
        save_file = True
    if type(path2file) == str:
        path2file = Path(path2file).expanduser()
        save_file = True

    print(f'\n{g.is_directed()=}')
    g = nx.nx_agraph.to_agraph(g)
    if attribute_name is not None:
        print(f'{g=}')
        # to label in pygrapviz make sure to have the AGraph obj have the label attribute set on the nodes
        g = str(g)
        g = g.replace(attribute_name, 'label')
        print(g)
        # g = pgv.AGraph(g)
        g = pgv.AGraph(g)
    g.layout()
    g.draw(path2file)

    # https://stackoverflow.com/questions/20597088/display-a-png-image-from-python-on-mint-15-linux
    img = mpimg.imread(path2file)
    plt.imshow(img)
    plt.show()

    # remove file https://stackoverflow.com/questions/6996603/how-to-delete-a-file-or-folder
    if not save_file:
        path2file.unlink()


def visualize_lark(string: str, parser: Lark, path2filename: Union[str, Path]):
    if type(path2filename) is str:
        path2filename = Path(path2filename).expanduser()
    else:
        path2filename = path2filename.expanduser()
    ast = parser.parse(string)
    tree.pydot__tree_to_png(ast, path2filename)
    # tree.pydot__tree_to_dot(parser.parse(sentence), filename)


def bfs(root_node: Tree, f) -> Tree:
    """

    To do BFS you want to process the elements in the dequeue int he order of a queue i.e. the "fair" way - whoever
    got first you process first and the next children go to the end of the line.
    Thus: BFS = append + popleft = "append_end + pop_frent".
    :param root_node: root_node (same as ast since ast object give pointers to top node & it's children)
    :param f: function to apply to the nodes.
    :return:
    """
    dq = deque([root_node])
    while len(dq) != 0:
        current_node = dq.popleft()  # pops from the front of the queue
        current_node.data = f(current_node.data)
        for child in current_node.children:
            dq.append(child)  # adds to the end
    return root_node


def dfs(root_node: Tree, f) -> Tree:
    """

    To do DFS we need to implement a stack. In other words we need to go down the depth until the depth has been
    fully processed. In other words what you want is the the current child you are adding to the dequeue to be processed
    first i.e. you want it to skip the line. To do that you can append to the front of the queue (with append_left and
    then pop_left i.e. pop from that same side).

    reference: https://codereview.stackexchange.com/questions/263604/how-does-one-write-dfs-in-python-with-a-deque-without-reversed-for-trees
    :param root_node:
    :param f:
    :return:
    """
    dq = deque([root_node])
    while len(dq) != 0:
        current_node = dq.popleft()  # to make sure you pop from the front since you are adding at the front.
        current_node.data = f(current_node.data)
        # print(current_node.children)
        for child in reversed(current_node.children):
            dq.appendleft(child)
    return root_node


def dfs_stack(ast: Tree, f) -> Tree:
    stack = [ast]
    while len(stack) != 0:
        current_node = stack.pop()  # pop from the end, from the side you are adding
        current_node.data = f(current_node.data)
        for child in reversed(current_node.children):
            stack.append(child)
    return ast


def dfs_recursive(ast: Tree, f) -> Tree:
    """
    Go through the first child in children before processing the rest of the tree.
    Note that you can apply f after you've processed all the children too to "colour" the nodes in a different order.
    :param ast:
    :param f:
    :return:
    """
    ast.data = f(ast.data)
    for child in ast.children:
        dfs_recursive(child, f)


def save_dataset_with_dill(path2data: str, dataset_name: str, data_set) -> None:
    path2data: str = str(Path(path2data).expanduser())
    with open(path2data / f'{dataset_name}.pt', 'wb') as f2dataset:
        dill.dump(data_set, f2dataset)


def save_with_dill(path: str, filename: str, python_obj, mode: str = 'wb') -> None:
    path: Path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    filename: str = f'{filename}.pt' if filename[-3:] != '.pt' else filename
    with open(path / filename, mode) as f:
        dill.dump(python_obj, f)


def load_with_dill(path2filename: Union[str, Path], mode='rb') -> Any:
    expanduser(path2filename)
    with open(path2filename, mode) as f:
        python_obj = dill.load(f)
    return python_obj


def load_with_pickle(path2filename: Union[str, Path], mode='rb') -> Any:
    expanduser(path2filename)
    with open(path2filename, mode) as f:
        python_obj = pickle.load(f)
    return python_obj


def load_with_torch(path2filename: Union[str, Path], mode='rb') -> Any:
    expanduser(path2filename)
    with open(path2filename, mode) as f:
        import torch
        python_obj = torch.load(f)
    return python_obj


def load_json(path2filename: Union[str, Path], mode: str = 'r') -> Union[dict, list]:
    expanduser(path2filename)
    with open(path2filename, mode) as f:
        data: dict = json.load(f)
    return data


def _load_json(path: str, filename: str, mode='r') -> Union[dict, list]:
    from pathlib import Path
    import json

    path = Path(path).expanduser()
    with open(path / filename, mode) as f:
        data: dict = json.load(f)
    return data


def _load_with_dill(path: str, filename: str, mode='rb') -> Any:
    path: Path = Path(path).expanduser()
    with open(path / filename, mode) as f:
        python_obj = dill.load(f)
    return python_obj


def load_json_list(path: str, filename: str, mode='r') -> list:
    return load_json(path / filename, mode)


def write_str_to_file(path: str, filename: str, file_content: str, mode: str = 'w'):
    path: Path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)  # if it already exists it WONT raise an error (exists is ok!)
    with open(path / filename, mode) as f:
        f.write(file_content)


def namespace2dict(args: Namespace) -> dict:
    """
    Retunrs a dictionary version of the namespace.

    ref: - https://docs.python.org/3/library/functions.html#vars
         - https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-argparse-namespace-as-a-dictionary
    Note: Return the __dict__ attribute for a module, class, instance, or any other object with a __dict__ attribute.
    :param args:
    :return:
    """
    # Note: Return the __dict__ attribute for a module, class, instance, or any other object with a __dict__ attribute.
    return vars(args)


def dict2namespace(dict_args: dict) -> Namespace:
    """
    Makes a dictionary of args to a Namespace args with the same values.

    ref:
        - https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
    """
    return Namespace(**dict_args)


def merge_two_dicts(starting_dict: dict, updater_dict: dict) -> dict:
    """
    Starts from base starting dict and then adds the remaining key values from updater replacing the values from
    the first starting dict with the second updater dict.
    Thus, the update_dict has precedence as it updates and replaces the values from the first if there is a collision.

    ref: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
    For later: how does d = {**d1, **d2} replace collision?
    :param starting_dict:
    :param updater_dict:
    :return:
    """
    new_dict: dict = starting_dict.copy()  # start with keys and values of starting_dict
    new_dict.update(updater_dict)  # modifies starting_dict with keys and values of updater_dict
    return new_dict


def merge_args_safe(args1: Namespace, args2: Namespace) -> Namespace:
    """
    Merges two namespaces but throws an error if there are keys that collide.
    Thus, args1 nor args2 will have precedence when returning the args if there is a collions.
    This does not take the union.

    ref: https://stackoverflow.com/questions/56136549/how-can-i-merge-two-argparse-namespaces-in-python-2-x
    :param args1:
    :param args2:
    :return:
    """
    # - the merged args
    # The vars() function returns the __dict__ attribute to values of the given object e.g {field:value}.
    args = Namespace(**vars(args1), **vars(args2))
    return args


def merge_args(starting_args: Namespace, updater_args: Namespace) -> Namespace:
    """
    Starts from base starting_args and then adds the remaining key/fields values from updater replacing the values from
    the first starting args with the second updater args.
    Thus, the update_args has precedence as it updates and replaces the values from the first if there is a collision.

    ref: https://stackoverflow.com/questions/56136549/how-can-i-merge-two-argparse-namespaces-in-python-2-x
    """
    # - the merged args
    # node: The vars() function returns the __dict__ attribute to values of the given object e.g {field:value}.
    merged_key_values_for_namespace: dict = merge_two_dicts(vars(starting_args), vars(updater_args))
    args = Namespace(**merged_key_values_for_namespace)
    return args


def is_pos_def(x: np.ndarray) -> bool:
    """
    ref:
        - https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    """
    return np.all(np.linalg.eigvals(x) > 0)


def _put_pm_to_pandas_data(data: dict) -> dict:
    """
    Change the +- to \pm for latex display.

    ref:
        - https://stackoverflow.com/questions/70008992/how-to-print-a-literal-backslash-to-get-pm-in-a-pandas-data-frame-to-generate-a
    """
    for column_name, data_values in data.items():
        # data[column_name] = [data_value.replace('+-', r'\pm') for data_value in data_values]
        # data[column_name] = [data_value.replace('+-', r'\\pm') for data_value in data_values]
        data[column_name] = [data_value.replace('+-', '\pm') for data_value in data_values]
    return data


def is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


# def to_float(element: Any) -> float:
#     try:
#         return float(element)
#     except ValueError:
#         return element

def is_positive_int(value: str) -> bool:
    """

    refs:
        - https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
        - isdigit vs isnumeric: https://stackoverflow.com/questions/44891070/whats-the-difference-between-str-isdigit-isnumeric-and-isdecimal-in-python
    """
    # return value.isdigit()
    return value.isalnum()
    return value.isnumeric()


def is_negative_int(value: str) -> bool:
    """
    ref:
        - https://www.kite.com/python/answers/how-to-check-if-a-string-represents-an-integer-in-python#:~:text=To%20check%20for%20positive%20integers,rest%20must%20represent%20an%20integer.
        - https://stackoverflow.com/questions/37472361/how-do-i-check-if-a-string-is-a-negative-number-before-passing-it-through-int
    """
    if value == "":
        return False
    is_positive_integer: bool = value.isdigit()
    if is_positive_integer:
        return True
    else:
        is_negative_integer: bool = value.startswith("-") and value[1:].isdigit()
        is_integer: bool = is_positive_integer or is_negative_integer
        return is_integer


# -- regex

def matches_regex(regex: str, content: str) -> Optional[Match[str]]:
    import re
    return re.match(regex, content)


# -- tests

def match_regex_test():
    import re

    regex = r"\s*Proof.\s*"
    contents = ['Proof.\n', '\nProof.\n']
    for content in contents:
        assert re.match(regex, content), f'Failed on {content=} with {regex=}'
        if re.match(regex, content):
            print(content)


def draw_test():
    # import pylab
    # import matplotlib.my_pyplot as plt
    import networkx as n
    # https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python
    g = nx.DiGraph()
    print(f'{g.is_directed()=}')
    g.directed = True
    print(f'{g=}')
    g.add_node('Golf', size='small')
    g.add_node('Hummer', size='huge')
    g.add_node('Soccer', size='huge')
    g.add_edge('Golf', 'Hummer')
    print(f'{g=}')
    print(f'{str(g)=}')
    print(f'{g.is_directed()=}')
    draw_nx_with_pygraphviz_attribtes_as_labels(g, attribute_name='size')
    draw_nx_with_pygraphviz(g)
    draw_nx(g)


def bfs_test():
    # token requires two values due to how Lark works, ignore it
    ast = Tree(1, [Tree(2, [Tree(3, []), Tree(4, [])]), Tree(5, [])])
    print()
    # the key is that 5 should go first than 3,4 because it is BFS
    bfs(ast, print)


def dfs_test():
    print()
    # token requires two values due to how Lark works, ignore it
    ast = Tree(1, [Tree(2, [Tree(3, []), Tree(4, [])]), Tree(5, [])])
    # the key is that 3,4 should go first than 5 because it is DFS
    dfs(ast, print)
    #
    print()
    # token requires two values due to how Lark works, ignore it
    ast = Tree(1, [Tree(2, [Tree(3, []), Tree(4, [])]), Tree(5, [])])
    # the key is that 3,4 should go first than 5 because it is DFS
    dfs_stack(ast, print)
    #
    print()
    # token requires two values due to how Lark works, ignore it
    ast = Tree(1, [Tree(2, [Tree(3, []), Tree(4, [])]), Tree(5, [])])
    # the key is that 3,4 should go first than 5 because it is DFS
    dfs_recursive(ast, print)


def good_progressbar_test():
    import time
    bar = get_good_progressbar()
    for i in bar(range(100)):
        time.sleep(0.1)
        bar.update(i)

    print('---- start context manager test ---')
    max_value = 10
    with get_good_progressbar(max_value=max_value) as bar:
        for i in range(max_value):
            time.sleep(1)
            bar.update(i)


def xor_test():
    assert xor(0, 0) == False
    assert xor(0, 1) == True
    assert xor(1, 0) == True
    assert xor(1, 1) == False
    print('passed xor test')


def merge_args_test():
    """
    After the merge the starting dict will be updated with values on the second. The second has precedence.
    """
    args1 = Namespace(foo="foo", collided_key='from_args1')
    args2 = Namespace(bar="bar", collided_key='from_args2')

    # - after the merge the starting dict will be updated with values on the second. The second has precedence.
    args = merge_args(starting_args=args1, updater_args=args2)
    print('-- merged args')
    print(f'{args=}')
    assert args.collided_key == 'from_args2', 'Error in merge dict, expected the second argument to be the one used' \
                                              'to resolve collision'


def _map_args_fields_from_string_to_usable_value_test():
    path_to_checkpoint: str = '/Users/brando/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668'
    args: Namespace = get_args_from_checkpoint_json_file(path=path_to_checkpoint, filename='args.json')
    pprint_args(args)
    args: Namespace = map_args_fields_from_string_to_usable_value(args)
    print('----')
    pprint_args(args)


if __name__ == '__main__':
    print('starting __main__ at __init__')
    # test_draw()
    # test_dfs()
    # test_good_progressbar()
    # xor_test()
    # merge_args_test()
    _map_args_fields_from_string_to_usable_value_test()
    print('Done!\a')
