"""
Utils class with useful helper functions

utils: https://www.quora.com/What-do-utils-files-tend-to-be-in-computer-programming-documentation
"""
import json
import pickle
import re
import subprocess
import time

import math
import torch
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

from typing import Union, Any, Optional, Match, Callable

import progressbar

from uutils.logging_uu.wandb_logging.common import setup_wandb
from uutils.torch_uu.distributed import find_free_port


def hello():
    import uutils
    print(f'\nhello from uutils __init__.py in:\n{uutils}\n')


def helloworld():
    hello()


def hello_world():
    hello()


def print_file(path_or_str: Union[str, Path]) -> None:
    """ prints the content of a file """
    cat_file(path2filename=path_or_str)


def clear_file_contents(path2file: Union[str, Path]):
    """
    Clears contents of a file.


    reason it works from SO:
        the reason this works (in both C++ and python) is because by default when you open a file for writing, it truncates the existing contents. So really it's sorta a side effect, and thus I would prefer the explicit call to truncate() for clarity reasons, even though it is unnecessary.

    ref: https://stackoverflow.com/questions/2769061/how-to-erase-the-file-contents-of-text-file-in-python
    """
    path2file: Path = expanduser(path2file)
    open(path2file, "w").close()
    # f = open('file.txt', 'r+')
    # f.truncate(0)  # need '0' when using r+


def print_python_version():
    import sys

    print(f'python version: {sys.version=}')


# - getting args for expts


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


def run_bash_command(cmd: str, use_run: bool = True) -> Any:
    """
    Runs a given command in bash from within python.

    Details (from SO):
        The main difference is that subprocess.run() executes a command and waits for it to finish, while with
        subprocess.Popen you can continue doing your stuff while the process finishes and then just repeatedly call
        Popen.communicate() yourself to pass and receive data to your process. Secondly, subprocess.run() returns
        subprocess.CompletedProcess.
        subprocess.run() just wraps Popen and Popen.communicate() so you don't need to make a loop to pass/receive data
        or wait for the process to finish.

    ref:
        - https://stackoverflow.com/questions/39187886/what-is-the-difference-between-subprocess-popen-and-subprocess-run#:~:text=The%20main%20difference%20is%20that,receive%20data%20to%20your%20process.
    """
    import subprocess
    if use_run:
        # - recommended, see comment above. My understanding is that this blocks until done
        res = subprocess.run(cmd.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res
    else:
        # - I think this is fine for simple commands but I think one has to loop through the communicate until done
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            raise Exception(error)
        else:
            return output


def stanford_reauth():
    """"
    re-authenticates the python process in the kerberos system so that the
    python process is not killed randomly.

    ref: https://unix.stackexchange.com/questions/724902/how-does-one-send-new-commands-to-run-to-an-already-running-nohup-process-or-run
    """
    reauth_cmd: str = f'echo $SU_PASSWORD | /afs/cs/software/bin/reauth'
    out = run_bash_command(reauth_cmd)
    print('Output of reauth (/afs/cs/software/bin/reauth with password): ')
    print(f'--> {out=}')
    raise Exception('For now we are not doing reauth within python')


def get_nvidia_smi_output() -> str:
    out = run_bash_command('nvidia-smi')
    return out


def _get_git_revision_hash():
    """ ref: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def _get_git_revision_short_hash():
    """ ref: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script """
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def cat_file(path2filename: Union[str, Path]):
    """prints/displays file contents. Do path / filename or the like outside of this function. ~ is alright to use. """
    path2filename = expanduser(path2filename)
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


def copy_folders_recursively(src_root: Union[str, Path], root4dst: Union[str, Path],
                             dirnames4dst: list[Union[str, Path]]):
    """
    Copying dirnames in src_root into roo4dst. Note dirnames4dst should be the same as the dirnames in the sorc.

    note: copying the files was fast locally! Likely the downloading of the data from network + extracting is what causes
    things to be slow.

    :param src_root:
    :param root4dst:
    :param dirnames4dst:
    :return:
    """
    root: Path = expanduser(root4dst)
    src_root: Path = expanduser(src_root)
    from distutils.dir_util import copy_tree
    for dirname in dirnames4dst:
        dirname: Path = expanduser(dirname)
        src: Path = src_root / expanduser(dirname)
        dst: Path = root / expanduser(dirname)
        src: str = str(src)
        dst: str = str(dst)
        print(f'copying: {src=} -> {dst=}')
        copy_tree(src, dst)


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


def report_times(start: float, verbose: bool = False) -> str:
    import time
    duration_secs = time.time() - start
    msg = f"time passed: hours:{duration_secs / (60 ** 2)}, minutes={duration_secs / 60}, seconds={duration_secs}"
    if verbose:
        print(msg)
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


def check_number_of_files_open_vs_allowed_open():
    """
    Checks if the number of files open is close to the number of files that can be open.

    ref: https://stackoverflow.com/questions/12090503/listing-open-files-using-python
    """
    import resource
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"soft_limit={soft_limit}, hard_limit={hard_limit}")
    # import psutil
    # p = psutil.Process()
    # print(f"number of files open={p.num_fds()}")
    import psutil
    open_files = 0
    for process in psutil.process_iter():
        try:
            files = process.open_files()
            open_files += len(files)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    print(f"Number of open files: {open_files}")


def get_filtered_local_params(local_vars, verbose: bool = False, var_name_in_front: str = '') -> dict[str, Any]:
    # Get all local variables done outside the function
    # e.g. local_vars = locals()

    # Filter out undesired variables
    filtered_vars = {k: v for k, v in local_vars.items() if k != 'self' and not k.startswith('__')}

    # Print the variable names and their values
    if verbose:
        if var_name_in_front != '':
            print(f"{var_name_in_front}:")
        for var_name, var_value in filtered_vars.items():
            print(f"{var_name}: {var_value}")
    return filtered_vars


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


def get_truly_random_seed_through_os(rand_size: int = 4) -> int:
    """
    Usually the best random sample you could get in any programming language is generated through the operating system. 
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    random_data = os.urandom(
        rand_size
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed: int = int.from_bytes(random_data, byteorder="big")
    return int(random_seed)


def get_different_pseudo_random_seed_every_time_using_time() -> int:
    """ Get a different pseudo random seed every time using time."""
    import random
    import time

    # random.seed(int(time.time()))
    seed: int = int(time.time())
    return seed


def seed_everything(seed: int,
                    seed_torch: bool = True,
                    ):
    """
    https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch
    """
    import random
    import numpy as np
    import os

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if seed_torch:
        import torch
        torch.manual_seed(seed)
        # makes convs deterministic: https://stackoverflow.com/a/66647424/1601580, torch.backends.cudnn.deterministic=True only applies to CUDA convolution operations, and nothing else. Therefore, no, it will not guarantee that your training process is deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # #torch.use_deterministic_algorithms(True) # do not uncomment
        # fully_deterministic: bool = uutils.torch_uu.make_code_deterministic(args.seed)


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
    print(f'args=')
    print_args(args)
    print()


def pprint_dict(dic):
    pprint_any_dict(dic)


def pprint_any_dict(dic: dict, indent: Optional[int] = None, var_name_in_front: str = ''):
    """
    This pretty prints a json.

    Note: this is not the same as pprint.

    Warning:
        - if indent is an int then the values will become strings.

    todo: how to have pprint do indent and keep value without making it into a string.
    """
    import json
    if var_name_in_front != '':
        print(f'{var_name_in_front}=')

    if indent:
        # make all keys strings recursively with their naitve str function
        dic = to_json(dic)
        print(json.dumps(dic, indent=4, sort_keys=True))  # only this one works...idk why
    else:
        pprint(dic)
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

    path2filename: Path = expanduser(path2filename)

    data = to_json(data) if force else data
    with open(path2filename, mode) as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)


def expanduser(path: Union[str, Path]) -> Path:
    """

    note: if you give in a path no need to get the output of this function because it mutates path. If you
    give a string you do need to assign the output to a new variable
    :param path:
    :return:
    """
    if not isinstance(path, Path):
        # path: Path = Path(path).expanduser()
        path: Path = Path(path).expanduser()
    path = path.expanduser()
    assert not '~' in str(path), f'Path username was not expanded properly see path: {path=}'
    return path


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


def dicts_to_jsonl(data_list: list[dict], path2filename: Union[str, Path], compress: bool = False) -> None:
    """
    Method saves list of dicts into jsonl file.
    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?

    credit:
        - https://stackoverflow.com/questions/73312575/what-is-the-official-way-to-save-list-of-dictionaries-as-jsonl-or-json-lines
        - https://ml-gis-service.com/index.php/2022/04/27/toolbox-python-list-of-dicts-to-jsonl-json-lines/
    """
    import gzip
    import json

    sjsonl = '.jsonl'
    sgz = '.gz'
    # Check filename
    if not str(path2filename).endswith(sjsonl):
        path2filename = Path(str(path2filename) + sjsonl)
    path2filename = expanduser(path2filename)
    # Save data

    if compress:
        filename = path2filename + sgz
        with gzip.open(filename, 'w') as compressed:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                jout = jout.encode('utf-8')
                compressed.write(jout)
    else:
        with open(path2filename, 'w') as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                out.write(jout)


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
    # from uutils import load_cluster_jobids_to
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


def get_home_pwd_local_machine_snap() -> None:
    """
    Gets the path to local machine in snap (lfs) with user name appended at the end.

bash command:
# Local machine as Home
export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print(f'/lfs/{hostname}/0/brando9');")
mkdir -p $LOCAL_MACHINE_PWD
export WANDB_DIR=$LOCAL_MACHINE_PWD
export HOME=$LOCAL_MACHINE_PWD

# Without python, doesn't work fix some day...
# export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print(f'/lfs/{hostname}/0/brando9');")
# export LOCAL_MACHINE_PWD=$(python3 -c "import uutils; uutils.get_home_pwd_local_machine_snap()")
export HOSTNAME=$(hostname)
export LOCAL_MACHINE_PWD="/lfs/${HOSTNAME::-13}/0/brando9"
mkdir -p $LOCAL_MACHINE_PWD
export WANDB_DIR=$LOCAL_MACHINE_PWD
export HOME=$LOCAL_MACHINE_PWD

ref:
    - one liner: https://stackoverflow.com/questions/27658675/how-to-remove-last-n-characters-from-a-string-in-bash
    """
    import socket
    hostname: str = socket.gethostname()
    hostname: str = hostname.split('.')[0]
    local_pwd: str = f'/lfs/{hostname}/0/brando9'
    print(local_pwd)  # returns to terminal
    # return local_pwd
    # on liner
    # import socket;hostname = socket.gethostname().split('.')[0];print(f'/lfs/{hostname}/0/brando9');


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

    if 'LSB_JOBID' in os.environ:
        jobid: str = str(os.environ['LSB_JOBID'])
        args.jobid = jobid


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
        path2file.unlink(missing_ok=True)


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
    path2filename = expanduser(path2filename)
    with open(path2filename, mode) as f:
        python_obj = dill.load(f)
    return python_obj


def load_with_pickle(path2filename: Union[str, Path], mode='rb') -> Any:
    path2filename = expanduser(path2filename)
    with open(path2filename, mode) as f:
        python_obj = pickle.load(f)
    return python_obj


def load_with_torch(path2filename: Union[str, Path], mode='rb') -> Any:
    path2filename = expanduser(path2filename)
    with open(path2filename, mode) as f:
        import torch
        python_obj = torch.load(f)
    return python_obj


def load_json(path2filename: Union[str, Path], mode: str = 'r') -> Union[dict, list]:
    path2filename = expanduser(path2filename)
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


def check_dict1_is_in_dict2(dict1: dict,
                            dict2: dict,
                            verbose: bool = False,
                            ) -> bool:
    """
    Check if dict1 is in dict2. i.e. dict1 <= dict2.
    """
    for k, v in dict1.items():
        if k not in dict2:
            print(f'--> {k=} is not in dict2 with value {dict1[k]=}')
            return False
        if v != dict2[k]:
            print(f'--> {k=} is in dict2 but with different value \n{dict1[k]=} \n{dict2[k]=}')
            return False
        if verbose:
            print(f"--> {k=} is in both dicts, look: \n{dict1[k]=} \n{dict2[k]=} \n")
    return True


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


def is_anonymous_function(f: Any) -> bool:
    """
    Returns true if it's an anonynouys function.

    ref: https://stackoverflow.com/questions/3655842/how-can-i-test-whether-a-variable-holds-a-lambda
    """
    if hasattr(f, '__name__'):
        return callable(f) and f.__name__ == "<lambda>"
    else:
        return False


def get_anonymous_function_attributes(anything: Any,
                                      halt: bool = False,
                                      verbose: bool = False,
                                      very_verbose: bool = False,
                                      ) -> dict[str, Callable]:
    """
    Returns the dictionary of name of fields to anonymous functions in the past anything thing.

    :param very_verbose:
    :param anything:
    :param halt:
    :param verbose:
    :return:
    """
    anons: dict = {}
    for field_name in dir(anything):
        field = getattr(anything, field_name)
        if very_verbose:
            print(f'{field_name=}')
            print(f'{field=}')
        if is_anonymous_function(field):
            if verbose or very_verbose:
                print(f'{field_name=}')
                print(f'{field=}')
            if halt:
                from pdb import set_trace as st
                st()
            anons[str(field_name)] = field
    return anons


def get_anonymous_function_attributes_recursive(anything: Any, path: str = '', print_output: bool = False) -> dict[
    str, Callable]:
    """
    Finds in a dictionary from path of field_name calling to the callable anonymous function.
    It is recommended that you hardcode path to the name of the top object being given to this function (sorry I tried
    doing .__name__ of anything but it doesn't always have that field set).

    :param anything:
    :param path:
    :return:
    """
    anons: dict = {}

    def _get_anonymous_function_attributes_recursive(anything: Any, path: Optional[str] = '') -> None:

        if is_anonymous_function(anything):
            # assert field is anything, f'Err not save thing/obj: \n{field=}\n{anything=}'
            # key: str = str(dict(obj=anything, field_name=field_name))
            key: str = str(path)
            anons[key] = anything
        else:
            for field_name in dir(anything):
                # most likely it's one of YOUR field causing the anonymous function bug, so only loop through those
                if not bool(re.search(r'__(.+)__', field_name)):
                    field = getattr(anything, field_name)
                    # only recurse if new field is not itself
                    if field is not anything:  # avoids infinite recursions
                        # needs a new variable or the paths for different field will INCORRECTLY crash
                        path_for_this_field = f'{path}.{field_name}'
                        print(f'{path_for_this_field}')
                        _get_anonymous_function_attributes_recursive(field, path_for_this_field)
        return

    _get_anonymous_function_attributes_recursive(anything, path)
    if print_output:
        print(f'top path given {path=}')
        print(f'{len(anons.keys())=}')
        for k, v in anons.items():
            print()
            print(f'{k=}')
            print(f'{v=}')
    return anons


def download_and_extract(url: str,
                         path_used_for_zip: Path = Path('~/data/'),
                         path_used_for_dataset: Path = Path('~/data/tmp/'),
                         rm_zip_file_after_extraction: bool = True,
                         force_rewrite_data_from_url_to_file: bool = False,
                         clean_old_zip_file: bool = False,
                         gdrive_file_id: Optional[str] = None,
                         gdrive_filename: Optional[str] = None,
                         ):
    """
    Downloads data and tries to extract it according to different protocols/file types.

    note:
        - to force a download do:
            force_rewrite_data_from_url_to_file = True
            clean_old_zip_file = True
        - to NOT remove file after extraction:
            rm_zip_file_after_extraction = False


    Tested with:
    - zip files, yes!

    Later:
    - todo: tar, gz, gdrive
    force_rewrite_data_from_url_to_file = remvoes the data from url (likely a zip file) and redownloads the zip file.
    """
    path_used_for_zip: Path = expanduser(path_used_for_zip)
    path_used_for_zip.mkdir(parents=True, exist_ok=True)
    path_used_for_dataset: Path = expanduser(path_used_for_dataset)
    path_used_for_dataset.mkdir(parents=True, exist_ok=True)
    # - download data from url
    if gdrive_filename is None:  # get data from url, not using gdrive
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        print("downloading data from url: ", url)
        import urllib
        import http
        response: http.client.HTTPResponse = urllib.request.urlopen(url, context=ctx)
        print(f'{type(response)=}')
        data = response
        # save zipfile like data to path given
        filename = url.rpartition('/')[2]
        path2file: Path = path_used_for_zip / filename
    else:  # gdrive case
        from torchvision.datasets.utils import download_file_from_google_drive
        # if zip not there re-download it or force get the data
        path2file: Path = path_used_for_zip / gdrive_filename
        if not path2file.exists():
            download_file_from_google_drive(gdrive_file_id, path_used_for_zip, gdrive_filename)
        filename = gdrive_filename
    # -- write downloaded data from the url to a file
    print(f'{path2file=}')
    print(f'{filename=}')
    if clean_old_zip_file:
        path2file.unlink(missing_ok=True)
    if filename.endswith('.zip') or filename.endswith('.pkl'):
        # if path to file does not exist or force to write down the data
        if not path2file.exists() or force_rewrite_data_from_url_to_file:
            # delete file if there is one if your going to force a rewrite
            path2file.unlink(missing_ok=True) if force_rewrite_data_from_url_to_file else None
            print(f'about to write downloaded data from url to: {path2file=}')
            # wb+ is used sinze the zip file was in bytes, otherwise w+ is fine if the data is a string
            with open(path2file, 'wb+') as f:
                # with open(path2file, 'w+') as f:
                print(f'{f=}')
                print(f'{f.name=}')
                f.write(data.read())
            print(f'done writing downloaded from url to: {path2file=}')
    elif filename.endswith('.gz'):
        pass  # the download of the data doesn't seem to be explicitly handled by me, that is done in the extract step by a magic function tarfile.open
    # elif is_tar_file(filename):
    #     os.system(f'tar -xvzf {path_2_zip_with_filename} -C {path_2_dataset}/')
    else:
        raise ValueError(f'File type {filename=} not supported.')

    # - unzip data written in the file
    extract_to = path_used_for_dataset
    print(f'about to extract: {path2file=}')
    print(f'extract to target: {extract_to=}')
    if filename.endswith('.zip'):
        import zipfile  # this one is for zip files, inspired from l2l
        zip_ref = zipfile.ZipFile(path2file, 'r')
        zip_ref.extractall(extract_to)
        zip_ref.close()
        if rm_zip_file_after_extraction:
            path2file.unlink(missing_ok=True)
    elif filename.endswith('.gz'):
        import tarfile
        file = tarfile.open(fileobj=response, mode="r|gz")
        file.extractall(path=extract_to)
        file.close()
    elif filename.endswith('.pkl'):
        # no need to extract it, but when you use the data make sure you torch.load it or pickle.load it.
        print(f'about to test torch.load of: {path2file=}')
        data = torch.load(path2file)  # just to test
        assert data is not None
        print(f'{data=}')
        pass
    else:
        raise ValueError(f'File type {filename=} not supported, edit code to support it.')
        # path_2_zip_with_filename = path_2_ziplike / filename
        # os.system(f'tar -xvzf {path_2_zip_with_filename} -C {path_2_dataset}/')
        # if rm_zip_file:
        #     path_2_zip_with_filename.unlink(missing_ok=True)
        # # raise ValueError(f'File type {filename=} not supported.')
    print(f'done extracting: {path2file=}')
    print(f'extracted at location: {path_used_for_dataset=}')
    print(f'-->Succes downloading & extracting dataset at location: {path_used_for_dataset=}')


def _download_url_no_ctx(url):
    # data = urllib.request.urlopen(url)
    # filename = url.rpartition('/')[2]
    # file_path = os.path.join(root, raw_folder, filename)
    # with open(file_path, 'wb') as f:
    #     f.write(data.read())
    # file_processed = os.path.join(root, processed_folder)
    pass


def _download_and_unzip_with_tar_xvzf_py_shell_cmd(url: str, extract_to: Path = Path('~/data/tmp/'),
                                                   mode="r|gz") -> Path:
    """

    this is based on my download_and_extract_miniimagenet but that one uses google to get data so idk if this will work.
    """
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print("downloading dataset from ", url)
    import urllib
    response = urllib.request.urlopen(url, context=ctx)
    # data = urllib.request.urlopen(url)  # note: l2l just does this without ctx
    # from torchvision.datasets.utils import download_file_from_google_drive, extract_archive
    # path = path.expanduser()
    # file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    # filename_zip = 'miniImagenet.tgz'
    # if zip not there re-download it
    # path_2_zip = path / filename_zip
    # if not path_2_zip.exists():
    #     download_file_from_google_drive(file_id, path, filename_zip)

    with open(response, mode='r') as file:
        print("extracting to ", extract_to)
        path_2_zip: str = str(expanduser(extract_to / str(file.name)).name)
        print("path_2_zip is ", path_2_zip)
        os.system(f'tar -xvzf {path_2_zip} -C {extract_to}/')
        return extract_to / str(file.name)


def _download_and_unzip_tinfer(url: str, extract_to: Path = Path('~/data/tmp/')) -> Path:
    """download and unzip ala tinfer proj & returns the path it extracted to."""
    extract_to: Path = expanduser(extract_to)
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print("downloading dataset from ", url)
    import urllib
    response = urllib.request.urlopen(url, context=ctx)
    import tarfile
    file = tarfile.open(fileobj=response, mode="r|gz")
    print("extracting to ", extract_to)
    file.extractall(path=extract_to)
    file.close()

    # todo test bellow to see if file.name is actually correct/works as I expect
    path_2_zip = extract_to / str(file.name)
    return path_2_zip


def _download_ala_l2l_their_original_code(urls, root, raw_folder, processed_folder):
    from six.moves import urllib
    import zipfile

    # if self._check_exists():
    #     return

    # download files
    try:
        os.makedirs(os.path.join(root, raw_folder))
        os.makedirs(os.path.join(root, processed_folder))
    except OSError as e:
        import errno
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    for url in urls:
        print('== Downloading ' + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(root, raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())
        file_processed = os.path.join(root, processed_folder)
        print("== Unzip from " + file_path + " to " + file_processed)

        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(file_processed)
        zip_ref.close()
    print("Download finished.")


# -- bytes for model size

def calculate_bytes_for_model_size(num_params: int,
                                   precision: str = 'float32_bytes',
                                   ) -> int:
    """
    Calculates size of model in given precision ideally in bytes:
        size = num_params * precision in bytes (number of bytes to represent float)

    bits -> bytes == bits / 8 (since 1 bytes is 8 bits)

    number of GigaBytes for model size = num_params * precision in bytes
    1 Byte = 8 bits ~ 1 character = 1 addressable unit in memmory
    FB32 = 4 bytes = 4 * 8 bits = 32 bits = 1S 8Exp 23Mantissa
    FB16 = 2 bytes = 2 * 8 bits = 16 bits = 1S 5Exp 10Mantissa
    BF16 = 2 bytes = 2 * 8 bits = 16 bits = 1S 8Exp 7Mantissa
    Example:
        num_params = 176B (bloom-176B)  # note gpt3 175B params
        precision = 'bfloat16_bytes'  # 4 bytes
        size = 176B * 4 = 176 * 10**8 * 2 = 352 * 10**8 = 352GB (giga bytes)
    :param num_params:
    :return:
    """
    size: int = -1
    if precision == 'float32_bytes':
        num_bytes: int = 4
        size: int = num_params * num_bytes
    if 'bytes' not in precision or 'bits' in precision:  #
        # return in bits
        size: int = num_params * num_bytes * 8
        return size
    else:
        # return in bytes
        return size


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


# --  other

def get_duplicates(lst: list) -> list:
    """
    Returns the duplicate elements in the list.

    ref: https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
    """
    seen: set = set()
    dup: list = []
    for val in lst:
        if val in seen:
            dup.append(val)
        else:
            seen.add(val)
    return dup


def get_non_intersecting_elements(lst: list) -> list:
    """
    Gets you the elements that are not common to the two lists i.e. not intersecting elements.

    ref: https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
    """
    seen: set = set()
    not_intersect: set = set()
    for val in lst:
        if val not in seen:
            not_intersect.add(val)
            seen.add(val)
        else:
            not_intersect.remove(val)
    return list(not_intersect)


def list_of_elements_present(lst: list) -> list:
    """
    Gets you the elements in the list (in the SO post it's refered as the unique elements list).

    ref: https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
    :param lst:
    :return:
    """
    # seen = set()
    # uniq = []
    # for x in a:
    #     if x not in seen:
    #         uniq.append(x)
    #         seen.add(x)
    # return uniq
    return list(set(lst))


def lists_equal(l1: list, l2: list) -> bool:
    """

    import collections
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    ref:
        - https://stackoverflow.com/questions/9623114/check-if-two-unordered-lists-are-equal
        - https://stackoverflow.com/questions/7828867/how-to-efficiently-compare-two-unordered-lists-not-sets
    """
    import collections
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    set_comp = set(l1) == set(l2)  # removes duplicates, so returns true when not sometimes :(
    multiset_comp = compare(l1, l2)  # approximates multiset
    return set_comp and multiset_comp  # set_comp is gere in case the compare function doesn't work


def get_intersection_overlap(a, b) -> float:
    """
    Returns the intersection over union of two bounding boxes.
    Note, lower and upper bounds intersect exactly, it is considered not an intersection.

    ref:
        - https://stackoverflow.com/a/2953979/1601580
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def get_intersection_overlap_care_about_exact_match(a, b) -> float:
    """
    Return the amount of overlap, in bp
    between a and b.
    If >0, the number of bp of overlap
    If 0,  they are book-ended.
    If <0, the distance in bp between them

    - positive if intersect
    - negative if not intersect
    - zero if exqct match

    ref:
        - https://stackoverflow.com/a/52388579/1601580
    """
    return min(a[1], b[1]) - max(a[0], b[0])


# -- tests

def overlap_intersection_test_():
    """
    want to test if two intervals intersect/overlap and return true if they do
    """
    print('----')
    print(f'{get_intersection_overlap([10, 25], [20, 38])}')
    assert get_intersection_overlap([10, 25], [20, 38]) == 5
    print(f'{get_intersection_overlap([20, 38], [10, 25])}')
    assert get_intersection_overlap([20, 38], [10, 25]) == 5

    print(f'{get_intersection_overlap([10, 15], [20, 38])}')
    assert get_intersection_overlap([10, 15], [20, 38]) == 0
    print(f'{get_intersection_overlap([20, 38], [10, 15])}')
    assert get_intersection_overlap([20, 38], [10, 15]) == 0

    print(f'{get_intersection_overlap([10, 15], [15, 38])}')
    assert get_intersection_overlap([10, 15], [15, 38]) == 0
    print(f'{get_intersection_overlap([15, 38], [10, 15])}')
    assert get_intersection_overlap([15, 38], [10, 15]) == 0

    # -
    print('----')
    # positive if intersect
    print(f'{get_intersection_overlap_care_about_exact_match([10, 25], [20, 38])}')
    assert get_intersection_overlap_care_about_exact_match([10, 25], [20, 38]) == 5
    print(f'{get_intersection_overlap_care_about_exact_match([20, 38], [10, 25])}')
    assert get_intersection_overlap_care_about_exact_match([20, 38], [10, 25]) == 5

    # negative if not intersect
    print(f'{get_intersection_overlap_care_about_exact_match([10, 15], [20, 38])}')
    assert get_intersection_overlap_care_about_exact_match([10, 15], [20, 38]) == -5
    print(f'{get_intersection_overlap_care_about_exact_match([20, 38], [10, 15])}')
    assert get_intersection_overlap_care_about_exact_match([20, 38], [10, 15]) == -5

    # zero if exqct match
    print(f'{get_intersection_overlap_care_about_exact_match([10, 15], [15, 38])}')
    assert get_intersection_overlap_care_about_exact_match([10, 15], [15, 38]) == 0
    print(f'{get_intersection_overlap_care_about_exact_match([15, 38], [10, 15])}')
    assert get_intersection_overlap_care_about_exact_match([15, 38], [10, 15]) == 0


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


# --

if __name__ == '__main__':
    print('starting __main__ at __init__')
    # test_draw()
    # test_dfs()
    # test_good_progressbar()
    # xor_test()
    # merge_args_test()
    # _map_args_fields_from_string_to_usable_value_test()
    overlap_intersection_test_()
    print('Done!\a')
