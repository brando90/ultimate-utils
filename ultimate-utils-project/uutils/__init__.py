'''
Utils class with useful helper functions

utils: https://www.quora.com/What-do-utils-files-tend-to-be-in-computer-programming-documentation
'''

import time

import math
import numpy as np
import random

import time
import logging
import argparse

import os
import shutil
import sys

import pathlib
from pathlib import Path

from pdb import set_trace as st

def parse_args():
    """
        Parses command line arguments
    """
    parser = argparse.ArgumentParser(description="DiMO")
    parser.add_argument(
        "data",
        metavar="DIR",
        help="path to dataset"
    )
    parser.add_argument(
        "--seed",
        help="seed for deterministic experimenting",
        default=61820
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="number of epochs",
        default=100
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="learning rate of the child model",
        dest="lr"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="size of batch for training and validation",
        default=64
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        help="number of nodes per cell",
        default=5
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        help="number of blocks of convolution/reduction cells",
        default=3
    )
    parser.add_argument(
        "--num-conv-cells",
        type=int,
        help="number of convolution cells per block",
        default=3
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="dropout probability",
        default=0.0
    )
    parser.add_argument(
        "--init-channels",
        type=int,
        help="initial channel to increase to after first convolution",
        default=64
    )

    return parser.parse_args()

def get_logger(log_path, log_filename):
    """
        Initializes and returns a standard logger
    """
    logger = logging.getLogger(log_filename)
    file_handler = logging.FileHandler(log_filename + ".log")
    logger.addHandler(file_handler)

    return logger

def HelloWorld():
    return 'HelloWorld in Utils!'

def remove_folders_recursively(path):
    print('WARNING: HAS NOT BEEN TESTED')
    path.expanduser()
    try:
        shutil.rmtree(str(path))
    except OSError:
        # do nothing if removing fails
        pass

def oslist_for_path(path):
    return [f for f in path.iterdir() if f.is_dir()]

def _make_and_check_dir(path):
    '''
    NOT NEEDED use:

    mkdir(parents=True, exist_ok=True) see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
    '''
    path = os.path.expanduser(path)
    path.makdir(parents=True, exit_ok=True) # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.

def timeSince(start):
    '''
    How much time has passed since the time "start"

    :param float start: the number representing start (usually time.time())
    '''
    now = time.time()
    s = now - start
    ## compute how long it took in hours
    h = s/3600
    ## compute numbers only for displaying how long it took
    m = math.floor(s / 60) # compute amount of whole integer minutes it took
    s -= m * 60 # compute how much time remaining time was spent in seconds
    ##
    msg = f'time passed: hours:{h}, minutes={m}, seconds={s}'
    return msg, h

def report_times(start, verbose=False):
    '''
    How much time has passed since the time "start"

    :param float start: the number representing start (usually time.time())
    '''
    meta_str=''
    ## REPORT TIMES
    start_time = start
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    if verbose:
        print(f"--- {seconds} {'seconds '+meta_str} ---")
        print(f"--- {minutes} {'minutes '+meta_str} ---")
        print(f"--- {hours} {'hours '+meta_str} ---")
        print('\a')
    ##
    msg = f'time passed: hours:{hours}, minutes={minutes}, seconds={seconds}'
    return msg, seconds, minutes, hours

def is_NaN(value):
    '''
    Checks is value is problematic by checking if the value:
    is not finite, is infinite or is already NaN
    '''
    return not np.isfinite(value) or np.isinf(value) or np.isnan(value)

##

def make_and_check_dir2(path):
    '''
        tries to make dir/file, if it exists already does nothing else creates it.
    '''
    try:
        os.makedirs(path)
    except OSError:
        pass

####

'''
Greater than 4 I get this error:

ValueError: Seed must be between 0 and 2**32 - 1
'''

def get_truly_random_seed_through_os():
    '''
    Usually the best random sample you could get in any programming language is generated through the operating system. 
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    '''
    RAND_SIZE = 4
    random_data = os.urandom(RAND_SIZE) # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

def seed_everything(seed=42):
    '''
    https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch
    '''
    import torch
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_hostname_mit():
    from socket import gethostname
    hostname = gethostname()
    if 'polestar-old' in hostname or hostname=='gpu-16' or hostname=='gpu-17':
        return 'polestar-old'
    elif 'openmind' in hostname:
        return 'OM'
    else:
        return hostname

def make_dirpath_current_datetime_hostname(path=None, comment=''):
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
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # make runs logdir path
    runs_log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
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
        local_hosts = ['Sarahs-iMac.local','Brandos-MacBook-Pro.local','Beatrizs-iMac.local']
    hostname = gethostname()
    if hostname in local_hosts:
        return True
    else: # not known local host
        return False

def my_print(*args, filepath='~/my_stdout.txt'):
    """Modified print statement that prints to terminal/scree AND to a given file (or default).

    Note: import it as follows:

    from utils.utils import my_print as print

    to overwrite builtin print function
    
    Keyword Arguments:
        filepath {str} -- where to save contents of printing (default: {'~/my_stdout.txt'})
    """
    filepath = Path(filepath).expanduser()
    # do normal print
    __builtins__['print'](*args, file=sys.__stdout__) #prints to terminal
    # open my stdout file in update mode
    with open(filepath, "a+") as f:
        # save the content we are trying to print
        __builtins__['print'](*args, file=f) #saves to file

def collect_content_from_file(filepath):
    filepath = Path(filepath).expanduser()
    contents = ''
    with open(filepath,'r') as f:
        for line in f.readlines():
            contents = contents + line
    return contents

## cluster stuff

def get_cluster_jobids(args):
    ## Get Get job number of cluster
    args.jobid = -1
    args.slurm_jobid, args.slurm_array_task_id = -1, -1
    if 'SLURM_JOBID' in os.environ:
        args.slurm_jobid = int(os.environ['SLURM_JOBID'])
        args.jobid = args.slurm_jobid
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        args.slurm_array_task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    args.condor_jobid = -1
    if 'MY_CONDOR_JOB_ID' in os.environ:
        args.condor_jobid = int(os.environ['MY_CONDOR_JOB_ID'])
        args.jobid = args.condor_jobid
    return args

if __name__ == '__main__':
    #send_email('msg','miranda9@illinois.edu')
    print('sending email test')
    p = Path('~/automl-meta-learning/automl/experiments/pw_app.config.json').expanduser()
    send_email(subject='TEST: send_email2', message='MESSAGE', destination='brando.science@gmail.com', password_path=p)
    print(f'EMAIL SENT\a')
    print('Done \n\a')
