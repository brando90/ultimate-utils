"""
Utils class with useful helper functions

utils: https://www.quora.com/What-do-utils-files-tend-to-be-in-computer-programming-documentation
"""
import json
import subprocess
import time

import math

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

import pygraphviz as pgv

from lark import Lark, tree, Tree, Token

from collections import deque

from argparse import Namespace

from typing import Union

import progressbar

def hello():
    print('hello from uutitls __init__.pyt')

def helloworld(arg1="arg1", arg2="arg2"):
    print("helloworld from uutils __init__.py")


def helloworld2(name="Brando", msg="try to see the params of this function!"):
    print(f"helloworld2: your name is <{name}> and your message <{msg}>")


def HelloWorld():
    return "HelloWorld in Utils!"

def print_pids():
    import torch.multiprocessing as mp

    print('running main()')
    print(f'current process: {mp.current_process()}')
    print(f'pid: {os.getpid()}')

def parse_args():
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

def get_good_progressbar(max_value: Union[int, None] = None) -> progressbar.ProgressBar:
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


def report_times(start):
    import time
    duration_secs = time.time() - start
    msg = f"time passed: hours:{duration_secs/(60**2)}, minutes={duration_secs/60}, seconds={duration_secs}"
    return msg

def is_NaN(value):
    """
    Checks is value is problematic by checking if the value:
    is not finite, is infinite or is already NaN
    """
    return not np.isfinite(value) or np.isinf(value) or np.isnan(value)

def make_args_pickable(args):
    """ Makes a namespace pickable """
    pickable_args = argparse.Namespace()
    # - go through fields in args, if they are not pickable make it a string else leave as it
    # The vars() function returns the __dict__ attribute of the given object.
    for field in vars(args):
        field_val = getattr(args, field)
        if not dill.pickles(field_val):
            field_val = str(field_val)
        setattr(pickable_args, field, field_val)
    return pickable_args

def make_opts_pickable(opts):
    """ Makes a namespace pickable """
    return make_args_pickable(opts)

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


## cluster stuff

def print_args(args: Namespace):
    """
    todo - compare with pprint_any_dict
    :param args:
    :return:
    """
    [print(f'{k, v}') for k, v in vars(args).items()]

def pprint_dict(dic):
    pprint_any_dict(dic)

def pprint_any_dict(dic):
    """
    This pretty prints a json.

    @param dic:
    @return:

    Note: this is not the same as pprint.
    """
    import json

    # make all keys strings recursively with their naitve str function
    dic = to_json(dic)
    # pretty print
    # pretty_dic = json.dumps(dic, indent=4, sort_keys=True)
    # print(pretty_dic)
    print(json.dumps(dic, indent=4, sort_keys=True))  # only this one works...idk why
    # return pretty_dic

def pprint_namespace(ns):
    """ pretty prints a namespace """
    pprint_any_dict(ns)

def _to_json_dict_with_strings(dictionary):
    """
    Convert dict to dict with leafs only being strings. So it recursively makes keys to strings
    if they are not dictionaries.

    Use case:
        - saving dictionary of tensors (convert the tensors to strins!)
        - saving arguments from script (e.g. argparse) for it to be pretty

    e.g.

    """
    # base case: if the input is not a dict make it into a string and return it
    if type(dictionary) != dict:
        return str(dictionary)
    # recurse into all the values that are dictionaries
    d = {k: _to_json_dict_with_strings(v) for k, v in dictionary.items()}
    return d

def to_json(dic):
    import types
    import argparse

    if type(dic) is dict:
        dic = dict(dic)
    else:
        dic = dic.__dict__
    return _to_json_dict_with_strings(dic)

def save_to_json_pretty(dic, path, mode='w', indent=4, sort_keys=True):
    import json

    with open(path, mode) as f:
        json.dump(to_json(dic), f, indent=indent, sort_keys=sort_keys)

def save_args_to_sorted_json(args, dirpath):
    with open(dirpath / 'args.json', 'w+') as argsfile:
        args_data = {key: str(value) for (key, value) in args.__dict__.items()}
        json.dump(args_data, argsfile, indent=4, sort_keys=True)

def save_opts_to_sorted_json(opts, dirpath):
    save_args_to_sorted_json(args, dirpath)

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
    # save opts that was used for experiment
    with open(opts.log_root / args_filename, 'w') as argsfile:
        # in case some things can't be saved to json e.g. tb object, torch.Tensors, etc.
        args_data = {key: str(value) for key, value in opts.__dict__.items()}
        json.dump(args_data, argsfile, indent=4, sort_keys=True)

def save_args(args: Namespace, args_filename: str = 'args.json'):
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

def bfs(root_node : Tree, f) -> Tree:
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

def dfs(root_node : Tree, f) -> Tree:
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

def dfs_stack(ast : Tree, f) -> Tree:
    stack = [ast]
    while len(stack) != 0:
        current_node = stack.pop()  # pop from the end, from the side you are adding
        current_node.data = f(current_node.data)
        for child in reversed(current_node.children):
            stack.append(child)
    return ast

def dfs_recursive(ast : Tree, f) -> Tree:
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

def save_with_dill(path2data: str, dataset_name: str, data_set) -> None:
    path2data: str = str(Path(path2data).expanduser())
    with open(path2data / f'{dataset_name}.pt', 'wb') as f2dataset:
        dill.dump(data_set, f2dataset)

def write_str_to_file(path: str, filename: str, file_content: str, mode: str = 'w'):
    path: Path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)  # if it already exists it WONT raise an error (exists is ok!)
    with open(path / filename, mode) as f:
        f.write(file_content)

# -- tests

def test_draw():
    # import pylab
    # import matplotlib.pyplot as plt
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

def test_bfs():
    # token requires two values due to how Lark works, ignore it
    ast = Tree(1, [Tree(2, [Tree(3, []), Tree(4, [])]), Tree(5, [])])
    print()
    # the key is that 5 should go first than 3,4 because it is BFS
    bfs(ast, print)

def test_dfs():
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

def test_good_progressbar():
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

if __name__ == '__main__':
    print('starting __main__ at __init__')
    # test_draw()
    # test_dfs()
    test_good_progressbar()
    print('Done!\a')