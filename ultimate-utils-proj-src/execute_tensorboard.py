"""
Read the read me of ulimate utils: https://github.com/brando90/ultimate-utils/edit/master/README.md

Since it's easier to just copy paste the path from the remote after downloading it to your local computer, what this script
aims to do is replace the front of the path with a local path. Then you can copy paste remote paths and run
tensorboard (assuming you downloaded the data from the remote) seeminglessly.

Code: Replaces te prefix with the remote/cluster path with the local - so that running tensorboard is seammingless.

e.g. 
Run if alias tbbb is not in .zshrc or .bashrc in local computer
    sh /Users/brando/ultimate-utils/run_tb.sh /home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb
    
else add the alias and run tbbb:
    alias tbb="sh ${HOME}/ultimate-utils/run_tb.sh"
    tbb /home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb
"""

# import os
# import subprocess
from copy import deepcopy

from pathlib import Path

import sys


# print(sys.argv)

def get_path_if_always_same_prefix():
    """

    Print statement means it outputs to stdout probably so that it can be set in a variable.
    @return:
    """
    print(Path(sys.argv[1].replace('/home/miranda9/', '~/')))


def cluster_path_2_local_path(path_cluster: str, target_dir: str = 'data', test_mode: bool = False) -> Path:
    """
    Path from cluster to any cluster

    Note: we could have done path_cluster.replace('/home/miranda9/', '~') but that
    assumes every clsuter I use uses the same HOME path. The one I wrote looks for the path
    data and then replaces the root from local with ~ and expands users to local user to
    work locally.
    @param path_cluster:
    @param target_dir:
    @return:
    """
    dirs: list[str] = path_cluster.split('/')
    # pop everything on the front until we get to out target directory
    for dir_name in deepcopy(dirs):
        if target_dir in dir_name:  # usually 'data'
            break
        else:
            dirs.pop(0)
    # make path to local
    dirs = ['~'] + dirs
    dirs = '/'.join(dirs)
    local_dir = Path(dirs).expanduser()
    if test_mode:
        print(f'{local_dir=}')
    return local_dir


def execute_tensorboard():
    """
    DOESN'T WORK.

    test:
    python /Users/brando/ultimate-utils/ultimate-utils-proj-src/execute_tensorboard.py /home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb

    @return:
    """
    # path_cluster = sys.argv[1]
    # local_dir = cluster_path_2_local_path(path_cluster)
    # print(local_dir)
    # cmd = f'tensorboard --logdir {local_dir}'
    # print(cmd)
    # # subprocess.run(cmd)
    # os.system(cmd)
    pass


def give_path_to_local_to_bash():
    """

    @return:
    """
    # second argument sys 0th is the 0th argument to python which is the name of this script
    path_cluster = sys.argv[1]
    # remove the home path by detecting "data" and then get the path to tb locally
    local_dir = cluster_path_2_local_path(path_cluster)
    local_dir = str(local_dir)
    # -- debugging print
    # print(f'--[local dir path python made]--> {local_dir=}', flush=True)
    # -- output to bash
    print(local_dir)


# --  test

def my_test():
    """
    Need to comment out since print of script is what bash will use.

    :return:
    """
    path_cluster_dgx = '/home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb'
    path_cluster_intel = '/homes/miranda9/data/logs/logs_Dec04_15-49-00_jobid_446010.iam-pbs/tb'
    path_vision = '/home/miranda9/data/logs/logs_Dec04_18-39-14_jobid_1528/tb'
    path_vision2 = '/home/miranda9/data_fall2020_spring2021/logs/logs_Nov13_20-29-28_jobid_851/tb'
    cluster_path_2_local_path(path_cluster_dgx, test_mode=True)
    cluster_path_2_local_path(path_cluster_intel, test_mode=True)
    cluster_path_2_local_path(path_vision, test_mode=True)
    cluster_path_2_local_path(path_vision2, test_mode=True)
    # execute_tensorboard()


if __name__ == '__main__':
    # my_test()
    # execute_tensorboard()
    # -- when this file is ran, the terminal/bash gets this as the output string
    give_path_to_local_to_bash()
