"""
Replaces te prefix with the home path in the cluster with the local so that running tensorboard is seammingless.

sh /Users/brando/ultimate-utils/run_tb.sh /home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb
tbb /home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb
alias tbb='sh /Users/brando/ultimate-utils/run_tb.sh'
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

def cluster_path_2_local_path(path_cluster, target_dir='data'):
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
    dirs = path_cluster.split('/')
    # pop everything on the front until we get to out target directory
    for dir_name in deepcopy(dirs):
        if dir_name == target_dir:  # usually 'data'
            break
        else:
            dirs.pop(0)
    # make path to local
    dirs = ['~'] + dirs
    dirs = '/'.join(dirs)
    local_dir = Path(dirs).expanduser()
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
    print(local_dir)

# --  test

def test():
    path_cluster_dgx = '/home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb'
    path_cluster_intel = '/homes/miranda9/data/logs/logs_Dec04_15-49-00_jobid_446010.iam-pbs/tb'
    path_vision = '/home/miranda9/data/logs/logs_Dec04_18-39-14_jobid_1528/tb'
    cluster_path_2_local_path(path_cluster_dgx)
    cluster_path_2_local_path(path_cluster_intel)
    cluster_path_2_local_path(path_vision)
    # execute_tensorboard()

if __name__ == '__main__':
    # test()
    # execute_tensorboard()
    give_path_to_local_to_bash()
