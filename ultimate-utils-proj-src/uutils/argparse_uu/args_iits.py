from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from uutils import load_cluster_jobids_to, merge_args


def parse_args_iit() -> Namespace:
    import argparse

    # - great terminal argument parser
    parser = argparse.ArgumentParser()

    # -- create argument options
    # parser.add_argument('--manual_loads_name', type=str, default='None')
    parser.add_argument('--path_to_save_new_dataset', type=str, default='~/data/debug_proj/', help="Example value: "
                                                                                                   "'~/data/pycoq_lf_debug/'"
                                                                                                   "'~/data/debug_proj/'"
                                                                                                   "'~/data/compcert/'"
                                                                                                   "'~/data/coqgym/'"
                        )
    parser.add_argument('--split', type=str, default=None, help='Example values: "train", "test".')
    parser.add_argument('--save_flatten_data_set_as_single_json_file', action='store_true', default=False)

    # - parse arguments
    args: Namespace = parser.parse_args()
    # - load cluster ids so that wandb can use it later for naming runs, experiments, etc.
    # load_cluster_jobids_to(args)
    return args
