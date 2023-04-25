"""
inspired from: https://github.com/brando90/diversity-for-predictive-success-of-meta-learning/blob/main/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py
"""
import os

from pprint import pprint

import time
from argparse import Namespace

# import uutils
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from learn2learn.vision.benchmarks import BenchmarkTasksets

from uutils import report_times, args_hardcoded_in_script, print_args, save_args, \
    save_to_json_pretty

# - args for each experiment
from uutils.argparse_uu.common import create_default_log_root
from uutils.argparse_uu.meta_learning import parse_args_meta_learning, fix_for_backwards_compatibility
from uutils.argparse_uu.supervised_learning import make_args_from_supervised_learning_checkpoint
from uutils.logging_uu.wandb_logging.common import cleanup_wandb, setup_wandb
from uutils.numpy_uu.common import get_diagonal, compute_moments
from uutils.plot import save_to
from uutils.plot.histograms_uu import get_histogram, get_x_axis_y_axis_from_seaborn_histogram
from uutils.torch_uu import get_device_from_model, get_device, count_number_of_parameters
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.dataloaders.common import get_dataset_size
from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
from uutils.torch_uu.distributed import is_lead_worker, set_devices
from uutils.torch_uu.metrics.diversity.diversity import \
    get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks, \
    get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks
from uutils.torch_uu.metrics.diversity.task2vec_based_metrics import task2vec, task_similarity
from uutils.torch_uu.metrics.diversity.task2vec_based_metrics.task2vec import ProbeNetwork
from uutils.torch_uu.models.probe_networks import get_probe_network

from uutils.torch_uu.metrics.confidence_intervals import mean_confidence_interval, \
    nth_central_moment_and_its_confidence_interval

from uutils.torch_uu.metrics.complexity.task2vec_norm_complexity import avg_norm_complexity, total_norm_complexity, \
    get_task_complexities, standardized_norm_complexity

# import matplotlib.pyplot as plt

from pdb import set_trace as st


# - mi

def diversity_ala_task2vec_mi_resnet18_pretrained_imagenet(args: Namespace) -> Namespace:
    # args.batch_size = 500
    args.batch_size = 2
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # args.experiment_name = f'diversity_ala_task2vec_mi_resnet18'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option} {current_time}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_mi_resnet18_random(args: Namespace) -> Namespace:
    args.batch_size = 500
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - probe_network
    args.model_option = 'resnet18_random'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_mi_resnet18'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_mi_resnet34_pretrained_imagenet(args: Namespace) -> Namespace:
    args.batch_size = 500
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - probe_network
    args.model_option = 'resnet34_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_mi_resnet34'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_mi_resnet34_random(args: Namespace) -> Namespace:
    args.batch_size = 500
    # args.batch_size = 2
    args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'lee2019'

    # - probe_network
    args.model_option = 'resnet34_random'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_mi_resnet34'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


# -cifar-fs

def diversity_ala_task2vec_cifarfs_resnet18_pretrained_imagenet(args: Namespace) -> Namespace:
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.batch_size = 500
    # args.batch_size = 3
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_cifarfs_resnet18'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_cifarfs_resnet18_random(args: Namespace) -> Namespace:
    args.batch_size = 500
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'

    # - probe_network
    args.model_option = 'resnet18_random'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_cifarfs_resnet18'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_cifarfs_resnet34_pretrained_imagenet(args: Namespace) -> Namespace:
    args.batch_size = 500
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'

    # - probe_network
    args.model_option = 'resnet34_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_cifarfs_resnet34'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_cifarfs_resnet34_random(args: Namespace) -> Namespace:
    args.batch_size = 500
    args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'rfs2020'

    # - probe_network
    args.model_option = 'resnet34_random'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_cifarfs_resnet34'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_option} {args.model_option}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


# - hdb1

def diversity_ala_task2vec_hdb1_resnet18_pretrained_imagenet(args: Namespace) -> Namespace:
    args.batch_size = 5
    args.data_option = 'hdb1'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb1'

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_hdb1_mio(args: Namespace) -> Namespace:
    # args.batch_size = 5
    args.batch_size = 500
    args.data_option = 'hdb1'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.classifier_opts = None
    args.data_augmentation = 'hdb1'

    # - probe_network
    # args.model_option = 'resnet18_random'
    # args.model_option = 'resnet18_pretrained_imagenet'
    args.model_option = 'resnet34_random'
    # args.model_option = 'resnet34_pretrained_imagenet'
    #
    # args.model_option = 'resnet18_random'
    # args.classifier_opts = dict(epochs=0)
    # args.model_option = 'resnet18_pretrained_imagenet'
    # args.classifier_opts = dict(epochs=0)

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


# - hdb2

def diversity_ala_task2vec_hdb2_cifo(args: Namespace) -> Namespace:
    args.batch_size = 500
    args.data_option = 'hdb2'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.classifier_opts = None
    args.data_augmentation = 'hdb2'

    # - probe_network
    args.model_option = 'resnet18_random'
    # args.model_option = 'resnet18_pretrained_imagenet'
    # args.model_option = 'resnet34_random'
    # args.model_option = 'resnet34_pretrained_imagenet'
    #
    # args.model_option = 'resnet18_random'
    # args.classifier_opts = dict(epochs=0)
    # args.model_option = 'resnet18_pretrained_imagenet'
    # args.classifier_opts = dict(epochs=0)

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_hdb2_resnet18_pretrained_imagenet(args: Namespace) -> Namespace:
    args.batch_size = 5
    # args.batch_size = 500
    args.data_option = 'hdb2'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb2'

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'

    # -- wandb args
    args.wandb_project = 'sl_vs_ml_iclr_workshop_paper'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.run_name = f'{args.experiment_name} {args.batch_size=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    args = fix_for_backwards_compatibility(args)
    return args


# - delaunay

def diversity_ala_task2vec_delaunay(args: Namespace) -> Namespace:
    # - data set options
    args.batch_size = 500
    args.data_option = 'delauny_uu_l2l_bm_split'
    args.data_path = Path('~/data/delauny_l2l_bm_splits').expanduser()
    args.data_augmentation = 'delauny_pad_random_resized_crop'
    args.classifier_opts = None

    # - probe_network
    # args.model_option = 'resnet18_random'
    args.model_option = 'resnet18_pretrained_imagenet'
    # args.model_option = 'resnet34_random'
    # args.model_option = 'resnet34_pretrained_imagenet'
    #
    # args.model_option = 'resnet18_random'
    # args.classifier_opts = dict(epochs=0)
    #
    # args.model_option = 'resnet18_pretrained_imagenet'
    # args.classifier_opts = dict(epochs=0)

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# - hdb4 = micod = mi + cifarfs + omniglot + delauanay

def diversity_ala_task2vec_hdb4_micod(args: Namespace) -> Namespace:
    # - data set options
    # args.batch_size = 2
    args.batch_size = 5
    # args.batch_size = 7
    # args.batch_size = 30
    args.batch_size = 500
    args.data_option = 'hdb4_micod'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = 'hdb4_micod'
    args.classifier_opts = None

    # - probe_network
    # args.model_option = 'resnet18_random'
    # args.model_option = 'resnet18_pretrained_imagenet'
    # args.model_option = 'resnet34_random'
    # args.model_option = 'resnet34_pretrained_imagenet'
    # - options for fine tuning (ft) or not
    # args.model_option = 'resnet18_random'
    # args.classifier_opts = dict(epochs=0)
    # args.model_option = 'resnet18_pretrained_imagenet'
    # args.classifier_opts = dict(epochs=0)
    if not hasattr(args, 'model_option'):
        args.model_option = 'resnet34_pretrained_imagenet'
    print(f'{args.model_option=}')

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


# - mds

def diversity_ala_task2vec_mds(args: Namespace) -> Namespace:
    # Mscoco, traffic_sign are VAL only (actually we could put them here, fixed script to be able to do so w/o crashing)
    args.sources = ['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'quickdraw', 'vgg_flower',
                    'mscoco', 'traffic_sign']

    args.batch_size = 500  # 5 for testing
    args.batch_size_eval = args.batch_size  # this determines batch size for test/eval

    # args.batch_size = 500
    args.data_option = 'mds'
    # set datapath if not already
    # args.data_path = '/shared/rsaas/pzy2/records/'

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None

    # -- wandb args
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    args.log_to_wandb = True
    # args.log_to_wandb = False

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_mds_vggaircraft(args: Namespace) -> Namespace:
    args.sources = ['aircraft', 'vgg_flower']

    args.batch_size = 500  # 5 for testing
    args.batch_size_eval = args.batch_size  # this determines batch size for test/eval

    # args.batch_size = 500
    args.data_option = 'mds'
    # set datapath if not already

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None

    # -- wandb args
    args.wandb_project = 'meta-dataset task2vec'  # 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    args.log_to_wandb = False  # True
    # args.log_to_wandb = False

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_mds_birdsdtd(args: Namespace) -> Namespace:
    args.sources = ['dtd', 'cu_birds']  # ['aircraft','vgg_flower','cu_birds']

    args.batch_size = 10  # 500  # 5 for testing
    args.batch_size_eval = args.batch_size  # this determines batch size for test/eval

    # args.batch_size = 500
    args.data_option = 'mds'
    # set datapath if not already

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None
    args.PID = 'None'

    # -- wandb args
    args.wandb_project = 'meta-dataset task2vec'  # 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_mds_ilsvrc(args: Namespace) -> Namespace:
    args.data_path = '/shared/rsaas/pzy2/records'
    args.sources = ['ilsvrc_2012']  # ['aircraft','vgg_flower','cu_birds']

    args.batch_size = 10  # 500  # 5 for testing
    args.batch_size_eval = args.batch_size  # this determines batch size for test/eval

    # args.batch_size = 500
    args.data_option = 'mds'
    # set datapath if not already

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None
    args.PID = 'None'

    # -- wandb args
    args.wandb_project = 'meta-dataset task2vec'  # 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def diversity_ala_task2vec_mds_omniglot(args: Namespace) -> Namespace:
    args.data_path = '/shared/rsaas/pzy2/records'
    args.sources = ['omniglot']  # ['aircraft','vgg_flower','cu_birds']

    args.batch_size = 10  # 500  # 5 for testing
    args.batch_size_eval = args.batch_size  # this determines batch size for test/eval

    # args.batch_size = 500
    args.data_option = 'mds'
    # set datapath if not already

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None
    args.PID = 'None'

    # -- wandb args
    args.wandb_project = 'meta-dataset task2vec'  # 'entire-diversity-spectrum'
    # - wandb expt args
    # args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.batch_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def div_hdb5(args: Namespace) -> Namespace:
    args.data_path = '/home/pzy2/data/l2l_data'#'/shared/rsaas/pzy2/records'
    #args.sources = ['aircraft']  # ['aircraft','vgg_flower','cu_birds']

    args.batch_size = 50#1000  # 500  # 5 for testing
    args.batch_size_eval = args.batch_size  # this determines batch size for test/eval

    # args.batch_size = 500
    args.data_option = 'hdb5_vggair'
    args.data_augmentation = 'hdb5_vggair'
    # set datapath if not already

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None
    args.PID = 'None'

    # -- wandb args
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'meta-dataset task2vec'  # 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    # args.log_to_wandb = True
    args.log_to_wandb = True

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args


def div_hdb6(args: Namespace) -> Namespace:
    args.data_path = '/home/pzy2/data/l2l_data'#'/shared/rsaas/pzy2/records'
    #args.sources = ['aircraft']  # ['aircraft','vgg_flower','cu_birds']

    args.batch_size = 500#1000  # 500  # 5 for testing
    args.batch_size_eval = args.batch_size  # this determines batch size for test/eval

    # args.batch_size = 500
    args.data_option = 'hdb6'
    args.data_augmentation = 'hdb4_micod'
    # set datapath if not already

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None
    args.PID = 'None'

    # -- wandb args
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'meta-dataset task2vec'  # 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    # args.log_to_wandb = True
    args.log_to_wandb = True

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args



def div_hdb7(args: Namespace) -> Namespace:
    args.data_path = '/home/pzy2/data/l2l_data'#'/shared/rsaas/pzy2/records'
    #args.sources = ['aircraft']  # ['aircraft','vgg_flower','cu_birds']

    args.batch_size = 50#0#1000  # 500  # 5 for testing
    args.batch_size_eval = args.batch_size  # this determines batch size for test/eval

    # args.batch_size = 500
    args.data_option = 'hdb7'
    args.data_augmentation = 'hdb4_micod'
    # set datapath if not already

    # - probe_network
    args.model_option = 'resnet18_pretrained_imagenet'
    args.classifier_opts = None
    args.PID = 'None'

    # -- wandb args
    args.wandb_entity = 'brando-uiuc'
    args.wandb_project = 'meta-dataset task2vec'  # 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'diversity_ala_task2vec_{args.data_option}_{args.model_option}'
    args.run_name = f'{args.experiment_name} {args.batch_size=} {args.data_augmentation=} {args.jobid} {args.classifier_opts=}'
    # args.log_to_wandb = True
    args.log_to_wandb = False#True

    from uutils.argparse_uu.meta_learning import fix_for_backwards_compatibility
    args = fix_for_backwards_compatibility(args)
    return args
# - main

def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    args: Namespace = parse_args_meta_learning()

    args.args_hardcoded_in_script = True  # <- REMOVE to remove manual loads
    # args.manual_loads_name = 'diversity_ala_task2vec_delauny'  # <- REMOVE to remove manual loads
    # args.manual_loads_name = 'diversity_ala_task2vec_mds'

    # -- set remaining args values (e.g. hardcoded, checkpoint etc.)
    print(f'{args.manual_loads_name=}')
    if resume_from_checkpoint(args):
        args: Namespace = make_args_from_supervised_learning_checkpoint(args=args, precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        args: Namespace = eval(f'{args.manual_loads_name}(args)')
    else:
        # NOP: since we are using args from terminal
        pass

    # -- Setup up remaining stuff for experiment
    # args: Namespace = setup_args_for_experiment(args)  # todo, why is this uncomented? :/
    setup_wandb(args)
    create_default_log_root(args)
    set_devices(args, verbose=True)
    return args


def main():
    # - load the args from either terminal, ckpt, manual etc.
    args: Namespace = load_args()

    # - real experiment
    compute_div_and_plot_distance_matrix_for_fsl_benchmark_for_all_splits(args)

    # - wandb
    from uutils.logging_uu.wandb_logging.common import cleanup_wandb
    cleanup_wandb(args)


def compute_div_and_plot_distance_matrix_for_fsl_benchmark_for_all_splits(args: Namespace, show_plots: bool = True):
    for probe_net in ['resnet18_pretrained_imagenet','resnet34_pretrained_imagenet']:
        print(f'\n----> {probe_net=}')
        args.model_option = probe_net
        if args.data_option == 'mds':
            splits: list[str] = ['train','val', 'test']
        else:
            splits: list[str] = ['train', 'validation', 'test']
        print_args(args)

        # -
        print(f'----> div computations...')
        splits_results = {}
        for split in splits:
            print(f'----> div computations for {split=}')
            results = compute_div_and_plot_distance_matrix_for_fsl_benchmark(args, split, show_plots)
            splits_results[split] = results

        # - print summary
        print('---- Summary of results for all splits...')
        for split in splits:
            results: dict = splits_results[split]
            div, ci, distance_matrix, split = results['div'], results['ci'], results['distance_matrix'], results['split']
            print(f'\n-> {split=}')
            print(f'Diversity: {(div, ci)=}')
            print(f'{distance_matrix=}')


def compute_div_and_plot_distance_matrix_for_fsl_benchmark(args: Namespace,
                                                           split: str = 'validation',
                                                           show_plots: bool = True,
                                                           ):
    """
    - sample one batch of tasks and use a random cross product of different tasks to compute diversity.
    """
    start = time.time()
    print(f'---- start task2vec analysis: {split=} ')

    # - print args for experiment
    save_args(args)

    # - create probe_network
    args.probe_network: ProbeNetwork = get_probe_network(args)
    print(f'{type(args.probe_network)=}')
    print(f'{get_device_from_model(args.probe_network)=}')
    print(f'{get_device()=}')
    print(f'{args.device=}')
    args.number_of_trainable_parameters = count_number_of_parameters(args.model)
    print(f'{args.number_of_trainable_parameters=}')

    # - create loader
    print(f'{args.data_augmentation=}')
    # note, you can't really detect if to use l2l with path since l2l can be converted to a torchmeta loader, for now
    # we are doing all analysis with mds or l2l (no torchmeta for now)
    # todo: idk if it works to switch a l2l normal datalaoder, also, if we use a normal data loader how would things be affected?
    # todo: idk if we want to do a USL div analysis also, it would remove the multi modes of the task2vec histograms
    if args.data_option == 'mds':
        from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
        args.dataloaders = get_meta_learning_dataloaders(args)
        print(f'{args.dataloaders=}')
        from uutils.torch_uu.metrics.diversity.diversity import get_task_embeddings_from_few_shot_dataloader
        from uutils.torch_uu.metrics.diversity.task2vec_based_metrics import task2vec, task_similarity
        embeddings: list[task2vec.Embedding] = get_task_embeddings_from_few_shot_dataloader(args,
                                                                                            args.dataloaders,
                                                                                            args.probe_network,
                                                                                            num_tasks_to_consider=args.batch_size,
                                                                                            split=split,
                                                                                            classifier_opts=args.classifier_opts,
                                                                                            )
    else:
        args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
        print(f'{args.tasksets=}')
        from uutils.torch_uu.metrics.diversity.diversity import get_task_embeddings_from_few_shot_l2l_benchmark
        from uutils.torch_uu.metrics.diversity.task2vec_based_metrics import task2vec, task_similarity
        embeddings: list[task2vec.Embedding] = get_task_embeddings_from_few_shot_l2l_benchmark(args.tasksets,
                                                                                               args.probe_network,
                                                                                               split=split,
                                                                                               num_tasks_to_consider=args.batch_size,
                                                                                               classifier_opts=args.classifier_opts,
                                                                                               )
    # print(f'{embeddings=}')
    # print(f'{embeddings[0]=}')
    print(f'\n {len(embeddings)=}')
    print(f'number of tasks to consider: {args.batch_size=}')
    print(f'{get_dataset_size(args)=}')

    # - compute complexity of benchmark (p determines which L_p norm we use to compute complexity. See task2vec_norm_complexity.py for details)
    from uutils.torch_uu.metrics.complexity.task2vec_norm_complexity import standardized_norm_complexity  # pycharm bug?
    p_norm = 1  # Set 1 for L1 norm, 2 for L2 norm, etc. 'nuc' for nuclear norm, np.inf for infinite norm
    all_complexities = get_task_complexities(embeddings, p=p_norm)
    print(f'{all_complexities=}')
    complexity_tot = total_norm_complexity(all_complexities)
    print(f'Total Complexity: {complexity_tot=}')
    complexity_avg, complexity_ci = avg_norm_complexity(all_complexities)
    print(f'Average Complexity: {(complexity_avg, complexity_ci)=}')
    standardized_norm_complexity = standardized_norm_complexity(embeddings)
    print(f'Standardized Norm Complexity: {standardized_norm_complexity=}')

    # - compute distance matrix & task2vec based diversity, to demo` task2vec, this code computes pair-wise distance between task embeddings
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    print(f'{distance_matrix=}')
    distances_as_flat_array, _, _ = get_diagonal(distance_matrix, check_if_symmetric=True)

    # - compute div
    # from uutils.torch_uu.metrics.diversity.diversity import get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks, get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks
    div_tot = float(distances_as_flat_array.sum())
    print(f'Diversity: {div_tot=}')
    div, ci = get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks(distance_matrix)
    print(f'Diversity: {(div, ci)=}')
    standardized_div: float = get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks(distance_matrix)
    print(f'Standardised Diversity: {standardized_div=}')
    # momments
    div_var, ci = nth_central_moment_and_its_confidence_interval(distances_as_flat_array, moment_idx=2)
    print(f'Diversity: {(div_var, ci)=}')
    div_std, ci = div_var ** 0.5, ci ** 0.5
    print(f'Diversity: {(div_std, ci)=}')

    # - compute central moments
    print(f'{len(distances_as_flat_array)=}')
    moment_idxs = [1, 2, 3, 4, 5, 6]
    print('-- central moments of the task2vec pair-wise distances')
    central_moments: dict = compute_moments(distances_as_flat_array, moment_idxs=moment_idxs)
    pprint(central_moments)
    print('-- moments of the task2vec pair-wise distances')
    moments: dict = compute_moments(distances_as_flat_array, moment_idxs=moment_idxs)
    pprint(moments)
    # compute size of data set
    size_dataset: int = -1
    # size_dataset = len()

    # compute div aware coeff
    from uutils.torch_uu.metrics.diversity.diversity import size_aware_div_coef_discrete_histogram_based
    effective_num_tasks, size_aware_div_coef, total_frequency, task2vec_dists_binned, frequencies_binned, num_bars_in_histogram, num_bins = size_aware_div_coef_discrete_histogram_based(
        distances_as_flat_array)
    print(
        f'{(effective_num_tasks, size_aware_div_coef, total_frequency, task2vec_dists_binned, frequencies_binned, num_bars_in_histogram, num_bins)=}')

    # - debug check, make sure num of params is right
    args.number_of_trainable_parameters = count_number_of_parameters(args.model)
    print(f'{args.number_of_trainable_parameters=}')

    # - save results
    torch.save(embeddings, args.log_root / 'embeddings.pt')  # saving obj version just in case
    results: dict = {'embeddings': [(embed.hessian, embed.scale, embed.meta) for embed in embeddings],
                     'distance_matrix': distance_matrix,
                     'div': div, 'ci': ci,
                     'standardized_div': standardized_div,
                     'split': split,
                     'effective_num_tasks': effective_num_tasks,
                     'size_aware_div_coef': size_aware_div_coef,
                     'total_frequency': total_frequency,
                     'task2vec_dists_binned': task2vec_dists_binned,
                     'frequencies_binned': frequencies_binned,
                     'num_bars_in_histogram': num_bars_in_histogram,
                     'num_bins': num_bins,
                     'size_dataset': size_dataset,
                     'complexity_tot': complexity_tot,
                     'complexity_avg': complexity_avg, 'complexity_ci': complexity_ci,
                     'all_complexities': all_complexities,
                     'standardized_norm_complexity': standardized_norm_complexity,
                     }
    torch.save(results, args.log_root / f'results_{split}.pt')
    save_to_json_pretty(results, args.log_root / f'results_{split}.json')

    # - save histograms
    title: str = 'Distribution of Task2Vec Distances'
    xlabel: str = 'Cosine Distance between Task Pairs'
    ylabel = 'Frequency Density (pmf)'
    ax = get_histogram(distances_as_flat_array, xlabel, ylabel, title, stat='probability', linestyle=None, color='b')
    save_to(args.log_root,
            plot_name=f'hist_density_task2vec_cosine_distances_{args.data_option}_{split}'.replace('-', '_'))
    ylabel = 'Frequency'
    get_histogram(distances_as_flat_array, xlabel, ylabel, title, linestyle=None, color='b')
    save_to(args.log_root,
            plot_name=f'hist_freq_task2vec_cosine_distances_{args.data_option}_{split}'.replace('-', '_'))

    # - histograms for complexity metric
    title: str = 'Distribution of Task Complexities'
    xlabel: str = f'Task Complexity (L{p_norm} norm of embedding)'
    ylabel = 'Frequency Density (pmf)'
    ax = get_histogram(all_complexities, xlabel, ylabel, title, stat='probability', linestyle=None, color='b')
    save_to(args.log_root,
            plot_name=f'hist_density_task_complexities_{args.data_option}_{split}'.replace('-', '_'))
    ylabel = 'Frequency'
    get_histogram(all_complexities, xlabel, ylabel, title, linestyle=None, color='b')
    save_to(args.log_root,
            plot_name=f'hist_freq_task_complexities_{args.data_option}_{split}'.replace('-', '_'))

    # - show plot, this code is similar to above but put computes the distance matrix internally & then displays it, hierchical clustering
    task_similarity.plot_distance_matrix(embeddings, labels=list(range(len(embeddings))), distance='cosine',
                                         show_plot=False)
    save_to(args.log_root, plot_name=f'clustered_distance_matrix_fsl_{args.data_option}_{split}'.replace('-', '_'))
    import matplotlib.pyplot as plt
    if show_plots:
        # plt.show()
        pass
    # heatmap
    task_similarity.plot_distance_matrix_heatmap_only(embeddings, labels=list(range(len(embeddings))),
                                                      distance='cosine',
                                                      show_plot=False)
    save_to(args.log_root, plot_name=f'heatmap_only_distance_matrix_fsl_{args.data_option}_{split}'.replace('-', '_'))
    if show_plots:
        # plt.show()
        pass

    # todo: log plot to wandb https://docs.wandb.ai/guides/track/log/plots, https://stackoverflow.com/questions/72134168/how-does-one-save-a-plot-in-wandb-with-wandb-log?noredirect=1&lq=1
    # import wandb
    # wandb.log({"chart": plt})
    # -
    # print(f"\n---- {report_times(start)}\n")
    return results


if __name__ == '__main__':
    import time

    start = time.time()
    # - run experiment
    main()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
