"""

Note:
    - the move_to_dpp moves things .to(device)
"""
from argparse import Namespace
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.distributed import move_to_ddp
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
from uutils.torch_uu.optim_uu.adafactor_uu import get_default_adafactor_opt_fairseq_and_hps_dict, \
    get_default_adafactor_scheduler_fairseq_and_hps_dict

from pdb import set_trace as st


def get_and_create_model_opt_scheduler(args: Namespace) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    if resume_from_checkpoint(args):
        load_model_optimizer_scheduler_from_ckpt(args)
    else:
        model, opt, scheduler = get_and_create_model_opt_scheduler_first_time(args)
    return model, opt, scheduler


def get_and_create_model_opt_scheduler_first_time(args: Namespace) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    # note: hps are empty dicts, so it uses defaults
    model, opt, scheduler = get_and_create_model_opt_scheduler(args)
    return model, opt, scheduler


def load_model_optimizer_scheduler_from_ckpt(args: Namespace) -> tuple[
    nn.Module, Optimizer, _LRScheduler]:
    """
    Load the most important things: model, optimizer, scheduler.

    Ref:
        - standard way: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    """
    # - prepare args from ckpt
    # ckpt: dict = torch.load(args.path_to_checkpoint, map_location=torch.device('cpu'))
    ckpt: dict = torch.load(args.path_to_checkpoint, map_location=args.device)
    model_option = ckpt['model_option']
    model_hps = ckpt['model_hps']

    opt_option = ckpt['opt_option']
    opt_hps = ckpt['opt_hps']

    scheduler_option = ckpt['scheduler_option']
    scheduler_hps = ckpt['scheduler_hps']
    get_and_create_model_opt_scheduler(args,
                                       model_option,
                                       model_hps,

                                       opt_option,
                                       opt_hps,

                                       scheduler_option,
                                       scheduler_hps,
                                       )

    # - load state dicts
    model_state_dict: dict = ckpt['model_state_dict']
    args.model.load_state_dict(model_state_dict)
    opt_state_dict: dict = ckpt['opt_state_dict']
    args.opt.load_state_dict(opt_state_dict)
    if hasattr(args.scheduler, 'load_state_dict'):
        scheduler_state_dict: dict = ckpt['scheduler_state_dict']
        args.scheduler.load_state_dict(scheduler_state_dict)

    # - load last step (it or epoch_num)
    args_from_ckpt = Namespace(**ckpt['args_dict'])
    if args_from_ckpt.training_mode == 'epochs':
        args.epoch_num = ckpt['epoch_num']
    else:
        args.it = ckpt['it']
    # assert next(mdl.parameters()).device == args.device
    return args.model, args.opt, args.scheduler


def get_and_create_model_opt_scheduler(args: Namespace,

                                       model_option: Optional[str] = None,
                                       model_hps: dict = {},

                                       opt_option: Optional[str] = None,
                                       opt_hps: dict = {},

                                       scheduler_option: Optional[str] = None,
                                       scheduler_hps: dict = {},

                                       ) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    """
    Creates for the first time the model, opt, scheduler needed for the experiment run in main.
    """
    # - get model the empty model from the hps for the cons for the model
    model_option: str = args.model_option if model_option is None else model_option  # if obj None, use ckpt value
    if model_option == '5CNN_opt_as_model_for_few_shot_sl':
        args.model, args.model_hps = get_default_learner_and_hps_dict(**model_hps)
    elif model_option == 'resnet12_rfs_mi' or model_option == 'resnet12_rfs':  # resnet12_rfs for backward compat
        from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_mi
        args.model, args.model_hps = get_resnet_rfs_model_mi(args.model_option, **model_hps)
    elif model_option == 'resnet12_rfs_cifarfs_fc100':
        from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_cifarfs_fc100
        args.model, args.model_hps = get_resnet_rfs_model_cifarfs_fc100(args.model_option, **model_hps)
    else:
        raise ValueError(f'Model option given not found: got {model_option=}')
    args.model = move_to_ddp(args.rank, args, args.model)

    # - get optimizer
    opt_option: str = args.opt_option if opt_option is None else opt_option
    if opt_option == 'AdafactorDefaultFair':
        args.opt, args.opt_hps = get_default_adafactor_opt_fairseq_and_hps_dict(args.model, **opt_hps)
    elif opt_option == 'Adam_rfs_cifarfs':
        from uutils.torch_uu.optim_uu.adam_uu import get_opt_adam_rfs_cifarfs
        args.opt, args.opt_hps = get_opt_adam_rfs_cifarfs(args.model, **opt_hps)
    else:
        raise ValueError(f'Optimizer option is invalid: got {opt_option=}')

    # - get scheduler
    scheduler_option: str = args.scheduler_option if scheduler_option is None else scheduler_option
    if scheduler_option == 'None':  # None != 'None', obj None means use the ckpt value
        args.scheduler = None
    elif scheduler_option == 'AdafactorSchedule':
        args.scheduler, args.scheduler_hps = get_default_adafactor_scheduler_fairseq_and_hps_dict(args.opt,
                                                                                                  **scheduler_hps)
    elif scheduler_option == 'Adam_cosine_scheduler_rfs_cifarfs':
        from uutils.torch_uu.optim_uu.adam_uu import get_cosine_scheduler_adam_rfs_cifarfs
        args.scheduler, args.scheduler_hps = get_cosine_scheduler_adam_rfs_cifarfs(args.opt, **scheduler_hps)
    else:
        raise ValueError(f'Scheduler option is invalid: got {scheduler_option=}')
    return args.model, args.opt, args.scheduler
