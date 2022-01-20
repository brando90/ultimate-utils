from argparse import Namespace
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.distributed import move_to_ddp, is_running_parallel
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
from uutils.torch_uu.optim_uu.adafactor_uu import get_default_adafactor_opt_fairseq_and_hps_dict, \
    get_default_adafactor_scheduler_fairseq_and_hps_dict

from pdb import set_trace as st

def get_and_create_model_opt_scheduler(args: Namespace,
                                       model_option: Optional[str] = None,
                                       opt_option: Optional[str] = None,
                                       scheduler_option: Optional[str] = None,
                                       ) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    if resume_from_checkpoint(args):
        load_model_optimizer_scheduler_from_ckpt_given_model_type(args)
    else:
        get_and_create_model_opt_scheduler_first_time(args,
                                                      model_option,
                                                      opt_option,
                                                      scheduler_option)


def get_and_create_model_opt_scheduler_first_time(args: Namespace,
                                                  model_option: Optional[str] = None,
                                                  opt_option: Optional[str] = None,
                                                  scheduler_option: Optional[str] = None,
                                                  ) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    """
    Creates for the first time the model, opt, scheduler needed for the experiment run in main.
    """
    # - get model the empty model from the hps for the cons for the model
    model_option: str = args.model_option if model_option is None else model_option  # if obj None, use ckpt value
    if model_option == '5CNN_opt_as_model_for_few_shot_sl':
        args.model, args.model_hps = get_default_learner_and_hps_dict()
    elif model_option == 'resnet12_rfs_mi' or model_option == 'resnet12_rfs':  # resnet12_rfs for backward compat
        from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_mi
        args.model, args.model_hps = get_resnet_rfs_model_mi(args.model_option)
    elif model_option == 'resnet12_rfs_cifarfs_fc100':
        from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_cifarfs_fc100
        args.model, args.model_hps = get_resnet_rfs_model_cifarfs_fc100(args.model_option)
    else:
        raise ValueError(f'Model option given not found: got {model_option=}')
    args.model = move_to_ddp(args.rank, args, args.model)

    # - get optimizer
    opt_option: str = args.opt_option if opt_option is None else opt_option
    if opt_option == 'AdafactorDefaultFair':
        args.opt, args.opt_hps = get_default_adafactor_opt_fairseq_and_hps_dict(args.model)
    elif opt_option == 'Adam_rfs_cifarfs':
        from uutils.torch_uu.optim_uu.adam_uu import get_opt_adam_rfs_cifarfs
        args.opt, args.opt_hps = get_opt_adam_rfs_cifarfs(args.model)
    else:
        raise ValueError(f'Optimizer option is invalid: got {opt_option=}')

    # - get scheduler
    scheduler_option: str = args.scheduler_option if scheduler_option is None else scheduler_option
    if scheduler_option == 'None':  # None != 'None', obj None means use the ckpt value
        args.scheduler = None
    elif scheduler_option == 'AdafactorSchedule':
        args.scheduler, args.scheduler_hps = get_default_adafactor_scheduler_fairseq_and_hps_dict(args.opt)
    elif scheduler_option == 'Adam_cosine_scheduler_rfs_cifarfs':
        from uutils.torch_uu.optim_uu.adam_uu import get_cosine_scheduler_adam_rfs_cifarfs
        args.scheduler, args.scheduler_hps = get_cosine_scheduler_adam_rfs_cifarfs(args.opt)
    else:
        raise ValueError(f'Scheduler option is invalid: got {scheduler_option=}')
    print(f'{args.model.cls.weight.device=}')
    return args.model, args.opt, args.scheduler


def load_model_optimizer_scheduler_from_ckpt_given_model_type(args: Namespace,
                                                              model_option: Optional[str] = None,
                                                              opt_option: Optional[str] = None,
                                                              scheduler_option: Optional[str] = None,
                                                              mutate_args: bool = True
                                                              ) \
        -> tuple[nn.Module, Optimizer, _LRScheduler]:
    """
    Load the standard most important things: model, optimizer, scheduler.

    Note:
        - for X_option if obj None then it means use ckpt value, if 'None' for scheduler means use no scheduler. Else,
        use checkpoint value as default.
        - Note, mutating args, is necessary, since the model, opt, scheduler objects were not saved in the ckpt, so
        to recover them we need to place them in args. Its ugly but seems like a neccessary evil.

    Ref:
        - standard way: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    """
    # - note, this shouldn't be needed since the args at this point should the one from the ckpt, but to be safe we
    # load the ckpt anyway. Note, we could have also, loaded the args in the ckpt and use that instead, but some values
    # are extra important, thus they are saved explicitly (and in the saved args_dict in the ckpt).
    # ckpt: dict = torch.load(args.path_to_checkpoint, map_location=torch.device('cpu'))
    ckpt: dict = torch.load(args.path_to_checkpoint, map_location=args.device)

    # - get model the empty model from the hps for the cons for the model
    model_option: str = args.model_option if model_option is None else model_option  # if obj None, use ckpt value
    if model_option == '5CNN_opt_as_model_for_few_shot':
        from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import load_model_5CNN_opt_as_model_for_few_shot
        model_hps: dict = ckpt['model_hps']
        model: nn.Module = load_model_5CNN_opt_as_model_for_few_shot(model_hps)
    elif 'resnet' in model_option and 'rfs' in model_option:
        raise NotImplementedError
    else:
        raise ValueError(f'Model option given not found: got {model_option=}')
    # load state dict for the model
    model_state_dict: dict = ckpt['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.to(args.device)
    model = move_to_ddp(model) if is_running_parallel(args.rank) else model

    # - get optimizer
    opt_option: str = args.opt_option if opt_option is None else opt_option
    if opt_option == 'AdafactorDefaultFair':
        from uutils.torch_uu.optim_uu.adafactor_uu import \
            load_uutils_default_adafactor_and_scheduler_fairseq_and_hps_dict
        opt_hps_for_cons_dict: dict = ckpt['opt_hps_for_cons_dict']
        scheduler_hps_for_cons_dict: dict = ckpt['scheduler_hps_for_cons_dict']
        opt, _ = load_uutils_default_adafactor_and_scheduler_fairseq_and_hps_dict(model,
                                                                                  opt_hps_for_cons_dict,
                                                                                  scheduler_hps_for_cons_dict)
    else:
        raise ValueError(f'Optimizer option is invalid: got {opt_option=}')
    # load state dict for the optimizer
    opt_state_dict: dict = ckpt['opt_state_dict']
    opt.load_state_dict(opt_state_dict)

    # - get scheduler
    if scheduler_option == 'None':  # None != 'None', obj None means use the ckpt value
        scheduler: None = None
    elif scheduler_option == 'AdafactorSchedule':
        from uutils.torch_uu.optim_uu.adafactor_uu import \
            load_uutils_default_adafactor_and_scheduler_fairseq_and_hps_dict
        opt_hps_for_cons_dict: dict = ckpt['opt_hps_for_cons_dict']
        scheduler_hps_for_cons_dict: dict = ckpt['scheduler_hps_for_cons_dict']
        _, scheduler = load_uutils_default_adafactor_and_scheduler_fairseq_and_hps_dict(model,
                                                                                        opt_hps_for_cons_dict,
                                                                                        scheduler_hps_for_cons_dict)
    else:
        raise ValueError(f'Scheduler option is invalid: got {scheduler_option=}')
    # load state dict for the scheduler
    if hasattr(scheduler, 'load_state_dict'):
        scheduler_state_dict: dict = ckpt['scheduler_state_dict']
        scheduler.load_state_dict(scheduler_state_dict)

    # - return & mutate args
    if mutate_args:
        args.model = model
        args.opt = opt
        args.scheduler = scheduler
    return model, opt, scheduler
