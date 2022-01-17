from argparse import Namespace
from typing import Optional

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.distributed import move_to_ddp, is_running_parallel
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
from uutils.torch_uu.optim_uu.adafactor_uu import get_default_adafactor_opt_fairseq_and_hps_dict, \
    get_default_adafactor_scheduler_fairseq_and_hps_dict


def get_and_create_model_opt_scheduler_first_time(args: Namespace,
                                                  model_option: Optional[str] = None,
                                                  opt_option: Optional[str] = None,
                                                  scheduler_option: Optional[str] = None,
                                                  ) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    if resume_from_checkpoint(args):
        get_and_create_model_opt_scheduler_from_ckpt(args)
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
        args.model, args.model_hps_for_cons_dict = get_default_learner_and_hps_dict()
        # args.model, args.model_hps_for_cons_dict = get_default_learner_and_hps_dict(in_channels=1)
    elif 'resnet' in model_option and 'rfs' in model_option:
        # args.k_eval = 30
        from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model
        args.model, args.model_hps_for_cons_dict = get_resnet_rfs_model(args.model_option)
    else:
        raise ValueError(f'Model option given not found: got {model_option=}')
    args.model = move_to_ddp(args.rank, args, args.model)
    args.model.to(args.device)
    args.model = move_to_ddp(args.model) if is_running_parallel(args.rank) else args.model

    # - get optimizer
    opt_option: str = args.opt_option if opt_option is None else opt_option
    if opt_option == 'AdafactorDefaultFair':
        args.opt, args.opt_hps = get_default_adafactor_opt_fairseq_and_hps_dict(args.model)
    else:
        raise ValueError(f'Optimizer option is invalid: got {opt_option=}')

    # - get scheduler
    scheduler_option: str = args.scheduler_option if scheduler_option is None else scheduler_option
    if scheduler_option == 'None':  # None != 'None', obj None means use the ckpt value
        args.scheduler = None
    elif scheduler_option == 'AdafactorSchedule':
        args.scheduler, args.scheduler_hps = get_default_adafactor_scheduler_fairseq_and_hps_dict(args.opt)
    else:
        raise ValueError(f'Scheduler option is invalid: got {scheduler_option=}')
    return args.model, args.opt, args.scheduler


def get_and_create_model_opt_scheduler_from_ckpt(args: Namespace,
                                                 model_option: Optional[str] = None,
                                                 opt_option: Optional[str] = None,
                                                 scheduler_option: Optional[str] = None,
                                                 ) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    """

    note:
        - design pattern: when creating models, saving the params is not enough since we have to create the empty arch
        first then load with the params. Thus, we need to the saved hps to create the saved arch and then load the
        params to it.
    """
    # - get model the empty model from the hps for the cons for the model
    model_option: str = args.model_option if model_option is None else model_option  # if obj None, use ckpt value
    if model_option == '5CNN_opt_as_model_for_few_shot_sl':
        args.model, _ = get_default_learner_and_hps_dict(**args.model_hps_for_cons_dict)
    elif 'resnet' in model_option and 'rfs' in model_option:
        from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model
        args.model, _ = get_resnet_rfs_model(args.model_option, **args.model_hps_for_cons_dict)
    else:
        raise ValueError(f'Model option given not found: got {model_option=}')
    args.model = move_to_ddp(args.rank, args, args.model)
    args.model.to(args.device)
    args.model = move_to_ddp(args.model) if is_running_parallel(args.rank) else args.model

    # - get optimizer
    opt_option: str = args.opt_option if opt_option is None else opt_option
    if opt_option == 'AdafactorDefaultFair':
        args.opt, args.opt_hps = get_default_adafactor_opt_fairseq_and_hps_dict(args.model, **args.opt_hps)
    else:
        raise ValueError(f'Optimizer option is invalid: got {opt_option=}')

    # - get scheduler
    scheduler_option: str = args.scheduler_option if scheduler_option is None else scheduler_option
    if scheduler_option == 'None':  # None != 'None', obj None means use the ckpt value
        args.scheduler = None
    elif scheduler_option == 'AdafactorSchedule':
        args.scheduler, args.scheduler_hps = get_default_adafactor_scheduler_fairseq_and_hps_dict(args.opt,
                                                                                                  **args.scheduler_hps)
    else:
        raise ValueError(f'Scheduler option is invalid: got {scheduler_option=}')
    return args.model, args.opt, args.scheduler