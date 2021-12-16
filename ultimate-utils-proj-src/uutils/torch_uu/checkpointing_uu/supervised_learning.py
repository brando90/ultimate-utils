"""
Note: you are recommended to follow this
https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
and likely avoid dill due to possible path to objects changes.

Refs:
    - for DDP checkpointing see: https://stackoverflow.com/questions/70386800/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data
"""
from argparse import Namespace
from typing import Optional

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import uutils
from uutils.torch_uu.checkpointing_uu import try_to_get_scheduler_state_dict
from uutils.torch_uu.distributed import is_lead_worker, get_model_from_ddp, is_running_parallel, move_to_ddp


def save_for_supervised_learning(args: Namespace, ckpt_filename: str = 'ckpt.pt'):
    """
    Save a model checkpoint now (if we are the lead process).

    Implementation details/comments:
    - since you need to create an instance of the model before loading it, you need to save the hyperparameters
    that go into the constructor of the object.
    - when you choose a specific model_option, opt_option, scheduler_option it comes with the very specific
    hyperparameters needed to construct an empty model. One saves that dict and the uses that to construct
    an empty model and then later load it's parameters.

    Warning:
        - if you save with dill but save the actual objects, your this dill unpickling will likely
         needs to know the paths to the code to load the class correctly. So it's safer to load the data e.g.
         weights, biases, optimizer state and then load the data into the objects when YOU instantiate those objects.

    Ref:
        - standard way: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        - https://stackoverflow.com/questions/70129895/why-is-it-not-recommended-to-save-the-optimizer-model-etc-as-pickable-dillable
        - DDP checkpointing: https://stackoverflow.com/questions/70386800/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data
    """
    if is_lead_worker(args.rank):
        # import dill
        import pickle
        args.logger.save_current_plots_and_stats()

        # - ckpt
        args_pickable: Namespace = uutils.make_args_pickable(args)
        # note not saving any objects, to make sure checkpoint is loadable later with no problems
        torch.save({'training_mode': args.training_mode,
                    'it': args.it,
                    'epoch_num': args.epoch_num,

                    # 'args': args_pickable,  # some versions of this might not have args!
                    # decided only to save the dict version to avoid this ckpt not working, make it args when loading
                    'args_dict': vars(args_pickable),  # some versions of this might not have args!

                    'model_state_dict': get_model_from_ddp(args.model).state_dict(),
                    # added later, to make it easier to check what optimizer was used
                    'model_str': str(args.model),  # added later, to make it easier to check what optimizer was used
                    'model_hps_for_cons_dict': args.model_hps_for_cons_dict,
                    # to create an empty new instance when loading model
                    'model_option': args.model_option,

                    'opt_state_dict': args.opt.state_dict(),
                    'opt_str': str(args.opt),
                    'opt_hps_for_cons_dict': args.opt_hps_for_cons_dict,
                    'opt_option': args.opt_option,

                    'scheduler_str': str(args.scheduler),
                    'scheduler_state_dict': try_to_get_scheduler_state_dict(args.scheduler),
                    'scheduler_hps_for_cons_dict': args.scheduler_hps_for_cons_dict,
                    'scheduler_option': args.scheduler_option,
                    },
                   # pickle_module=dill,
                   pickle_module=pickle,
                   f=args.log_root / ckpt_filename)


def load_model_optimizer_scheduler_from_checkpoint_given_model_type(args: Namespace,
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
        model_hps_for_cons_dict: dict = ckpt['model_hps_for_cons_dict']
        model: nn.Module = load_model_5CNN_opt_as_model_for_few_shot(model_hps_for_cons_dict)
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

    # - return & perhaps mutate args
    if mutate_args:
        args.model = model
        args.opt = opt
        args.scheduler = scheduler
    return model, opt, scheduler


# todo - make a folder for it in models uutils and put this at the bottom
# def load_model_resnet12_rfs(args: Namespace) -> nn.Module:
#     pass
#     # - get the hps of the model & build the instance
#
#     # - load the parameters into the empty instance
#
#     # - return the loaded instance
#     # return model
