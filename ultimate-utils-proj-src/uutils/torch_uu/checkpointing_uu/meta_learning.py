from argparse import Namespace
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler

import uutils
from uutils.torch_uu.checkpointing_uu import try_to_get_scheduler_state_dict
from uutils.torch_uu.distributed import is_lead_worker, get_model_from_ddp


def save_for_meta_learning(args: Namespace, ckpt_filename: str = 'ckpt.pt'):
    """
    Warning:
        - if you save with dill but save the actual objects, this dill unpickling will likely
         needs to know the paths to the code to load the class correctly. So it's safer to load the data e.g.
         weights, biases, optimizer state and then load the data into the objects when YOU instantiate those objects.

    ref:
        - https://stackoverflow.com/questions/70129895/why-is-it-not-recommended-to-save-the-optimizer-model-etc-as-pickable-dillable
    """
    if is_lead_worker(args.rank):
        import pickle
        args.logger.save_current_plots_and_stats()

        # - ckpt
        args_pickable: Namespace = uutils.make_args_pickable(args)
        torch.save({'training_mode': args.training_mode,
                    'it': args.it,
                    'epoch_num': args.epoch_num,

                    # 'args': args_pickable,  # some versions of this might not have args!
                    # decided only to save the dict version to avoid this ckpt not working, make it args when loading
                    'args_dict': vars(args_pickable),  # some versions of this might not have args!

                    'agent_type': type(args.agent),

                    'model_state_dict': get_model_from_ddp(args.model).state_dict(),
                    'model_str': str(args.model),  # added later, to make it easier to check what optimizer was used
                    'model_hps': args.model_hps,
                    'model_option': args.model_option,

                    'opt_state_dict': args.opt.state_dict(),
                    'opt_str': str(args.opt),
                    'opt_hps': args.opt_hps,
                    'opt_option': args.opt_option,

                    'scheduler_str': str(args.scheduler),
                    'scheduler_state_dict': try_to_get_scheduler_state_dict(args.scheduler),
                    'scheduler_hps': args.scheduler_hps,
                    'scheduler_option': args.scheduler_option,
                    },
                   pickle_module=pickle,
                   f=args.log_root / ckpt_filename)


# --

def get_model_opt_meta_learner_to_resume_checkpoint_resnets_rfs(args: Namespace,
                                                                path2ckpt: str,
                                                                filename: str,
                                                                device: Optional[torch.device] = None,
                                                                # precedence_to_args_checkpoint: bool = True,
                                                                ) \
        -> tuple[nn.Module, optim.Optimizer, _LRScheduler, object]:
    """
    Get the model, optimizer, meta_learner to resume training from checkpoint.

    Examples:
        - see: _resume_from_checkpoint_meta_learning_for_resnets_rfs_test

    ref:
        - https://stackoverflow.com/questions/70129895/why-is-it-not-recommended-to-save-the-optimizer-model-etc-as-pickable-dillable
    """
    import uutils

    path2ckpt: Path = Path(path2ckpt).expanduser() if isinstance(path2ckpt, str) else path2ckpt.expanduser()
    ckpt: dict = torch.load(path2ckpt / filename, map_location=torch.device('cpu'))

    # - args
    # args_ckpt: Namespace = ckpt['args']
    # if args_ckpt is not None:
    #     if precedence_to_args_checkpoint:
    #         args: Namespace = uutils.merge_args(starting_args=args, updater_args=args_ckpt)
    #     else:
    #         args: Namespace = uutils.merge_args(starting_args=args_ckpt, updater_args=args)

    # - get training mode (this makes sense because even training a single batch can be mode)
    training_mode = ckpt.get('training_mode', None)  # Return the value for key if key is in dict, else default None.
    if training_mode is not None:
        assert uutils.xor(training_mode == 'epochs', training_mode == 'iterations')
        if training_mode == 'epochs':
            args.epoch_num = ckpt['epoch_num']
        else:
            args.it = ckpt['it']  # note even if batch_idx is set, .it is the actual value to track the iteration num

    # - get meta-learner
    meta_learner = ckpt['meta_learner']

    # - get model
    model: nn.Module = meta_learner.base_model

    # - get outer-opt
    # outer_opt_str = ckpt.get('outer_opt_str', None)
    # if outer_opt_str is not None:
    #     # use the string to create optimizer, load the state dict, etc.
    #     outer_opt: optim.Optimizer = get_optimizer(outer_opt_str)
    #     outer_opt_state_dict: dict = ckpt['outer_opt_state_dict']
    #     outer_opt.load_state_dict(outer_opt_state_dict)
    # else:
    #     # this is not ideal, but since Adam has an exponentially moving average for its adaptive learning rate,
    #     # hopefully this doesn't screw my checkpoint to much
    #     outer_opt: optim.Optimizer = optim.Adam(model.parameters(), lr=args.outer_lr)

    # - scheduler
    # scheduler = ckpt.get('scheduler', None)  # Return the value for key if key is in dict, else default None.
    # outer_opt = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    # scheduler = AdafactorSchedule(args.outer_opt)

    # - overwritting the optimizer since, we accidentally didn't save it in previous checkpoints
    # scheduler_opt = None
    # scheduler_opt = 'adafactor_scheduler_from_huggingface'
    # outer_opt, scheduler = get_uutils_default_adafactor_from_torch_optimizer_and_scheduler_default(mdl=model, lr=1e-3, scheduler_opt=scheduler_opt)
    # outer_opt, scheduler = get_uutils_default_adafactor_from_torch_optimizer_and_scheduler_default(mdl=model, lr=1e-4, scheduler_opt=scheduler_opt)
    # outer_opt, scheduler = get_uutils_default_adafactor_from_torch_optimizer_and_scheduler_default(mdl=model, lr=1e-5, scheduler_opt=scheduler_opt)

    # scheduler_opt = None
    # outer_opt, scheduler = get_default_uu_adafactor_optimizer_and_scheduler_fairseq(mdl=model, lr=None, scheduler_opt=scheduler_opt)
    # outer_opt, scheduler = get_default_uu_adafactor_optimizer_and_scheduler_fairseq(mdl=model, lr=1e-3, scheduler_opt=scheduler_opt)
    # outer_opt, scheduler = get_default_uu_adafactor_optimizer_and_scheduler_fairseq(mdl=model, lr=1e-4, scheduler_opt=scheduler_opt)
    # outer_opt, scheduler = get_default_uu_adafactor_optimizer_and_scheduler_fairseq(mdl=model, lr=1e-5, scheduler_opt=scheduler_opt)

    from uutils.torch_uu import get_default_uu_adafactor_optimizer_and_scheduler_fairseq

    scheduler_opt = 'adafactor_scheduler_from_huggingface'
    # outer_opt, scheduler = get_default_uu_adafactor_optimizer_and_scheduler_fairseq(mdl=model, lr=None, scheduler_opt=scheduler_opt)
    # outer_opt, scheduler = get_default_uu_adafactor_optimizer_and_scheduler_fairseq(mdl=model, lr=1e-3, scheduler_opt=scheduler_opt)
    # outer_opt, scheduler = get_default_uu_adafactor_optimizer_and_scheduler_fairseq(mdl=model, lr=1e-4, scheduler_opt=scheduler_opt)
    outer_opt, scheduler = get_default_uu_adafactor_optimizer_and_scheduler_fairseq(mdl=model, lr=1e-5,
                                                                                    scheduler_opt=scheduler_opt)

    # - device setup
    if device is not None:
        meta_learner.base_model.to(device)
        meta_learner.to(device)

    # - put values in args
    # args.base_model = model
    # args.outer_opt = outer_opt
    # args.meta_learner = meta_learner
    # args.scheduler = scheduler
    return model, outer_opt, scheduler, meta_learner


def get_optimizer(optimizer_name: str) -> optim.Optimizer:
    raise ValueError('Not implemented')
