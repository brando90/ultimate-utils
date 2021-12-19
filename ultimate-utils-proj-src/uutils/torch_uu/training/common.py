"""

# - scheduling rate tips
- Do "But the most commonly used method is when the validation loss does not improve for a few epochs." according to https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
- for epochs it seems every epoch step is common: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

ref:
    - https://forums.pytorchlightning.ai/t/what-is-the-recommended-or-most-common-practices-to-using-a-scheduler/1416

"""
from argparse import Namespace
from typing import Any

import progressbar
import transformers.optimization
from progressbar import ProgressBar
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.optimization import AdafactorSchedule


def get_trainer_progress_bar(args: Namespace) -> ProgressBar:
    import uutils
    if args.training_mode == 'fit_single_batch':
        bar: ProgressBar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    elif args.training_mode == 'iterations':
        bar: ProgressBar = uutils.get_good_progressbar(max_value=args.num_its)
    elif args.training_mode == 'epochs':
        bar: ProgressBar = uutils.get_good_progressbar(max_value=args.num_epochs)
    elif args.training_mode == 'iterations_train_convergence':
        bar: ProgressBar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    elif args.training_mode == 'epochs_train_convergence':
        bar: ProgressBar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    else:
        raise ValueError(f'Invalid training_mode value, got: {args.training_mode}')
    return bar


def check_halt(args: Namespace) -> bool:
    """
    Check if we should halt depending on training mode.

    For logic for idx:
    If idx + 1 >= n then we are done. idx is zero index, so after the end of the 0th loop's actual logic (e.g. train_step)
    we have done 1 step although the index is zero. So we need to only check for idx + 1 >= n.
    """
    if args.training_mode == 'fit_single_batch':
        # halt: bool = train_acc >= acc_tolerance and train_loss <= train_loss_tolerance
        # return converge cirterion
    elif args.training_mode == 'iterations':
        #  idx + 1 == n, this means you've done n loops where idx = it or epoch_num
        halt: bool = args.it + 1 >= args.num_its
    elif args.training_mode == 'epochs':
        #  idx + 1 == n, this means you've done n loops where idx = it or epoch_num
        halt: bool = args.epoch_num + 1 >= args.num_epochs
    elif args.training_mode == 'iterations_train_convergence':
    elif args.training_mode == 'epochs_train_convergence':
    else:
        raise ValueError(f'Invalid training_mode value, got: {args.training_mode}')
    return halt


def gradient_clip(args, meta_opt):
    """Do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖

    depending on args it does it per parameter or all parameters together.

    Note:
        - don't do this if you're using Adafactor.

    Arguments:
        args {Namespace} -- arguments for experiment
        meta_opt {Optimizer} -- optimizer that train the meta-learner

    Raises:
        ValueError: For invalid arguments to args.grad_clip_mode
    """
    # do gradient clipping: If ‖g‖ ≥ c Then g := c * g/‖g‖
    # note: grad_clip_rate is a number for clipping the other is the type
    # of clipping we are doing
    if hasattr(args, 'grad_clip_rate'):
        if args.grad_clip_rate is not None:
            if args.grad_clip_mode == 'clip_all_seperately':
                for group_idx, group in enumerate(meta_opt.param_groups):
                    for p_idx, p in enumerate(group['params']):
                        nn.utils.clip_grad_norm_(p, args.grad_clip_rate)
            elif args.grad_clip_mode == 'clip_all_together':
                # [y for x in list_of_lists for y in x]
                all_params = [p for group in meta_opt.param_groups for p in group['params']]
                nn.utils.clip_grad_norm_(all_params, args.grad_clip_rate)
            elif args.grad_clip_mode == 'no_grad_clip' or args.grad_clip_mode is None:  # i.e. do not clip if grad_clip_rate is None
                pass
            else:
                raise ValueError(f'Invalid, args.grad_clip_mode = {args.grad_clip_mode}')


def scheduler_step(args: Namespace):
    if not hasattr(args, 'scheduler'):
        return
    else:
        # args.scheduler.step() if (args.scheduler is not None) else None
        if args.scheduler is None:
            return
        elif isinstance(args.scheduler, ReduceLROnPlateau):
            # todo, get validaton loss, and give it to scheduler, make sure it make sense with the log_scheduler_freq
            val_batch: Any = next(iter(args.dataloaders['val']))
            val_loss, val_loss_ci, val_acc, val_acc_ci = args.mdl.eval_forward(val_batch)
            args.scheduler.step(val_loss)
            raise NotImplementedError
        elif isinstance(args.scheduler, transformers.optimization.AdafactorSchedule):
            args.scheduler.step()
        else:
            raise ValueError(f'Error, invalid Scheduler: the scheduler {args.scheduler=} is not supported.')

# def should_we_call_scheduler(args: Namespace) -> bool:
#     """"""
#     # do not schedule on the first epoch, since the args.epoch_num +
#     on_first_epocn: bool = args.it != 0
#     call_scheduler: bool = (args.it % args.log_scheduler_freq == 0 and args.it != 0) or args.debug
#     return call_scheduler


