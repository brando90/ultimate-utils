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
from torch import nn, Tensor, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
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
        halt: bool = args.convg_meter.check_converged()
    elif args.training_mode == 'iterations':
        #  idx + 1 == n, this means you've done n loops where idx = it or epoch_num
        halt: bool = args.it + 1 >= args.num_its
    elif args.training_mode == 'epochs':
        #  idx + 1 == n, this means you've done n loops where idx = it or epoch_num
        halt: bool = args.epoch_num + 1 >= args.num_epochs
    elif args.training_mode == 'iterations_train_convergence':
        halt: bool = args.convg_meter.check_converged()
    elif args.training_mode == 'epochs_train_convergence':
        halt: bool = args.convg_meter.check_converged()
    else:
        raise ValueError(f'Invalid training_mode value, got: {args.training_mode}')
    return halt


def gradient_clip(args, meta_opt):
    """
    Do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖
    depending on args it does it per parameter or all parameters together.

    Note:
        - don't do this if you're using Adafactor.
    """
    # do gradient clipping: If ‖g‖ ≥ c Then g := c * g/‖g‖
    if hasattr(args, 'grad_clip_mode'):
        if args.grad_clip_mode is None:
            pass
        elif args.grad_clip_mode == 'no_grad_clip' or args.grad_clip_mode is None:  # for backwards compatibility
            pass
        elif args.grad_clip_mode == 'clip_all_seperately':
            for group_idx, group in enumerate(meta_opt.param_groups):
                for p_idx, p in enumerate(group['params']):
                    nn.utils.clip_grad_norm_(p, args.grad_clip_rate)
        elif args.grad_clip_mode == 'clip_all_together':
            # [y for x in list_of_lists for y in x]
            all_params = [p for group in meta_opt.param_groups for p in group['params']]
            nn.utils.clip_grad_norm_(all_params, args.grad_clip_rate)
        else:
            raise ValueError(f'Invalid, args.grad_clip_mode = {args.grad_clip_mode}')


def scheduler_step(args: Namespace, scheduler: _LRScheduler):
    """

    Note:
        - use the if i % log_freq outside. So it assumes you already decided how often to call this. With this setup it
        means that when you do s.step(val_loss) the step will be with respect to a step/collection of its or epochs.
    """
    assert args.scheduler is scheduler
    if not hasattr(args, 'scheduler'):
        return
    else:
        # args.scheduler.step() if (args.scheduler is not None) else None
        if args.scheduler is None:
            return
        # -- ReduceLROnPlateu
        elif isinstance(args.scheduler, ReduceLROnPlateau):
            val_batch: Any = next(iter(args.dataloaders['val']))
            val_loss, val_loss_ci, val_acc, val_acc_ci = args.mdl.eval_forward(val_batch)
            args.scheduler.step(val_loss)
            raise NotImplementedError
        # -- AdafactorSchedule transformers/hugging face
        elif isinstance(args.scheduler, transformers.optimization.AdafactorSchedule):
            args.scheduler.step()
        # -- CosineAnnealingLR
        # based: https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/train_supervised.py#L243
        # rfs seems to call cosine scheduler every epoch, since cosine takes in max step, I assume it must be fine to
        # call it every step, it probably decays with a cosine according to the step num they are on (epoch or it).
        elif isinstance(args.scheduler, optim.lr_scheduler.CosineAnnealingLR):
            args.scheduler.step()
        # -- Error catcher
        else:
            raise ValueError(f'Error, invalid Scheduler: the scheduler {args.scheduler=} is not supported.')


# - meter

class ConvergenceMeter:
    """
    Class that keeps track if you have converged or not. If you track val_loss this will be equivalent to
    Early Stopping.
    """

    def __init__(self, name: str, convergence_patience: int = 5, current_lowest: float = float('inf')):
        """

        :param name:
        :param patience: number of checks with no improvement after which training will be stopped. Note that the
        intention is that the training code decides how often to call this, ideally in the if i % log_freq conditional.
        """
        self.name = name
        self.current_lowest = current_lowest
        self.counts_above_current_lowest = 0
        self.convergence_patience = convergence_patience
        self.reset(current_lowest)

    def reset(self, new_lowest: float):
        self.current_lowest = new_lowest
        self.counts_above_current_lowest = 0

    def update(self, val):
        """
        Attempt to update the current lowest. If it is lower set this new lowest to be the lowest.
        """
        # - if you have a loss that decreased the lowest you've seen re-start counting.
        if val < self.current_lowest:
            self.reset(new_lowest=val)
        # - if you were not bellow the lowest, then you've been above the lowest for +1 counts
        else:
            self.counts_above_current_lowest += 1

    def item(self):
        if type(self.current_lowest) is Tensor:
            return self.current_lowest.item()
        else:
            return float(self.current_lowest)

    def check_converged(self) -> bool:
        """
        Check if you have converged. If the loss/value you are tracking has been above the current lowest you have seen
        enough times, then you have converged.
        """
        # - halt if you've been above the current lowest enough times.
        return self.convergence_patience <= self.counts_above_current_lowest

    def __str__(self):
        """
        ref:
            - .__dict__ or vars( ) ? https://stackoverflow.com/questions/21297203/use-dict-or-vars
        :return:
        """
        return f'ConvergenceMeter({vars(self)}'
