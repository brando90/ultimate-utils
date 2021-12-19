"""

# - scheduling rate tips
Do "But the most commonly used method is when the validation loss does not improve for a few epochs." according to https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

"""
from argparse import Namespace

import progressbar
from progressbar import ProgressBar
from torch import nn


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

def check_halt(args: Namespace):
    if args.training_mode == 'fit_single_batch':
        train_agent_fit_single_batch(args, agent, args.dataloaders, args.opt, args.scheduler)
    elif args.training_mode == 'iterations':
        # note train code will see training mode to determine halting criterion
        train_agent_iterations(args, agent, args.dataloaders, args.opt, args.scheduler)
    elif args.training_mode == 'epochs':
        # note train code will see training mode to determine halting criterion
        train_agent_epochs(args, agent, args.dataloaders, args.opt, args.scheduler)
    else:
        raise ValueError(f'Invalid training_mode value, got: {args.training_mode}')


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

def scheduler_step(scheduler):
