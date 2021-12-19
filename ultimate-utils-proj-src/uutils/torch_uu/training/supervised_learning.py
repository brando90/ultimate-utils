"""

To save time, debug, coding; have everything be in terms of iterations, but control the frequency of logging with
the log frequency parameter. When calling the epochs function, automatically set that in that function,
else set some defaults in the argparse.


# - halt when converged
ref:
    - https://forums.pytorchlightning.ai/t/what-is-the-standard-way-to-halt-a-script-when-it-has-converged/1415
    - https://stackoverflow.com/questions/70405985/what-is-the-standard-way-to-train-a-pytorch-script-until-convergence
"""
from argparse import Namespace
from typing import Any

import progressbar
from progressbar import ProgressBar
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import uutils
from uutils.torch_uu import AverageMeter, gradient_clip
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.checkpointing_uu.supervised_learning import save_for_supervised_learning
from uutils.torch_uu.distributed import print_dist, is_lead_worker
from uutils.torch_uu.training.common import get_trainer_progress_bar


def train_agent_fit_single_batch(args: Namespace,
                                 mdl: Agent,
                                 dataloaders: dict,
                                 opt: Optimizer,
                                 scheduler: _LRScheduler,
                                 acc_tolerance: float = 1.0,
                                 train_loss_tolerance: float = 0.01):
    """
    Train for a single batch
    """
    train_batch: Any = next(iter(dataloaders['train']))
    val_batch: Any = next(iter(dataloaders['val']))

    def log_train_stats(it: int, train_loss: float, train_acc: float, bar: ProgressBar, save_val_ckpt: bool = True,
                        force_log: bool = False):
        # - get eval stats
        val_loss, val_loss_ci, val_acc, val_acc_ci = mdl.eval_forward(val_batch)
        if float(val_loss - val_loss_ci) < float(args.best_val_loss) and save_val_ckpt:
            args.best_val_loss = float(val_loss)
            save_for_supervised_learning(args, ckpt_filename='ckpt_best_val.pt')

        # - log ckpt
        if it % 10 == 0 or force_log:
            save_for_supervised_learning(args, ckpt_filename='ckpt.pt')

        # - save args
        uutils.save_args(args, args_filename='args.json')

        # - update progress bar at the end
        bar.update(it)

        # - print
        print_dist(f"\n{it=}: {train_loss=} {train_acc=}")
        print_dist(f"{it=}: {val_loss=} {val_acc=}")

        # - for now no wandb for logging for one batch...perhaps change later

    # first batch
    args.it = 0  # training a single batch shouldn't need to use ckpts so this is ok
    args.best_val_loss = float('inf')  # training a single batch shouldn't need to use ckpts so this is ok
    avg_loss = AverageMeter('train loss')
    avg_acc = AverageMeter('train accuracy')
    bar: ProgressBar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    while True:
        train_loss, train_acc = mdl(train_batch, training=True)

        opt.zero_grad()
        train_loss.backward()  # each process synchronizes its gradients in the backward pass
        opt.step()  # the right update is done since all procs have the right synced grads
        gradient_clip(args, opt)
        if (args.it % 15 == 0 and args.it != 0) or args.debug:
            scheduler.step() if (scheduler is not None) else None

        if args.it % 10 == 0 and is_lead_worker(args):
            log_train_stats(args.it, train_loss, train_acc, bar)

        # - break
        halt: bool = train_acc >= acc_tolerance and train_loss <= train_loss_tolerance
        if halt:
            log_train_stats(args.it, train_loss, train_acc, force_log=True)
            return avg_loss.item(), avg_acc.item()
        args.it += 1


def train_agent_iterations():
    pass


def train_agent_epochs(args: Namespace,
                                       mdl: Agent,
                                       dataloaders: dict,
                                       opt: Optimizer,
                                       scheduler: _LRScheduler,
                                       acc_tolerance: float = 1.0,
                                       train_loss_tolerance: float = 0.01) -> tuple[Tensor, Tensor]:
    """
    Trains model one epoch at a time - i.e. it's epochs based rather than iteration based.
    """
    print_dist('Starting training...')
    B: int = args.batch_size

    # - create progress bar
    bar_epoch: ProgressBar = get_trainer_progress_bar(args)

    # - train in epochs
    train_loss, train_acc = float('inf'), 0.0
    halt: bool = False
    while not halt:
        # -- train for one epoch
        avg_loss = AverageMeter('train loss')
        avg_acc = AverageMeter('train accuracy')
        for i, batch in enumerate(mdl.dataloaders['train']):
            train_loss, train_acc = mdl(batch, training=True)
            opt.zero_grad()
            train_loss.backward()  # each process synchronizes its gradients in the backward pass
            opt.step()  # the right update is done since all procs have the right synced grads
            gradient_clip(args, opt)

            # - meter updates
            avg_loss.update(train_loss.item(), B), avg_acc.update(train_acc, B)
        # - scheduler
        if (args.it % 500 == 0 and args.it != 0 and hasattr(args, 'scheduler')) or args.debug:
            args.scheduler.step() if (args.scheduler is not None) else None

        # - log full epoch stats
        if log_condition():
            self.log_train_stats(self.args.epoch_n, avg_loss.item(), avg_acc.item(), save_val_ckpt=True)

        # - go to next it & before that check if we should halt
        args.epoch_num += 1
        halt = check_halt(args)

    return avg_loss.item(), avg_acc.item()  #


# - quick evals

def eval(args: Namespace,
         mdl: nn.Module,
         training: bool = False,
         val_iterations: int = 0,
         split: str = 'val'
         ) -> tuple:
    """

    Note:
        -  Training=True for eval only for meta-learning, here we do want .eval(), but then undo it

    ref for BN/eval:
        - For SL: do .train() for training and .eval() for eval in SL.
        - For Meta-learning do train in both, see: https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
    """
    assert val_iterations == 0, f'Val iterations has to be zero but got {val_iterations}, ' \
                                f'if you want more precision increase (meta) batch size.'
    args.meta_learner.train() if training else args.meta_learner.eval()
    for batch_idx, eval_batch in enumerate(args.dataloaders[split]):
        eval_loss_mean, eval_loss_ci, eval_acc_mean, eval_acc_ci = mdl.eval_forward(eval_batch)
        if batch_idx >= val_iterations:
            break
    return eval_loss_mean, eval_loss_ci, eval_acc_mean, eval_acc_ci
