"""

To save time, debug, coding; have everything be in terms of iterations, but control the frequency of logging with
the log frequency parameter. When calling the epochs function, automatically set that in that function,
else set some defaults in the argparse.
"""
from argparse import Namespace
from typing import Any

import progressbar
from progressbar import ProgressBar
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import uutils
from uutils.torch_uu import AverageMeter
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.checkpointing_uu.supervised_learning import save_for_supervised_learning
from uutils.torch_uu.distributed import print_dist, is_lead_worker


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


def train_agent_fixed_number_of_iterations():
    pass


def train_agent_fixed_number_of_epochs(args: Namespace,
                                       mdl: nn.Module,
                                       dataloaders: dict,
                                       opt: Optimizer,
                                       scheduler: _LRScheduler):
    """
    Trains model one epoch at a time - i.e. it's epochs based rather than iteration based.
    """
    print_dist('Starting training...')

    bar_epoch = uutils.get_good_progressbar(max_value=args.num_epochs)
    train_loss, train_acc = float('inf'), 0.0
    epochs: iter = range(start_epoch, start_epoch + args.num_epochs)
    assert len(epochs) == args.num_epochs
    # - train in epochs
    for epoch_num in range(start_epoch, start_epoch + args.num_epochs):
        # -- train for one epoch
        avg_loss = AverageMeter('train loss')
        avg_acc = AverageMeter('train accuracy')
        for i, batch in enumerate(self.dataloaders['train']):
            train_loss, train_acc = self.forward_one_batch(batch, training=True)
            avg_loss.update(train_loss.item(), self.args.batch_size)
            avg_acc.update(train_acc, self.args.batch_size)

            self.optimizer.zero_grad()
            train_loss.backward()  # each process synchronizes its gradients in the backward pass
            self.optimizer.step()  # the right update is done since all procs have the right synced grads

            # - scheduler, annealing/decaying the learning rate
            if (args.it % 500 == 0 and args.it != 0 and hasattr(args, 'scheduler')) or args.debug:
                args.scheduler.step() if (args.scheduler is not None) else None

        # -- log full epoch stats
        self.log_train_stats(self.args.epoch_n, avg_loss.item(), avg_acc.item(), save_val_ckpt=True)
        args.epoch_num += 1

    return avg_loss.item(), avg_acc.item()  #


def train_agent_fit_until_convergence(agent, args: Namespace, acc_tolerance: float = 1.0,
                                      train_loss_tolerance: float = 0.001):
    pass


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
