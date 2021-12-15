from argparse import Namespace
from typing import Any

import progressbar
from torch import Module, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import uutils
from uutils.torch_uu import AverageMeter
from uutils.torch_uu.agents import Agent
from uutils.torch_uu.checkpointing_uu.supervised_learning import save_for_supervised_learning
from uutils.torch_uu.distributed import print_dist


def train_single_batch_agent(args: Namespace,
                             mdl: Module,
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

    def log_train_stats(it: int, train_loss: float, acc: float):
        val_loss, val_acc = mdl(val_batch)

        # todo - extend to proper log if needed...that supports wandb and take args in etc.
        print_dist(f"\n{it=}: {train_loss=} {acc=}")
        print_dist(f"{it=}: {val_loss=} {val_acc=}")

    # first batch
    args.it = 0  # training a single batch shouldn't need to use ckpts so this is ok
    avg_loss = AverageMeter('train loss')
    avg_acc = AverageMeter('train accuracy')
    bar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    while True:
        train_loss, train_acc = mdl(train_batch, training=True)

        opt.zero_grad()
        train_loss.backward()  # each process synchronizes it's gradients in the backward pass
        opt.step()  # the right update is done since all procs have the right synced grads

        # if agent.agent.is_lead_worker() and agent.args.it % 10 == 0:
        if args.it % 10 == 0:
            # todo - think if to put all this in log function
            bar.update(args.it)
            log_train_stats(args.it, train_loss, train_acc)
            # agent.save(agent.args.it)  # very expensive! but if fitting one batch its ok to save it every time

        args.it += 1
        if train_acc >= acc_tolerance and train_loss <= train_loss_tolerance:
            bar.update(args.it)
            log_train_stats(args.it, train_loss, train_acc)
            # agent.save(agent.args.it)  # very expensive! but if fitting one batch its ok to save it every time
            break  # halt once performance is good enough

    return avg_loss.item(), avg_acc.item()


def main_train_fixed_number_of_epochs(args: Namespace,
                                      mdl: Module,
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


def main_train_loop_until_convergence(agent, args: Namesspace, acc_tolerance: float = 1.0,
                                      train_loss_tolerance: float = 0.001):
    pass


# - quick evals

def _eval(args: Namespace,
          mdl: nn.Module,
          training: bool = False,
          val_iterations: int = 0,
          save_val_ckpt: bool = True,
          split: str = 'val'
          ) -> tuple:
    """
    ref for BN/eval:
        - For SL: do .train() for training and .eval() for eval in SL.
        - For Meta-learning do train in both, see: https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
    """
    assert val_iterations == 0, f'Val iterations has to be zero but got {val_iterations}, ' \
                                f'if you want more precision increase (meta) batch size.'
    # - Training=True for eval only for meta-learning, here we do want .eval(), but then undo it
    args.meta_learner.train() if training else args.meta_learner.eval()
    original_reduction: str = mdl.loss.reduction
    mdl.loss.reduction = 'none'
    for batch_idx, eval_batch in enumerate(args.dataloaders[split]):
        # Forward pass
        # eval_loss, eval_acc = args.meta_learner(spt_x, spt_y, qry_x, qry_y)
        # eval_loss, eval_acc, eval_loss_std, eval_acc_std = args.meta_learner(spt_x, spt_y, qry_x, qry_y,
        #                                                                      training=training)
        eval_loss, eval_acc = mdl(eval_batch)

        # store eval info
        if batch_idx >= val_iterations:
            break

    if float(eval_loss) < float(args.best_val_loss) and save_val_ckpt:
        args.best_val_loss = float(eval_loss)
        save_for_supervised_learning(args, ckpt_filename='ckpt_best_val.pt')
    # - return things to normal before returning
    args.meta_learner.train() if training else args.meta_learner.eval()
    mdl.loss.reduction = original_reduction
    eval_loss_std, eval_acc_std = eval_loss.std(), eval_acc
    return eval_loss, eval_acc, eval_loss_std, eval_acc_std
    # return eval_loss, eval_acc
