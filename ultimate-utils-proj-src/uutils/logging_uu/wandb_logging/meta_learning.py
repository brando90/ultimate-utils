from argparse import Namespace

import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Optimizer

import uutils
from uutils.logging_uu.wandb_logging.common import log_2_wanbd
from uutils.torch_uu import r2_score_from_torch
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.checkpointing_uu.meta_learning import save_for_meta_learning
from uutils.torch_uu.distributed import is_lead_worker

from pdb import set_trace as st

def _log_train_val_stats(args: Namespace,
                         it: int,

                         train_loss: float,
                         train_acc: float,

                         valid,

                         bar,

                         log_freq: int = 10,
                         ckpt_freq: int = 50,
                         mdl_watch_log_freq: int = 50,
                         force_log: bool = False,  # e.g. at the final it/epoch

                         save_val_ckpt: bool = False,
                         log_to_tb: bool = False,
                         log_to_wandb: bool = False
                         ):
    """
    Log train and val stats where it is iteration or epoch step.

    Note: Unlike save ckpt, this one does need it to be passed explicitly (so it can save it in the stats collector).
    """
    import wandb
    from uutils.torch_uu.tensorboard import log_2_tb_supervisedlearning
    # - is it epoch or iteration
    it_or_epoch: str = 'epoch_num' if args.training_mode == 'epochs' else 'it'

    # if its
    total_its: int = args.num_epochs if args.training_mode == 'epochs' else args.num_its

    if (it % log_freq == 0 or it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        # - get eval stats
        val_loss, val_acc, val_loss_std, val_acc_std = valid(args, save_val_ckpt=save_val_ckpt)
        # - log ckpt
        if it % ckpt_freq == 0:
            save_for_meta_learning(args)

        # - save args
        uutils.save_args(args, args_filename='args.json')

        # - update progress bar at the end
        bar.update(it)

        # - print
        st()
        args.logger.log('\n')
        args.logger.log(f"{it_or_epoch}={it}: {train_loss=}, {train_acc=}")
        args.logger.log(f"{it_or_epoch}={it}: {val_loss=}, {val_acc=}")

        print(f'{args.it=}')
        print(f'{args.num_its=}')

        # - record into stats collector
        args.logger.record_train_stats_stats_collector(it, train_loss, train_acc)
        args.logger.record_val_stats_stats_collector(it, val_loss, val_acc)
        args.logger.save_experiment_stats_to_json_file()
        args.logger.save_current_plots_and_stats()

        # - log to wandb
        if log_to_wandb:
            log_2_wanbd(it, train_loss, train_acc, val_loss, val_acc, it_or_epoch)

        # - log to tensorboard
        if log_to_tb:
            log_2_tb_supervisedlearning(args.tb, args, it, train_loss, train_acc, 'train')
            log_2_tb_supervisedlearning(args.tb, args, it, val_loss, val_acc, 'val')


def log_zeroth_step(args: Namespace, meta_learner: Agent):
    from learn2learn.data import TaskDataset
    from uutils.logging_uu.wandb_logging.supervised_learning import log_train_val_stats
    task_dataset: TaskDataset = args.tasksets.train
    train_loss, train_loss_std, train_acc, train_acc_std = meta_learner(task_dataset)
    step_name: str = 'epoch_num' if 'epochs' in args.training_mode else 'it'
    log_train_val_stats(args, args.it, step_name, train_loss, train_acc, training=True)


# - tests

def get_args() -> Namespace:
    args = uutils.parse_args_synth_agent()
    args = uutils.setup_args_for_experiment(args)
    return args


def valid_for_test(args: Namespace, mdl: nn.Module, save_val_ckpt: bool = False):
    import torch

    for t in range(1):
        x = torch.randn(args.batch_size, 5)
        y = (x ** 2 + x + 1).sum(dim=1)

        y_pred = mdl(x).squeeze(dim=1)
        val_loss, val_acc = mse_loss(y_pred, y), r2_score_from_torch(y_true=y, y_pred=y_pred)

    if val_loss.item() < args.best_val_loss and save_val_ckpt:
        args.best_val_loss = val_loss.item()
        # save_ckpt(args, args.mdl, args.optimizer, ckpt_name='ckpt_best_val.pt')
    return val_loss, val_acc


def train_for_test(args: Namespace, mdl: nn.Module, optimizer: Optimizer, scheduler=None):
    for it in range(50):
        x = torch.randn(args.batch_size, 5)
        y = (x ** 2 + x + 1).sum(dim=1)

        y_pred = mdl(x).squeeze(dim=1)
        train_loss, train_acc = mse_loss(y_pred, y), r2_score_from_torch(y_true=y, y_pred=y_pred)

        optimizer.zero_grad()
        train_loss.backward()  # each process synchronizes it's gradients in the backward pass
        optimizer.step()  # the right update is done since all procs have the right synced grads
        scheduler.step()

        if it % 2 == 0 and is_lead_worker(args.rank):
            _log_train_val_stats(args, it, train_loss, train_acc, valid_for_test, save_val_ckpt=True, log_to_tb=True)
            if it % 10 == 0:
                # save_ckpt(args, args.mdl, args.optimizer)
                pass

    return train_loss, train_acc


def debug_test():
    args: Namespace = get_args()

    # - get mdl, opt, scheduler, etc
    from uutils.torch_uu.models import get_simple_model
    args.mdl = get_simple_model(in_features=5, hidden_features=20, out_features=1, num_layer=2)
    args.optimizer = torch.optim.Adam(args.mdl.parameters(), lr=1e-1)
    args.scheduler = torch.optim.lr_scheduler.ExponentialLR(args.optimizer, gamma=0.999, verbose=False)

    # - train
    train_loss, train_acc = train_for_test(args, args.mdl, args.optimizer, args.scheduler)
    print(f'{train_loss=}, {train_loss=}')

    # - eval
    val_loss, val_acc = valid_for_test(args, args.mdl)

    print(f'{val_loss=}, {val_acc=}')


if __name__ == '__main__':
    import time

    start = time.time()
    debug_test()
    duration_secs = time.time() - start
    print(
        f"Success, time passed: hours:{duration_secs / (60 ** 2)}, minutes={duration_secs / 60}, seconds={duration_secs}")
    print('Done!\a')
