"""
https://docs.wandb.ai/guides/track/advanced/distributed-training

import wandb

# 1. Start a new run
wandb.init(project='playground', entity='brando')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# 3. Log gradients and model parameters
wandb.watch(model)
for batch_idx, (data, target) in enumerate(train_loader):
    ...
    if batch_idx % args.log_interval == 0:
        # 4. Log metrics to visualize performance
        wandb.log({"loss": loss})


Notes:
    - call wandb.init and wandb.log only from the leader process
"""

from argparse import Namespace
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Optimizer

import uutils
from uutils.torch_uu import r2_score_from_torch
from uutils.torch_uu.distributed import is_lead_worker
from uutils.torch_uu.models import get_simple_model
from uutils.torch_uu.tensorboard import log_2_tb_supervisedlearning


import wandb

def log_2_wandb_nice(it, loss, inputs, outputs, captions):
    wandb.log({"loss": loss, "epoch": it,
               "inputs": wandb.Image(inputs),
               "logits": wandb.Histogram(outputs),
               "captions": wandb.HTML(captions)})

def log_2_wandb(**metrics):
    """ Log to wandb """
    new_metrics: dict = {}
    for key, value in metrics.items():
        key = str(key).strip('_')
        new_metrics[key] = value
    wandb.log(new_metrics)


def log_train_val_stats(args: Namespace,
                        it: int,

                        train_loss: float,
                        train_acc: float,

                        valid,

                        log_freq: int = 10,
                        ckpt_freq: int = 50,
                        force_log: bool = False,  # e.g. at the final it/epoch

                        save_val_ckpt: bool = False,
                        log_to_tb: bool = False,
                        log_to_wandb: bool = False
                        ):
    """

    log train and val stats.

    Note: Unlike save ckpt, this one does need it to be passed explicitly (so it can save it in the stats collector).
    """
    from uutils.torch_uu.tensorboard import log_2_tb
    from matplotlib import pyplot as plt

    # - is it epoch or iteration
    it_or_epoch: str = 'epoch_num' if args.training_mode == 'epochs' else 'it'
    # if its
    total_its: int = args.num_empochs if args.training_mode == 'epochs' else args.num_its

    print(f'-- {it == total_its - 1}')
    print(f'-- {it}')
    print(f'-- {total_its}')
    if (it % log_freq == 0 or is_lead_worker(args.rank) or it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        print('inside log')
        # - get eval stats
        val_loss, val_acc = valid(args, args.mdl, save_val_ckpt=save_val_ckpt)

        # - print
        args.logger.log('\n')
        args.logger.log(f"{it_or_epoch}={it}: {train_loss=}, {train_acc=}")
        args.logger.log(f"{it_or_epoch}={it}: {val_loss=}, {val_acc=}")

        # - record into stats collector
        args.logger.record_train_stats_stats_collector(it, train_loss, train_acc)
        args.logger.record_val_stats_stats_collector(it, val_loss, val_acc)
        args.logger.save_experiment_stats_to_json_file()
        fig = args.logger.save_current_plots_and_stats()

        # - log to wandb
        if log_to_wandb:
            if it == 0:
                wandb.watch(args.mdl)
                print('watching model')
            # log_2_wandb(train_loss=train_loss, train_acc=train_acc)
            print('inside wandb log')
            wandb.log(data={'train loss': train_loss, 'train acc': train_acc, 'val loss': val_loss, 'val acc': val_acc}, step=it)
            wandb.log(data={'it': it}, step=it)
            if it == total_its - 1:
                wandb.log(data={'fig': fig}, step=it)
        plt.close('all')

        # - log to tensorboard
        if log_to_tb:
            log_2_tb_supervisedlearning(args.tb, args, it, train_loss, train_acc, 'train')
            log_2_tb_supervisedlearning(args.tb, args, it, train_loss, train_acc, 'val')
            # log_2_tb(args, it, val_loss, val_acc, 'train')
            # log_2_tb(args, it, val_loss, val_acc, 'val')

    # - log ckpt
    if (it % ckpt_freq == 0 or it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        save_ckpt(args, args.mdl, args.optimizer)


def save_ckpt(args: Namespace, mdl: nn.Module, optimizer: torch.optim.Optimizer,
              dirname: Union[None, Path] = None, ckpt_name: str = 'ckpt.pt'):
    """
    Saves checkpoint for any worker.
    Intended use is to save by worker that got a val loss that improved.


    """
    import dill

    dirname = args.log_root if (dirname is None) else dirname
    # - pickle ckpt
    assert uutils.xor(args.training_mode == 'epochs', args.training_mode == 'iterations')
    pickable_args = uutils.make_args_pickable(args)
    torch.save({'state_dict': mdl.state_dict(),
                'epoch_num': args.epoch_num,
                'it': args.it,
                'optimizer': optimizer.state_dict(),
                'args': pickable_args,
                'mdl': mdl},
               pickle_module=dill,
               f=dirname / ckpt_name)  # f'mdl_{epoch_num:03}.pt'


def get_args() -> Namespace:
    args = uutils.parse_args_synth_agent()
    # we can place model here...
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
        save_ckpt(args, args.mdl, args.optimizer, ckpt_name='ckpt_best_val.pt')
    return val_loss, val_acc


def train_for_test(args: Namespace, mdl: nn.Module, optimizer: Optimizer, scheduler=None):
    wandb.watch(args.mdl)
    for it in range(args.num_its):
        x = torch.randn(args.batch_size, 5)
        y = (x ** 2 + x + 1).sum(dim=1)

        y_pred = mdl(x).squeeze(dim=1)
        train_loss, train_acc = mse_loss(y_pred, y), r2_score_from_torch(y_true=y, y_pred=y_pred)

        optimizer.zero_grad()
        train_loss.backward()  # each process synchronizes it's gradients in the backward pass
        optimizer.step()  # the right update is done since all procs have the right synced grads
        scheduler.step()

        log_train_val_stats(args, it, train_loss, train_acc, valid_for_test,
                            log_freq=2, ckpt_freq=10,
                            save_val_ckpt=True, log_to_tb=True, log_to_wandb=True)

    return train_loss, train_acc


def debug_test():
    args: Namespace = get_args()
    args.num_its = 12

    # - get mdl, opt, scheduler, etc
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
    import os

    # print(os.environ['WANDB_API_KEY'])
    import time
    start = time.time()
    debug_test()
    duration_secs = time.time() - start
    print(f"Success, time passed: hours:{duration_secs / (60 ** 2)}, minutes={duration_secs / 60}, seconds={duration_secs}")
    print('Done!\a')
