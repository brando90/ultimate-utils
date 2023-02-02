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


def log_zeroth_step(args: Namespace, meta_learner: Agent):
    print('log_zeroth_step')
    from uutils.logging_uu.wandb_logging.supervised_learning import log_train_val_stats
    if hasattr(args, 'tasksets'):
        from learn2learn.data import TaskDataset
        task_dataset: TaskDataset = args.tasksets.train
        print(task_dataset)
        train_loss, train_loss_std, train_acc, train_acc_std = meta_learner(task_dataset)
    else:
        batch = next(iter(args.dataloaders['train']))  # this might advance the dataloader one step
        print(f'{args.dataloaders=}')
        train_loss, train_loss_std, train_acc, train_acc_std = meta_learner(batch)
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
    from uutils.logging_uu.wandb_logging.supervised_learning import log_train_val_stats
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
            log_train_val_stats(args, it, train_loss, train_acc, valid_for_test, save_val_ckpt=True, log_to_tb=True)
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
