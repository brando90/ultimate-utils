"""
"""
from argparse import Namespace
from typing import Any

import torch
from learn2learn.data import TaskDataset
from progressbar import ProgressBar
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import uutils
from uutils.logging_uu.wandb_logging.meta_learning import log_zeroth_step
from uutils.logging_uu.wandb_logging.supervised_learning import log_train_val_stats
from uutils.torch_uu.checkpointing_uu.meta_learning import save_for_meta_learning
from uutils.torch_uu.training.common import gradient_clip, scheduler_step, check_halt, get_trainer_progress_bar, \
    ConvergenceMeter

from pdb import set_trace as st


def print_inside_halt(args: Namespace, halt: bool, i: int = 0):
    if args.it % args.log_freq == 0 or halt or args.debug:  # todo: remove? temporary for debugging
        print(f'-- inside the halt, ith: {i=}')


def meta_train_agent_fit_single_batch(args: Namespace,
                                      meta_learner,
                                      dataloaders: dict,
                                      opt: Optimizer,
                                      scheduler: _LRScheduler,
                                      acc_tolerance: float = 1.0,
                                      train_loss_tolerance: float = 0.01,
                                      ):
    """
    Train for a single batch
    """
    # get one fixed meta-batch
    # if hasattr(args, 'tasksets'):
    #    # hack for l2l
    #    from learn2learn.data import TaskDataset
    #    split: str = 'validation' if split == 'val' else split
    #    task_dataset: TaskDataset = getattr(args.tasksets, split)
    #    assert self.args.batch_size >= 1
    #    train_batch = : list = [task_dataset.sample() for task_num in range(self.args.batch_size)]
    #    # train_batch = task_dataset
    # else:  # torchmeta which has same api as pytorch dataloader
    train_batch: Any = next(iter(dataloaders['train']))

    # first batch
    args.it = 0  # training a single batch shouldn't need to use ckpts so this is ok
    args.best_val_loss = float('inf')  # training a single batch shouldn't need to use ckpts so this is ok

    # - create progress bar
    args.bar: ProgressBar = get_trainer_progress_bar(args)

    # - train in epochs
    args.convg_meter: ConvergenceMeter = ConvergenceMeter(name='train loss',
                                                          convergence_patience=args.train_convergence_patience)
    # log_zeroth_step(args, model)
    halt: bool = False
    while not halt:
        opt.zero_grad()
        train_loss, train_loss_ci, train_acc, train_acc_ci = meta_learner(train_batch, call_backward=True)
        # train_loss.backward()  # each process synchronizes its gradients in the backward pass
        assert opt.param_groups[0]['params'][0].grad is not None
        gradient_clip(args, opt)
        opt.step()  # the right update is done since all procs have the right synced grads
        if (args.it % args.log_scheduler_freq == 0) or args.debug:
            scheduler_step(args, scheduler)

        # - break
        halt: bool = train_acc >= acc_tolerance and train_loss <= train_loss_tolerance
        check_halt(args)  # for the sake of making sure check_halt runs

        args.it += 1

        # - log full stats
        # when logging after +=1, log idx will be wrt real idx i.e. 0 doesn't mean first it means true 0
        if args.epoch_num % args.log_freq == 0 or halt or args.debug:
            step_name: str = 'epoch_num' if 'epochs' in args.training_mode else 'it'
            log_train_val_stats(args, args.it, step_name, train_loss, train_acc, training=True)

        if halt:
            break

    return train_loss, train_acc


def meta_train_fixed_iterations(args: Namespace,
                                meta_learner,
                                dataloaders,
                                outer_opt,
                                scheduler,
                                training: bool = True
                                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Train using the meta-training (e.g. episodic training) over batches of tasks using a fixed number of iterations
    assuming the number of tasks is small i.e. one epoch is doable and not infinite/super exponential
    (e.g. in regression when a task can be considered as a function).

    Note: if num tasks is small then we have two loops, one while we have not finished all fixed its and the other
    over the dataloader for the tasks.
    """
    print('Starting training!')

    # args.bar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    args.bar = uutils.get_good_progressbar(max_value=args.num_its)
    meta_learner.train() if training else meta_learner.eval()
    halt: bool = False

    # ----added - 0th iter---#
    log_zeroth_step(args, meta_learner)
    # --------#
    while not halt:
        for batch_idx, batch in enumerate(dataloaders['train']):
            outer_opt.zero_grad()
            assert outer_opt is args.opt
            train_loss, train_loss_ci, train_acc, train_acc_ci = meta_learner(batch, call_backward=True)
            # train_loss.backward()  # NOTE: backward was already called in meta-learner due to MEM optimization.
            assert outer_opt.param_groups[0]['params'][0].grad is not None
            gradient_clip(args, outer_opt)  # do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖
            outer_opt.step()

            # - scheduler
            if (args.it % args.log_scheduler_freq == 0) or args.debug:
                scheduler_step(args, scheduler)

            # - convergence (or idx + 1 == n, this means you've done n loops where idx = it or epoch_num).
            halt: bool = check_halt(args)

            # - go to next it & before that check if we should halt
            args.it += 1

            # - log full stats
            # when logging after +=1, log idx will be wrt real idx i.e. 0 doesn't mean first it means true 0
            if args.it % args.log_freq == 0 or halt or args.debug:
                step_name: str = 'epoch_num' if 'epochs' in args.training_mode else 'it'
                log_train_val_stats(args, args.it, step_name, train_loss, train_acc, training=True)

            # - break out of the inner loop to start halting, the outer loop will terminate too since halt is True.
            if halt:
                break


def meta_train_iterations_ala_l2l(args: Namespace,
                                  meta_learner,
                                  outer_opt,
                                  scheduler,
                                  training: bool = True
                                  ):
    """"""
    # torch.distributed.barrier()
    print('Starting training!')
    meta_batch_size: int = args.batch_size // args.world_size
    # meta_batch_size: int = max(args.batch_size // args.world_size, 1)
    # assert args.batch_size >= args.world_size, f'If batch size is smaller training might be slightly wrong when in distributed.'

    # print(args.batch_size, meta_batch_size, "args and Meta BatchSize")
    # args.bar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    args.bar = uutils.get_good_progressbar(max_value=args.num_its)
    meta_learner.train() if training else meta_learner.eval()
    halt: bool = False

    # ----added - 0th iter---#
    log_zeroth_step(args, meta_learner)
    # --------#
    while not halt:
        # print(f'{args.rank=}')
        # print_inside_halt(args, halt, 0)  # todo: remove? temporary for debugging
        outer_opt.zero_grad()
        # print_inside_halt(args, halt, 1)  # todo: remove? temporary for debugging

        # - forward pass. Since the data fetching is different for l2l we do it this way
        task_dataset: TaskDataset = args.tasksets.train
        # print_inside_halt(args, halt, 2)  # todo: remove? temporary for debugging
        train_loss, train_loss_ci, train_acc, train_acc_ci = meta_learner(task_dataset, call_backward=True)
        # print_inside_halt(args, halt, 3)  # todo: remove? temporary for debugging
        # train_loss.backward()  # NOTE: backward was already called in meta-learner due to MEM optimization.
        assert outer_opt.param_groups[0]['params'][0].grad is not None
        # print_inside_halt(args, halt, 4)  # todo: remove? temporary for debugging

        # - Grad clip  (optional)
        gradient_clip(args, outer_opt)  # do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖
        # print_inside_halt(args, halt, 5)  # todo: remove? temporary for debugging

        # - Opt Step - Average the accumulated gradients and optimize
        from uutils.torch_uu.distributed import is_running_parallel
        # print_inside_halt(args, halt, 6)  # todo: remove? temporary for debugging
        if is_running_parallel(args.rank):
            for p in meta_learner.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / meta_batch_size)
        # print_inside_halt(args, halt, 7)  # todo: remove? temporary for debugging
        outer_opt.step()  # averages gradients across all workers
        # print_inside_halt(args, halt, 8)  # todo: remove? temporary for debugging

        # - Scheduler
        if (args.it % args.log_scheduler_freq == 0) or args.debug:
            scheduler_step(args, scheduler)

        # - Convergence (or idx + 1 == n, this means you've done n loops where idx = it or epoch_num).
        halt: bool = check_halt(args)

        # - go to next it & before that check if we should halt
        args.it += 1

        # - log full stats
        print_dist(msg=f'[{args.it=}] {train_loss=}, {train_acc=}', rank=args.rank, flush=True)
        # when logging after +=1, log idx will be wrt real idx i.e. 0 doesn't mean first it means true 0
        if args.it % args.log_freq == 0 or halt or args.debug:
            step_name: str = 'epoch_num' if 'epochs' in args.training_mode else 'it'
            log_train_val_stats(args, args.it, step_name, train_loss, train_acc, training=True)
            # args.convg_meter.update(train_loss)

        # - break out of the inner loop to start halting, the outer loop will terminate too since halt is True.
        if halt:
            break
