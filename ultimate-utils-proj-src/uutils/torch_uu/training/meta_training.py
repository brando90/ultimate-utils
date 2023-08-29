"""
"""
from argparse import Namespace
from typing import Any

import torch
from learn2learn.data import TaskDataset
import progressbar
from progressbar import ProgressBar
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import uutils
from uutils.logging_uu.wandb_logging.meta_learning import log_zeroth_step
from uutils.logging_uu.wandb_logging.supervised_learning import log_train_val_stats
from uutils.torch_uu.checkpointing_uu.meta_learning import save_for_meta_learning
from uutils.torch_uu.training.common import gradient_clip, scheduler_step, check_halt, get_trainer_progress_bar, \
    ConvergenceMeter
from uutils.torch_uu.distributed import print_dist, is_lead_worker

from pdb import set_trace as st
import math


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
    train_batch: Any = next(iter(dataloaders['train']))

    # first batch
    args.it = 0  # training a single batch shouldn't need to use ckpts so this is ok
    args.best_val_loss = float('inf')  # training a single batch shouldn't need to use ckpts so this is ok

    # - create progress bar
    args.bar: ProgressBar = get_trainer_progress_bar(args)

    # - train in epochs
    args.convg_meter = ConvergenceMeter(name='train loss', convergence_patience=args.train_convergence_patience)
    # log_zeroth_step(args, model) # not needed for fit single batch
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
    # - imports
    from uutils.torch_uu.dataloaders.meta_learning.l2l_to_torchmeta_dataloader import \
        episodic_batch_2_task_dataset

    # - start!
    print('----> Starting training!')
    print(f'{meta_train_fixed_iterations=}')

    # - create progress bar
    args.bar: ProgressBar = get_trainer_progress_bar(args)
    meta_learner.train() if training else meta_learner.eval()
    halt: bool = False

    # - meta-train
    args.convg_meter = ConvergenceMeter(name='train loss', convergence_patience=args.train_convergence_patience)
    log_zeroth_step(args, meta_learner)
    # - continually meta-train until halt condition is met (usually convergence or reaching max its)
    print('--> Starting the episodic meta-training loop. Getting batches ala episodic way (e.g. ala torchmeta way)...')
    loader: DataLoader = dataloaders['train']
    while not halt:
        for batch_idx, batch in enumerate(loader):
            outer_opt.zero_grad()
            assert outer_opt is args.opt

            if hasattr(loader, 'episodic_batch_2_task_dataset'):  # mainly for mds
                batch: TaskDataset = episodic_batch_2_task_dataset(batch, loader, meta_learner)
            train_loss, train_acc = meta_learner(batch, call_backward=True)
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
                args.convg_meter.update(train_loss)

            # - break out of the inner loop to start halting, the outer loop will terminate too since halt is True.
            if halt:
                break
    return train_loss, train_acc


def meta_train_gpt2(args: Namespace, meta_learner, dataloaders: dict, outer_opt, scheduler, training: bool = True):
    print('Starting training!')
    meta_batch_size: int = args.batch_size // args.world_size
    # meta_batch_size: int = max(args.batch_size // args.world_size, 1)
    # assert args.batch_size >= args.world_size, f'If batch size is smaller training might be slightly wrong when in distributed.'

    # print(args.batch_size, meta_batch_size, "args and Meta BatchSize")
    # args.bar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    args.bar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    args.convg_meter = ConvergenceMeter(name='train loss', convergence_patience=args.train_convergence_patience)

    meta_learner.train() if training else meta_learner.eval()
    halt: bool = False

    # ----added - 0th iter---#
    log_zeroth_step(args, meta_learner)
    for i, batch in enumerate(dataloaders['train']):
        if args.opt_option == 'adamw_gpt' and args.scheduler_option == 'None':
            ####### clumsy workaround for custom scheduler
            # 1) linear warmup for warmup_iters steps
            if args.it < args.scheduler_hps['warmup_iters']:
                lr =  args.lr * args.it / args.scheduler_hps['warmup_iters']
            # 2) if iter > lr_decay_iters, return min learning rate
            elif args.it > args.scheduler_hps['lr_decay_iters']:
                lr =  args.scheduler_hps['min_lr']
            # 3) in between, use cosine decay down to min learning rate
            else:
                decay_ratio = (args.it - args.scheduler_hps['warmup_iters']) / (args.scheduler_hps['lr_decay_iters'] - args.scheduler_hps['warmup_iters'])
                assert 0 <= decay_ratio <= 1
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
                lr =  args.scheduler_hps['min_lr'] + coeff * (args.lr - args.scheduler_hps['min_lr'])

            for param_group in outer_opt.param_groups:
                param_group['lr'] = lr

        batch = (batch[0].to(args.device), batch[1].to(args.device))
        outer_opt.zero_grad()

        # - forward pass. Since the data fetching is different for l2l we do it this way
        train_loss, train_loss_ci, train_acc, train_acc_ci = meta_learner(batch, call_backward=True)
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


        # - go to next it & before that check if we should halt
        args.it += 1

        # - log full stats
        print_dist(msg=f'[{args.it=}] {train_loss=}, {train_acc=}, {args.convg_meter.counts_above_current_lowest=}', rank=args.rank, flush=True)
        # when logging after +=1, log idx will be wrt real idx i.e. 0 doesn't mean first it means true 0
        if args.it % args.log_freq == 0 or halt or args.debug:
            step_name: str = 'epoch_num' if 'epochs' in args.training_mode else 'it'
            log_train_val_stats(args, args.it, step_name, train_loss, train_acc, training=True)
        
        args.convg_meter.update(train_loss)
        # - Convergence (or idx + 1 == n, this means you've done n loops where idx = it or epoch_num).
        halt: bool = check_halt(args)

        # - break out of the inner loop to start halting, the outer loop will terminate too since halt is True.
        if halt:
            print("halting...")
            log_train_val_stats(args, args.it, step_name, train_loss, train_acc)
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
    print(f'{meta_train_iterations_ala_l2l=}')
    meta_batch_size: int = args.batch_size // args.world_size
    # meta_batch_size: int = max(args.batch_size // args.world_size, 1)
    # assert args.batch_size >= args.world_size, f'If batch size is smaller training might be slightly wrong when in distributed.'

    # - create progress bar
    args.bar: ProgressBar = get_trainer_progress_bar(args)
    meta_learner.train() if training else meta_learner.eval()
    halt: bool = False

    # - meta-train
    args.convg_meter = ConvergenceMeter(name='train loss', convergence_patience=args.train_convergence_patience)
    log_zeroth_step(args, meta_learner)
    # - continually meta-train until halt condition is met (usually convergence or reaching max its)
    while not halt:
        outer_opt.zero_grad()

        # - forward pass. Since the data fetching is different for l2l we do it this way
        task_dataset: TaskDataset = args.dataloaders.train
        train_loss, train_acc = meta_learner(task_dataset, call_backward=True)
        # train_loss.backward()  # NOTE: backward was already called in meta-learner due to MEM optimization.
        assert outer_opt.param_groups[0]['params'][0].grad is not None

        # - Grad clip  (optional)
        gradient_clip(args, outer_opt)  # do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖

        # - Opt Step - Average the accumulated gradients and optimize
        from uutils.torch_uu.distributed import is_running_parallel
        if is_running_parallel(args.rank):
            for p in meta_learner.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / meta_batch_size)
        outer_opt.step()  # averages gradients across all workers

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
            args.convg_meter.update(train_loss)

        # - break out of the inner loop to start halting, the outer loop will terminate too since halt is True.
        if halt:
            break
    return train_loss, train_acc

# - tests tutorials

def training_test_():
    # - torchmeta
    from uutils.argparse_uu.meta_learning import get_args_mi_torchmeta_default
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
    args: Namespace = get_args_mi_torchmeta_default()
    get_and_create_model_opt_scheduler_for_run(args)
    args.agent = MAMLMetaLearner(args, args.model)
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
    args.dataloaders = get_meta_learning_dataloaders(args)
    train_loss, train_acc = meta_train_fixed_iterations(args, args.agent, args.dataloaders, args.opt, args.scheduler)
    print(f'{train_loss, train_acc=}')
    # - l2l
    from uutils.argparse_uu.meta_learning import get_args_mi_l2l_default
    from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearnerL2L
    args: Namespace = get_args_mi_l2l_default()
    get_and_create_model_opt_scheduler_for_run(args)
    args.agent = MAMLMetaLearnerL2L(args, args.model)
    args.dataloaders = get_l2l_tasksets(args)
    train_loss, train_acc = meta_train_iterations_ala_l2l(args, args.agent, args.opt, args.scheduler)
    print(f'{train_loss, train_acc=}')

# - run __main__

if __name__ == '__main__':
    import time
    start = time.time()
    training_test_()
    print(f'Done in {time.time() - start} seconds.')