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

from progressbar import ProgressBar
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from uutils.logging_uu.wandb_logging.supervised_learning import log_train_val_stats, log_zeroth_step, \
    log_train_val_stats_simple
from uutils.torch_uu import AverageMeter
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.distributed import print_dist, is_lead_worker
from uutils.torch_uu.training.common import get_trainer_progress_bar, scheduler_step, check_halt, gradient_clip, \
    ConvergenceMeter


def train_agent_fit_single_batch(args: Namespace,
                                 model: Agent,
                                 dataloaders: dict,
                                 opt: Optimizer,
                                 scheduler: _LRScheduler,
                                 acc_tolerance: float = 1.0,
                                 train_loss_tolerance: float = 0.01,
                                 ):
    """
    Train for a single batch

    Note:
        - eval_sl always chooses a different val set. So the val set should be stuck at ~0 if the model is only
        fitting 1 batch (as this code does).
    """
    train_batch: Any = next(iter(dataloaders['train']))

    # first batch
    args.it = 0  # training a single batch shouldn't need to use ckpts so this is ok
    args.best_val_loss = float('inf')  # training a single batch shouldn't need to use ckpts so this is ok

    # - create progress bar
    args.bar: ProgressBar = get_trainer_progress_bar(args)

    # - train in epochs
    args.convg_meter: ConvergenceMeter = ConvergenceMeter(name='train loss', convergence_patience=args.train_convergence_patience)
    log_zeroth_step(args, model)
    halt: bool = False
    while not halt:
        opt.zero_grad()
        train_loss, train_acc = model(train_batch, training=True)
        train_loss.backward()  # each process synchronizes its gradients in the backward pass
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
            log_train_val_stats(args, args.it, step_name, train_loss.item(), train_acc.item())

        if halt:
            break

    return train_loss, train_acc


def train_agent_iterations(args: Namespace,
                           model: Agent,
                           dataloaders: dict,
                           opt: Optimizer,
                           scheduler: _LRScheduler,
                           halt_loss: str = 'train',
                           target_loss = None
                           ) -> tuple[Tensor, Tensor]:
    """
    Trains models wrt to number of iterations given. Should halt once the number of iterations desired is reached. 
    """
    print_dist('Starting training...', args.rank)

    # - create progress bar
    args.bar: ProgressBar = get_trainer_progress_bar(args)

    # - train in epochs
    if halt_loss == 'train':
        args.convg_meter = ConvergenceMeter(name='train loss', convergence_patience=args.train_convergence_patience)
    else:
        args.convg_meter = ConvergenceMeter(name='val loss', convergence_patience=args.train_convergence_patience, target_lowest = target_loss)
    log_zeroth_step(args, model)
    halt: bool = False
    # -- continually try to train accross the an entire epoch but stop once the number of iterations desired is reached
    while not halt:
        # -- train for one epoch
        for i, batch in enumerate(dataloaders['train']):
            # import time
            # t_bto = time.time()
            batch = (batch[0].to(args.device), batch[1].to(args.device))
            opt.zero_grad()
            # t_bm = time.time()
            train_loss, train_acc = model(batch, training=True)
            # t_am = time.time()
            train_loss.backward()  # each process synchronizes its gradients in the backward pass
            # t_lb = time.time()
            gradient_clip(args, opt)
            opt.step()  # the right update is done since all procs have the right synced grads
            # t_opt = time.time()
            # print("t_bto - t_bm=", t_bm - t_bto)
            # print("t_am-t_bm", t_am-t_bm)
            # print("t_lb-t_am", t_lb-t_am)
            # print("t_opt - t_lb", t_opt-t_lb)

            # - scheduler
            if (args.it % args.log_scheduler_freq == 0) or args.debug:
                scheduler_step(args, scheduler)

            # - convergence (or idx + 1 == n, this means you've done n loops where idx = it or epoch_num).
            halt: bool = check_halt(args)

            # - go to next it & before that check if we should halt
            args.it += 1

            # - log full stats
            # when logging after +=1, log idx will be wrt real idx i.e. 0 doesn't mean first it means true 0
            print_dist(msg=f'[{args.epoch_num=}, {i=}] {train_loss=}, {train_acc=}', rank=args.rank, flush=True)
            if i % args.log_freq == 0 or halt or args.debug:
                step_name: str = 'epoch_num' if 'epochs' in args.training_mode else 'it'
                log_train_val_stats(args, args.it, step_name, train_loss, train_acc)
                # print_dist(msg=f'[{args.epoch_num=}, {i=}] {train_loss=}, {train_acc=}', rank=args.rank, flush=True)
                if halt_loss == 'train':
                    args.convg_meter.update(train_loss)
                else:
                    args.convg_meter.update(val_loss)

            # - break out of the inner loop to start halting, the outer loop will terminate too since halt is True.
            if halt:
                break

    return train_loss, train_acc


def train_agent_epochs(args: Namespace,
                       model: Agent,
                       dataloaders: dict,
                       opt: Optimizer,
                       scheduler: _LRScheduler,
                       ) -> tuple[Tensor, Tensor]:
    """
    Trains model one epoch at a time - i.e. it's epochs based rather than iteration based.
    """
    print_dist('Starting training...', args.rank)
    B: int = args.batch_size

    # - create progress bar
    args.bar: ProgressBar = get_trainer_progress_bar(args)

    # - train in epochs
    args.convg_meter = ConvergenceMeter(name='train loss', convergence_patience=args.train_convergence_patience)
    #log_zeroth_step(args, model)
    halt: bool = False

    # # ----added - 0th iter---#
    # args.epochs_num = 0
    # avg_loss = AverageMeter('train loss')
    # avg_acc = AverageMeter('train accuracy')
    # for i, batch in enumerate(dataloaders['train']):
    #     # print("batch.device = ", batch.device)
    #     #opt.zero_grad()
    #     train_loss, train_acc = model(batch, training=True)
    #     #train_loss.backward()  # each process synchronizes its gradients in the backward pass
    #     #gradient_clip(args, opt)
    #     #opt.step()  # the right update is done since all procs have the right synced grads

    #     # - meter updates
    #     avg_loss.update(train_loss.item(), B), avg_acc.update(train_acc, B)
    #     if args.debug:
    #         print_dist(msg=f'[{args.epoch_num=}, {i=}] {train_loss=}, {train_acc=}', rank=args.rank, flush=True)

    step_name: str = 'epoch_num' if 'epochs' in args.training_mode else 'it'
    # log_train_val_stats(args, args.epoch_num, step_name, avg_loss.item(), avg_acc.item())
    args.epochs_num = 0
    # --------#
    while not halt:
        print("training starting........")
        # -- train for one epoch
        avg_loss = AverageMeter('train loss')
        avg_acc = AverageMeter('train accuracy')
        for i, batch in enumerate(dataloaders['train']):
            #print("batch usl", batch)
            opt.zero_grad()
            train_loss, train_acc = model(batch, training=True)
            train_loss.backward()  # each process synchronizes its gradients in the backward pass
            gradient_clip(args, opt)
            print("taking opt step now..")
            opt.step()  # the right update is done since all procs have the right synced grads

            # - meter updates
            avg_loss.update(train_loss.item(), B), avg_acc.update(train_acc, B)
            if args.debug:
                print_dist(msg=f'[{args.epoch_num=}, {i=}] {train_loss=}, {train_acc=}', rank=args.rank, flush=True)

        # - scheduler, not in first/0th epoch though
        if (args.epoch_num % args.log_scheduler_freq == 0) or args.debug:
            scheduler_step(args, scheduler)

        # convergence (or idx + 1 == n, this means you've done n loops where idx = it or epoch_num).
        halt: bool = check_halt(args)

        # - go to next it & before that check if we should halt
        args.epoch_num += 1

        # - log full epoch stats
        # when logging after +=1, log idx will be wrt real idx i.e. 0 doesn't mean first it means true 0
        if args.epoch_num % args.log_freq == 0 or halt or args.debug:
            step_name: str = 'epoch_num' if 'epochs' in args.training_mode else 'it'
            log_train_val_stats(args, args.epoch_num, step_name, avg_loss.item(), avg_acc.item())
            args.convg_meter.update(avg_loss.item())

    return avg_loss.item(), avg_acc.item()


