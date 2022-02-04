"""
"""
from argparse import Namespace
from typing import Any

from progressbar import ProgressBar
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import uutils
from uutils.logging_uu.wandb_logging.supervised_learning import log_train_val_stats
from uutils.torch_uu.checkpointing_uu.meta_learning import save_for_meta_learning
from uutils.torch_uu.training.common import gradient_clip, scheduler_step, check_halt, get_trainer_progress_bar, \
    ConvergenceMeter

from pdb import set_trace as st


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
    args.convg_meter: ConvergenceMeter = ConvergenceMeter(name='train loss',
                                                          convergence_patience=args.train_convergence_patience)
    # log_zeroth_step(args, model)
    halt: bool = False
    while not halt:
        opt.zero_grad()
        train_loss, train_acc, train_loss_std, train_acc_std = meta_learner(train_batch, call_backward=True)
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
                                ) -> tuple[Tensor, Tensor]:
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
    while not halt:
        for batch_idx, batch in enumerate(dataloaders['train']):
            outer_opt.zero_grad()
            train_loss, train_acc, train_loss_std, train_acc_std = meta_learner(batch, call_backward=True)
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
                # args.convg_meter.update(train_loss)

            # - break out of the inner loop to start halting, the outer loop will terminate too since halt is True.
            if halt:
                break

    return train_loss, train_acc


def meta_train_iterations_ala_l2l(args: Namespace,

                              ):


    # -
    meta_batch_size = 32 // args.world_size,

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = torch.optim.Adam(maml.parameters(), meta_lr)
    opt = cherry.optim.Distributed(maml.parameters(), opt=opt, sync=1)
    opt.sync_parameters()
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    print(rank, ':', device)

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                adaptation_steps,
                shots,
                ways,
                device,
            )
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                adaptation_steps,
                shots,
                ways,
                device,
            )
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        if rank == 0:
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / meta_batch_size)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            print('Meta Valid Error', meta_valid_error / meta_batch_size)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()  # averages gradients across all workers

