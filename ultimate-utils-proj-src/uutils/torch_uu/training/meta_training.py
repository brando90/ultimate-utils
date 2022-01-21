"""
"""
from argparse import Namespace

from torch import Tensor

import uutils
from uutils.logging_uu.wandb_logging.meta_learning import log_train_val_stats
from uutils.torch_uu import process_meta_batch
from uutils.torch_uu.checkpointing_uu.meta_learning import save_for_meta_learning
from uutils.torch_uu.training.common import gradient_clip, scheduler_step, check_halt


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
                log_train_val_stats(args, args.it, step_name, train_loss, train_acc)
                args.convg_meter.update(train_loss)

            # - break out of the inner loop to start halting, the outer loop will terminate too since halt is True.
            if halt:
                break

    return train_loss, train_acc


# - evaluation code

def meta_eval(args: Namespace,
              training: bool = True,
              val_iterations: int = 0,
              save_val_ckpt: bool = True,
              split: str = 'val',
              ) -> tuple:
    """
    Evaluates the meta-learner on the given meta-set.

    ref for BN/eval:
        - tldr: Use `mdl.train()` since that uses batch statistics (but inference will not be deterministic anymore).
        You probably won't want to use `mdl.eval()` in meta-learning.
        - https://stackoverflow.com/questions/69845469/when-should-one-call-eval-and-train-when-doing-maml-with-the-pytorch-highe/69858252#69858252
        - https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
        - https://github.com/tristandeleu/pytorch-maml/issues/19
    """
    # - need to re-implement if you want to go through the entire data-set to compute an epoch (no more is ever needed)
    assert val_iterations == 0, f'Val iterations has to be zero but got {val_iterations}, if you want more precision increase (meta) batch size.'
    args.meta_learner.train() if training else args.meta_learner.eval()
    for batch_idx, batch in enumerate(args.dataloaders[split]):
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

        # Forward pass
        # eval_loss, eval_acc = args.meta_learner(spt_x, spt_y, qry_x, qry_y)
        eval_loss, eval_acc, eval_loss_std, eval_acc_std = args.meta_learner(spt_x, spt_y, qry_x, qry_y,
                                                                             training=training)

        # store eval info
        if batch_idx >= val_iterations:
            break

    if float(eval_loss) < float(args.best_val_loss) and save_val_ckpt:
        args.best_val_loss = float(eval_loss)
        save_for_meta_learning(args, ckpt_filename='ckpt_best_val.pt')
    return eval_loss, eval_acc, eval_loss_std, eval_acc_std