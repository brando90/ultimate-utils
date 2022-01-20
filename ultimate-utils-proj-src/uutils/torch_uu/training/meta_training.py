"""
"""
from argparse import Namespace

import uutils
from uutils.logging_uu.wandb_logging.meta_learning import log_train_val_stats
from uutils.torch_uu import process_meta_batch
from uutils.torch_uu.checkpointing_uu.meta_learning import save_for_meta_learning
from uutils.torch_uu.training.common import gradient_clip


def meta_train_fixed_iterations(args: Namespace, training: bool = True):
    """
    Train using the meta-training (e.g. episodic training) over batches of tasks using a fixed number of iterations
    assuming the number of tasks is small i.e. one epoch is doable and not infinite/super exponential
    (e.g. in regression when a task can be considered as a function).

    Note: if num tasks is small then we have two loops, one while we have not finished all fixed its and the other
    over the dataloader for the tasks.
    """
    print('Starting training!')

    # bar_it = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    bar_it = uutils.get_good_progressbar(max_value=args.num_its)
    args.meta_learner.train() if training else args.meta_learner.eval()
    while True:
        for batch_idx, batch in enumerate(args.dataloaders['train']):
            spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

            # - clean gradients, especially before meta-learner is ran since it uses gradients
            args.outer_opt.zero_grad()

            # - forward pass A(f)(x)
            train_loss, train_acc, train_loss_std, train_acc_std = args.meta_learner(spt_x, spt_y, qry_x, qry_y)

            # - outer_opt step
            gradient_clip(args, args.outer_opt)  # do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖
            args.outer_opt.step()

            # - scheduler
            if (args.it % 500 == 0 and args.it != 0 and hasattr(args, 'scheduler')) or args.debug:
                args.scheduler.step() if (args.scheduler is not None) else None

            # -- log it stats
            log_train_val_stats(args, args.it, train_loss, train_acc, valid=meta_eval, bar=bar_it,
                                log_freq=100, ckpt_freq=100,
                                save_val_ckpt=True, log_to_wandb=args.log_to_wandb,
                                force_log=args.force_log)
            # log_sim_to_check_presence_of_feature_reuse(args, args.it,
            #                                            spt_x, spt_y, qry_x, qry_y,
            #                                            log_freq_for_detection_of_feature_reuse=int(args.num_its//3)
            #                                            , parallel=False)

            # - break
            halt: bool = args.it >= args.num_its - 1
            if halt:
                return train_loss, train_acc

            args.it += 1


# - evaluation code

def meta_eval(args: Namespace, training: bool = True, val_iterations: int = 0, save_val_ckpt: bool = True,
              split: str = 'val') -> tuple:
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
    # return eval_loss, eval_acc
