from argparse import Namespace
from typing import Any

from torch import Tensor

from uutils.torch_uu.agents.common import Agent


def eval_sl(args: Namespace,
            model: Agent,
            dataloaders: dict,
            split: str = 'val',
            training: bool = False,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Evaluate the current model on the eval set (val set strongly recommended).

    note:
        - Training=True for eval only for meta-learning, here we do want .eval(). This is because .train()
        uses batch stats while .eval() uses the running means.
        See: https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
        - Note, we don't need to loop through the data loader, we can get confidence intervals for the mean error
        from 1 batch - since we are estimating the mean loss from the eval set.
    """
    if isinstance(dataloaders, dict):
        batch: Any = next(iter(dataloaders[split]))
        val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(batch, training)
    else:
        # hack for l2l
        from learn2learn.data import TaskDataset
        split: str = 'validation' if split == 'val' else split
        task_dataset: TaskDataset = getattr(args.tasksets, split)
        val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(task_dataset, training)
    return val_loss, val_loss_ci, val_acc, val_acc_ci

# # - evaluation code
#
# def meta_eval(args: Namespace,
#               training: bool = True,
#               val_iterations: int = 0,
#               save_val_ckpt: bool = True,
#               split: str = 'val',
#               ) -> tuple:
#     """
#     Evaluates the meta-learner on the given meta-set.
#
#     ref for BN/eval:
#         - tldr: Use `mdl.train()` since that uses batch statistics (but inference will not be deterministic anymore).
#         You probably won't want to use `mdl.eval()` in meta-learning.
#         - https://stackoverflow.com/questions/69845469/when-should-one-call-eval-and-train-when-doing-maml-with-the-pytorch-highe/69858252#69858252
#         - https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
#         - https://github.com/tristandeleu/pytorch-maml/issues/19
#     """
#     # - need to re-implement if you want to go through the entire data-set to compute an epoch (no more is ever needed)
#     assert val_iterations == 0, f'Val iterations has to be zero but got {val_iterations}, if you want more precision increase (meta) batch size.'
#     args.meta_learner.train() if training else args.meta_learner.eval()
#     for batch_idx, batch in enumerate(args.dataloaders[split]):
#         eval_loss, eval_acc, eval_loss_std, eval_acc_std = args.meta_learner.forward_eval(batch, training=training)
#
#         # store eval info
#         if batch_idx >= val_iterations:
#             break
#
#     if float(eval_loss) < float(args.best_val_loss) and save_val_ckpt:
#         args.best_val_loss = float(eval_loss)
#         save_for_meta_learning(args, ckpt_filename='ckpt_best_val.pt')
#     return eval_loss, eval_acc, eval_loss_std, eval_acc_std
