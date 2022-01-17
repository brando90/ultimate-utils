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
    batch: Any = next(iter(dataloaders[split]))
    val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(batch, training)
    return val_loss, val_loss_ci, val_acc, val_acc_ci

# def eval(args: Namespace,
#          model: nn.Module,
#          training: bool = False,
#          val_iterations: int = 0,
#          split: str = 'val'
#          ) -> tuple:
#     """
#
#     Note:
#         -  Training=True for eval only for meta-learning, here we do want .eval(), but then undo it
#
#     ref for BN/eval:
#         - For SL: do .train() for training and .eval() for eval in SL.
#         - For Meta-learning do train in both, see: https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
#     """
#     assert val_iterations == 0, f'Val iterations has to be zero but got {val_iterations}, ' \
#                                 f'if you want more precision increase (meta) batch size.'
#     args.meta_learner.train() if training else args.meta_learner.eval()
#     for batch_idx, eval_batch in enumerate(args.dataloaders[split]):
#         eval_loss_mean, eval_loss_ci, eval_acc_mean, eval_acc_ci = model.eval_forward(eval_batch)
#         if batch_idx >= val_iterations:
#             break
#     return eval_loss_mean, eval_loss_ci, eval_acc_mean, eval_acc_ci
