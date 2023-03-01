import logging
from argparse import Namespace
from typing import Any, Optional

from torch import Tensor, tensor, nn

from uutils.torch_uu.agents.common import Agent

# -- get loss & acc
from uutils.torch_uu.training.common import get_data


def do_eval(args: Namespace,
            model: Agent,
            dataloaders,
            split: str = 'val',
            training: bool = True,  # True to avoid different tasks: https://stats.stackexchange.com/a/551153/28986
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    note:
        - note we have "get_eval" function cuz the api for data loaders for meta-learning is different from the SL one.
        Despite the Agent having the same api as the MetaLearner object.
        - assumption: your agent has the .forward interface needed
    """
    # - get loss & acc
    if not hasattr(model, 'eval_forward'):
        from uutils.torch_uu.metrics.confidence_intervals import mean_confidence_interval
        data: Any = get_data(dataloaders, split)
        losses, accs = model.get_lists_accs_losses(data, training)
        loss, loss_ci = mean_confidence_interval(losses)
        acc, acc_ci = mean_confidence_interval(accs)
    else:
        data: Any = get_data(dataloaders, split)
        loss, loss_ci, acc, acc_ci = model.eval_forward(data, training)
    return loss, loss_ci, acc, acc_ci


# - tests, tutorials, examples

def eval_test_():
    # - usl
    # - torchmeta
    # - l2l
    pass


def train_test_():
    pass
