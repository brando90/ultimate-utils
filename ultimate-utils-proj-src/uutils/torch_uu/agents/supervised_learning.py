"""
Design pattern: since each model has it's own data, we wrap each model in an agent class that can take care of the
specifics of the data manipulation form a batch. e.g. if they are tensors, or symbolic objects or NL etc.

Note:
    - the forward functions are used in training so calling .item() on the values it returns will create issues for
    training.
"""
from argparse import Namespace

import torch
from torch import nn, Tensor

from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.distributed import process_batch_ddp, process_batch_ddp_union_rfs
from uutils.torch_uu.metrics.confidence_intervals import torch_compute_confidence_interval
from uutils.torch_uu.metrics.metrics import accuracy

from pdb import set_trace as st

class ClassificationSLAgent(Agent):

    def __init__(self,
                 args: Namespace,
                 model: nn.Module,
                 ):
        super().__init__()
        self.args = args
        self.model = model
        if hasattr(args, 'loss'):
            self.loss = nn.CrossEntropyLoss() if args.loss is not None else args.loss

    def forward(self, batch: Tensor, training: bool = True) -> tuple[Tensor, Tensor]:
        self.model.train() if training else self.model.eval()
        batch_x, batch_y = process_batch_ddp(self.args, batch)
        logits: Tensor = self.model(batch_x)
        loss: Tensor = self.loss(logits, batch_y)
        acc, = accuracy(logits, batch_y)
        assert loss.size() == torch.Size([])
        assert acc.size() == torch.Size([])
        return loss, acc

    def eval_forward(self, batch: Tensor, training: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_x, batch_y = process_batch_ddp(self.args, batch)
        B: int = batch_x.size(0)
        with torch.no_grad():  # note, this might not be needed in meta-eval due to MAML using grads at eval
            # - to make sure we get the [B] tensor to compute confidence intervals/error bars
            original_reduction: str = self.loss.reduction
            self.loss.reduction = 'none'
            self.model.train() if training else self.model.eval()

            # -- forward
            logits: Tensor = self.model(batch_x)
            loss: Tensor = self.loss(logits, batch_y)
            acc, = accuracy(logits, batch_y, reduction='none')
            assert loss.size() == torch.Size([B])
            assert acc.size() == torch.Size([B])

            # - return loss to normal
            self.loss.reduction = original_reduction
            self.model.train()

            # - stats
            eval_loss_mean, eval_loss_ci = torch_compute_confidence_interval(loss)
            eval_acc_mean, eval_acc_ci = torch_compute_confidence_interval(acc)
        return eval_loss_mean, eval_loss_ci, eval_acc_mean, eval_acc_ci


class RegressionSLAgent(Agent):
    """
    todo - main difference is to use R2 score (perhaps squeezed, for accuracy instead of loss), and
        to make sure it has the options for reduction.
    """

    def __init__(self, model: nn.Module, loss: nn.Module):
        super().__init__()
        self.model = model
        self.loss = loss


class UnionClsSLAgent(Agent):
    """
    A wraper for the model that is compatible with the SL from rfs:
    ref: https://github.com/WangYueFt/rfs/
    """

    def __init__(self,
                 args: Namespace,
                 model: nn.Module,
                 ):
        super().__init__()
        self.args = args
        self.model = model
        if hasattr(args, 'loss'):
            self.loss = nn.CrossEntropyLoss() if args.loss is not None else args.loss

    def forward(self, batch: Tensor, training: bool = True) -> tuple[Tensor, Tensor]:
        self.model.train() if training else self.model.eval()
        batch_x, batch_y = process_batch_ddp_union_rfs(self.args, batch)
        logits: Tensor = self.model(batch_x)
        loss: Tensor = self.loss(logits, batch_y)
        acc, = accuracy(logits, batch_y)
        assert loss.size() == torch.Size([])
        assert acc.size() == torch.Size([])
        return loss, acc

    def eval_forward(self, batch: Tensor, training: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_x, batch_y = process_batch_ddp_union_rfs(self.args, batch)
        B: int = batch_x.size(0)
        with torch.no_grad():  # note, this might not be needed in meta-eval due to MAML using grads at eval
            # - to make sure we get the [B] tensor to compute confidence intervals/error bars
            original_reduction: str = self.loss.reduction
            assert original_reduction == self.loss.reduction
            self.loss.reduction = 'none'
            assert original_reduction != self.loss.reduction
            self.model.train() if training else self.model.eval()

            # -- forward
            logits: Tensor = self.model(batch_x)
            loss: Tensor = self.loss(logits, batch_y)
            acc, = accuracy(logits, batch_y, reduction='none')
            assert loss.size() == torch.Size([B])
            assert acc.size() == torch.Size([B])

            # - return loss to normal
            self.loss.reduction = original_reduction
            self.model.train()

            # - stats
            eval_loss_mean, eval_loss_ci = torch_compute_confidence_interval(loss)
            eval_acc_mean, eval_acc_ci = torch_compute_confidence_interval(acc)
        return eval_loss_mean, eval_loss_ci, eval_acc_mean, eval_acc_ci
