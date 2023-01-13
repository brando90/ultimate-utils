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
        """
        Note:
            - does not need get_lists_accs_losses (in contrast to meta-learning) because in meta-learning with episodic
            meta-learning we always compute the list of losses/accs for both forward & eval pass. But in SL we only
            need the list info when explicitly needed and in eval (so we can compute CIs). Basically, the train forward
            in eval just returns the single float/tensor element for loss/acc -- doesn't explicitly use the iter representations.
        """
        self.model.train() if training else self.model.eval()
        batch_x, batch_y = process_batch_ddp(self.args, batch)
        logits: Tensor = self.model(batch_x)
        loss: Tensor = self.loss(logits, batch_y)
        acc, = accuracy(logits, batch_y)
        assert loss.size() == torch.Size([])
        assert acc.size() == torch.Size([])
        return loss, acc

    def eval_forward(self, batch: Tensor, training: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Note:
            - reason this is different than forward is because we have a torch.no_grad() and because in eval we are
            also returning the CI's, so we need to get the flat tensor/list/iterable to compute those, so we can't just
            get the loss as it's usually done in forward passes in pytorch that only return the loss/acc. We use the
            accs & losses info basically.
        """
        batch_x, batch_y = process_batch_ddp(self.args, batch)
        B: int = batch_x.size(0)
        with torch.no_grad():  # note, this might not be needed in meta-eval due to MAML using grads at eval
            # - to make sure we get the [B] tensor to compute confidence intervals/error bars
            loss, acc = self.get_lists_accs_losses(batch, training)
            assert loss.size() == torch.Size([B])
            assert acc.size() == torch.Size([B])

            # - stats, compute this in context manager to avoid memory issues (perhaps not needed but safer)
            eval_loss_mean, eval_loss_ci = torch_compute_confidence_interval(loss)
            eval_acc_mean, eval_acc_ci = torch_compute_confidence_interval(acc)
        return eval_loss_mean, eval_loss_ci, eval_acc_mean, eval_acc_ci

    def get_lists_accs_losses(self, batch: Tensor, training: bool, as_list_floats: bool = False) -> tuple[iter, iter]:
        """
        Note:
            - train doesn't need this but eval does. Also, it could be useful to get lists to do other analysis with them if needed.
        """
        # -- get data
        batch_x, batch_y = process_batch_ddp(self.args, batch)
        B: int = batch_x.size(0)

        # -- get ready to collect losses and accs
        original_reduction: str = self.loss.reduction
        self.loss.reduction = 'none'
        self.model.train() if training else self.model.eval()

        # -- "forward"
        logits: Tensor = self.model(batch_x)
        loss: Tensor = self.loss(logits, batch_y)
        acc, = accuracy(logits, batch_y, reduction='none')
        assert loss.size() == torch.Size([B])
        assert acc.size() == torch.Size([B])

        # -- convert losses & accs to list of floats
        if as_list_floats:
            # todo: not sure if needed, but in the past keeping everything as tensors creates GPU memory issues since it never forget the computation graph
            #   might be fine due to torch.no_grad() but not sure. Perhaps assert they are of type list[float]?
            raise NotImplementedError

        # - return loss to normal
        self.loss.reduction = original_reduction
        self.model.train()

        # -- return
        assert loss.size() == torch.Size([B])
        assert acc.size() == torch.Size([B])
        return loss, acc



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
