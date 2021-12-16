import torch
from torch import nn, Tensor

from uutils.torch_uu.metrics.confidence_intervals import torch_compute_confidence_interval
from uutils.torch_uu.metrics.metrics import accuracy


class ClassificationSLAgent(nn.Module):

    def __init__(self, model: nn.Module, loss: nn.Module):
        self.model = model
        self.loss = loss

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        batch_x, batch_y = batch
        # todo - process batch to gpu
        logits: Tensor = self.mdl(batch_x)
        loss: Tensor = self.loss(logits)
        acc, = accuracy(logits, batch_y)
        assert loss.size() == torch.Size([]) == acc.size()
        return loss, acc

    def eval_forward(self, batch: Tensor, training: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_x, batch_y = batch
        # todo - process batch to gpu
        B: int = batch_x.size(0)
        with torch.no_grad():  # note, this might not be needed in meta-eval due to MAML using grads at eval
            # - to make sure we get the [B] tensor to compute confidence intervals/error bars
            original_reduction: str = self.loss.reduction
            self.loss.reduction = 'none'
            self.model.train() if training else self.model.eval()

            # -- forward
            logits: Tensor = self.mdl(batch_x)
            loss: Tensor = self.loss(logits)
            acc, = accuracy(logits, batch, reduction='acc_none')
            assert loss.size() == torch.Size([B]) == acc.size()

            # - return loss to normal
            self.loss.reduction = original_reduction
            self.model.train()

            # - stats
            eval_loss_mean, eval_loss_ci = torch_compute_confidence_interval(loss)
            eval_acc_mean, eval_acc_ci = torch_compute_confidence_interval(acc)
        return eval_loss_mean, eval_loss_ci, eval_acc_mean, eval_acc_ci


class RegressionSLAgent(nn.Module):
    """
    todo - main difference is to use R2 score (perhaps squeezed, for accuracy instead of loss), and
        to make sure it has the options for reduction.
    """

    def __init__(self, model: nn.Module, loss: nn.Module):
        self.model = model
        self.loss = loss
