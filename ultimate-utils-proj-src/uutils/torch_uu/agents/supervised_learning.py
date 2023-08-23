"""
Design pattern: since each model has it's own data, we wrap each model in an agent class that can take care of the
specifics of the data manipulation form a batch. e.g. if they are tensors, or symbolic objects or NL etc.

Note:
    - the forward functions are used in training so calling .item() on the values it returns will create issues for
    training.
"""
from argparse import Namespace

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.distributed import process_batch_ddp, process_batch_ddp_union_rfs, is_running_parallel
from uutils.torch_uu.metrics.confidence_intervals import torch_compute_confidence_interval, mean_confidence_interval
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
            self.criterion = args.loss

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
            eval_loss_mean, eval_loss_ci = mean_confidence_interval(loss)
            eval_acc_mean, eval_acc_ci = mean_confidence_interval(acc)
        return eval_loss_mean, eval_loss_ci, eval_acc_mean, eval_acc_ci

    def get_lists_accs_losses(self, batch: Tensor,
                              training: bool,
                              as_list_floats: bool = False,
                              as_numpy_data: bool = False,  # careful, if NOT set you might get GPU memory issues
                              ) -> tuple[iter, iter]:
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
            loss: list[float] = loss.detach().cpu().numpy().tolist()
            acc: list[float] = acc.detach().cpu().numpy().tolist()

        if as_numpy_data:
            loss: np.ndarray = loss.detach().cpu().numpy()
            acc: np.ndarray = acc.detach().cpu().numpy()

        # - return loss to normal
        self.loss.reduction = original_reduction
        self.model.train()
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

class GPT2SLAgent(Agent):
    """
    For training the GPT2 model
    """


    def __init__(self, args, model):
        super().__init__()
        self.is_parallel = is_running_parallel(args.rank)
        self.model = model

    def forward(self, batch, training = True, half_loss = False):
        # print(batch)
        # for name, param in self.model.named_parameters():
        #     print(name, ":", param.data, "req_grad:", param.requires_grad)
        #     print("grad:", param.grad)
        # exit()
        self.model.train() if training else self.model.eval()
        # import time
        # t_bl = time.time()
        logits = self.model(batch[0])
        # t_al = time.time()
        # print("t_logits - t_blogits=", t_al - t_bl)
        # targets = batch[1].to(logits.device)
        # t3 = time.time()
        # print("t3 - t_logits=", t3 - t_al)
        # print("forward called with half_loss = ", half_loss)
        if half_loss:
            loss, acc = self.loss_for_half(logits, batch[1], first_half = False)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch[1].view(-1), ignore_index=-1)
            # t4 = time.time()
            # print("t4 - t3=", t4-t3)
            preds = torch.argmax(logits, dim = 2)
            # t5 = time.time()
            # print("t5-t4=",t5-t4)
            acc = torch.sum((preds == batch[1]))/(batch[1].shape[0]*batch[1].shape[1])
        # t6 = time.time()
        # print("t6-t5=", t6-t5)
        # assert loss.size() == torch.Size([])
        # assert acc.size() == torch.Size([])
        # if training:
        #     loss, acc = self.model(batch[0], batch[1])
        # else:
        #     loss, acc = self.model(batch[0])

        return loss, acc

    def eval_forward(self, batch, training = False, half_loss = False):
        B, W = batch[1].shape
        with torch.no_grad():
            self.model.train() if training else self.model.eval()
            logits = self.model(batch[0])
            targets = batch[1].to(logits.device)
            # print("logits.size=", logits.size())
            # print("eval_forward batch_size = ",B)
            if half_loss:
                # print("half_loss called")
                loss, acc = self.loss_for_half(logits, targets, first_half = False, reduction = 'none')
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction = 'none')
                # print("loss.size=", loss.size())
                preds = torch.argmax(logits, dim = 2)
                # print("preds.size=", preds.size())
                acc = torch.sum((preds == targets), dim = 1)/targets.shape[1]
                # print("acc.size=", acc.size())

            # assert loss.size() == torch.Size([B*W])
            assert acc.size() == torch.Size([B])

            self.model.train()
            # - stats
            eval_loss_mean, eval_loss_ci = torch_compute_confidence_interval(loss)
            eval_acc_mean, eval_acc_ci = torch_compute_confidence_interval(acc)
        return eval_loss_mean, eval_loss_ci, eval_acc_mean, eval_acc_ci


    def loss_for_half(self, logits, target, first_half, reduction = 'mean'):
        """
        Evaluates loss on either the first half or the second half of the batch
        """
        if self.is_parallel:
            mid_point = self.model.module.config.block_size//2
        else:
            mid_point = self.model.config.block_size//2
        # print("logits.size:", logits.size())
        # print("target.size:", target.size())
        if first_half:
            cut_logits = logits[:, :mid_point, :]
            cut_target = target[:, :mid_point]
        else:
            cut_logits = logits[:, mid_point:, :]
            cut_target = target[:, mid_point:]

        # print("cut_logits.size:", cut_logits.size())
        # print("cut_target.size:", cut_target.size())

        # TODO: reshape can make it slower
        loss = F.cross_entropy(cut_logits.reshape(-1, cut_logits.size(-1)), cut_target.reshape(-1), ignore_index=-1, reduction = reduction)

        preds = torch.argmax(cut_logits, dim = 2)
        # print("preds.size:", preds.size())

        if reduction == 'none':
            # retain information for the batch to compute confidence intervals
            acc = torch.sum((preds == cut_target), dim = 1)/cut_target.shape[1]
        else:
            acc = torch.sum((preds == cut_target))/(cut_target.shape[0]*cut_target.shape[1])

        return loss, acc








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
