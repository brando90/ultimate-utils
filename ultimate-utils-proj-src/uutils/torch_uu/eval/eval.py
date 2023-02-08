"""

TODO: note we can change all the Agents to always return loss/accs-lists, then the forward pass returns the avgs, then
    the eval returns the avs & cis. Then eval code here calls the eval_forward.
    Ok there are issues with the Agent code. The forward & eval_forward are the same. We need to fix this & clarify it.
"""
import logging
from argparse import Namespace
from typing import Any, Optional

from torch import Tensor, tensor

from uutils.torch_uu.agents.common import Agent


# -- get loss & acc

def eval_sl(args: Namespace,
            model: Agent,
            dataloaders,
            split: str = 'val',
            training: bool = False,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Evaluate the current model on the eval set (val set recommended).

    note:
        - note we have "get_eval" function cuz the api for data loaders for meta-learning is different from the SL one.
        Despite the Agent having the same api as the MetaLearner object.
        - Training=True for eval only for meta-learning, here we do want .eval(). This is because .train()
        uses batch stats while .eval() uses the running means.
        See: https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
        - Note, we don't need to loop through the data loader, we can get confidence intervals for the mean error
        from 1 batch - since we are estimating the mean loss from the eval set.
    """
    # todo: maybe some day, just get the losses, accs then take the mean & ci here.
    losses, accs = get_sl_eval_lists_accs_losses(args, model, dataloaders, split, training)
    from uutils.torch_uu.metrics.confidence_intervals import mean_confidence_interval
    loss, loss_ci = mean_confidence_interval(losses)
    acc, acc_ci = mean_confidence_interval(accs)
    return loss, loss_ci, acc, acc_ci


def meta_eval(args: Namespace,
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
    losses, accs = get_meta_eval_lists_accs_losses(args, model, dataloaders, split, training)
    from uutils.torch_uu.metrics.confidence_intervals import mean_confidence_interval
    loss, loss_ci = mean_confidence_interval(losses)
    acc, acc_ci = mean_confidence_interval(accs)
    return loss, loss_ci, acc, acc_ci


# -- get accs & losses

def get_meta_eval_lists_accs_losses(args: Namespace,
                                    model: Agent,
                                    dataloaders,
                                    split: str = 'val',
                                    training: bool = True,
                                    # True to avoid different tasks: https://stats.stackexchange.com/a/551153/28986
                                    ) -> tuple[list[float], list[float]]:
    """
    Get list of accuracies and losses for all task in a batch from the dataloader.

    note:
        - note we have "get_eval" function cuz the api for data loaders for meta-learning is different from the SL one.
        Despite the Agent having the same api as the MetaLearner object.
    """
    meta_losses, meta_accs = [], []
    if training == False:
        print(f'You sure {training=}? Recall you always want batch stats in BN layer in MetaL so put True.')
        logging.warning(f'You sure {training=}? Recall you always want batch stats in BN layer in MetaL so put True.')
    # - hack for l2l using other maml for 5CNN1024
    from uutils.torch_uu.dataloaders.meta_learning.l2l_to_torchmeta_dataloader import TorchMetaDLforL2L
    if isinstance(dataloaders[split], TorchMetaDLforL2L):
        # dl needs to be in "torchmeta format"
        batch: any = next(iter(dataloaders[split]))
        meta_losses, meta_accs = model.get_lists_accs_losses(batch, training)
        return meta_losses, meta_accs
    # - l2l
    if hasattr(args, 'tasksets'):
        # hack for l2l
        from learn2learn.data import TaskDataset
        split: str = 'validation' if split == 'val' else split
        task_dataset: TaskDataset = getattr(args.tasksets, split)
        meta_losses, meta_accs = model.get_lists_accs_losses(task_dataset, training)
        return meta_losses, meta_accs
    # - rfs meta-loader
    from uutils.torch_uu.dataset.rfs_mini_imagenet import MetaImageNet
    if isinstance(dataloaders['val'].dataset, MetaImageNet):
        eval_loader = dataloaders[split]
        if eval_loader is None:  # split is train, rfs code doesn't support that annoying :/
            raise NotImplementedError
        batch: tuple[Tensor, Tensor, Tensor, Tensor] = get_meta_batch_from_rfs_metaloader(eval_loader)
        meta_losses, meta_accs = model.get_lists_accs_losses(batch, training)
        return meta_losses, meta_accs
    # - else normal data loader (so torchmeta, or normal pytorch data loaders)
    if isinstance(dataloaders, dict):
        batch: Any = next(iter(dataloaders[split]))
        meta_losses, meta_accs = model.get_lists_accs_losses(batch, training)
    else:
        raise ValueError(f'Unexpected error, dataloaders is of type {dataloaders=} but expected '
                         f'dict or something else (perhaps train, val, test loader type objects).')
    # -- return
    return meta_losses, meta_accs


def get_sl_eval_lists_accs_losses(args: Namespace,
                                  model: Agent,
                                  dataloaders,
                                  split: str = 'val',
                                  training: bool = False,
                                  as_list_floats: bool = False,
                                  ) -> tuple[iter, iter]:
    if isinstance(dataloaders, dict):
        batch: Any = next(iter(dataloaders[split]))
        # - return losses/accs as a tensor ~ [B, D1, ...]
        losses, accs = model.get_lists_accs_losses(batch, training, as_list_floats=as_list_floats)
        # - assert to display what should be happening
        # batch_x, batch_y = process_batch_ddp(args, batch)
        # B: int = batch_x.size(0)
        # assert loss.size() == torch.Size([B])
        # assert acc.size() == torch.Size([B])
    else:
        # hack for l2l
        from learn2learn.data import TaskDataset
        split: str = 'validation' if split == 'val' else split
        task_dataset: TaskDataset = getattr(args.tasksets, split)
        # val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(task_dataset, training)
        raise NotImplementedError  # idk what sl for task2vec means, what data do I get? do I just sample it, then make sure it's a tensor and make sure SL agent does the right thing?
    return losses, accs


# -- some extra rfs code

def get_meta_batch_from_rfs_metaloader(loader) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    notes:
    - you don't need to call cuda here cuz the agents/meta-learners should be doing that on their own.
    """
    # return [B, n*k, C, H, W]
    spt_xs, spt_ys, qry_xs, qry_ys = [], [], [], []
    for idx, data in enumerate(loader):
        support_xs, support_ys, query_xs, query_ys = data
        # not needed, this is like a dataloader op, so the code outside is suppose to do it, in particular
        # the metalearner/agent should be doing it.
        # if torch.cuda.is_available():
        #     support_xs = support_xs.cuda()
        #     query_xs = query_xs.cuda()
        batch_size, _, channel, height, width = support_xs.size()
        # - the -1 infers the size of that dimension. Operationally [k, n, C,H,W] -> [n*k, C, H, W]
        support_xs = support_xs.view(-1, channel, height, width)
        query_xs = query_xs.view(-1, channel, height, width)
        # - flatten the target tensors
        # support_ys = support_ys.view(-1).numpy()
        # query_ys = query_ys.view(-1).numpy()
        support_ys = support_ys.view(-1)
        query_ys = query_ys.view(-1)
        # - collect to later stack
        spt_xs.append(support_xs)
        spt_ys.append(support_ys)
        qry_xs.append(query_xs)
        qry_ys.append(query_ys)
    # - stack in the first (new) dimension
    from torch import stack
    spt_xs, spt_ys, qry_xs, qry_ys = stack(spt_xs), stack(spt_ys), stack(qry_xs), stack(qry_ys)
    # spt_ys, qry_ys = spt_ys.squeeze(), qry_ys.squeeze()
    assert len(spt_xs.size()) == 5, f'Error, should be [B, n*k, C, H, W] but got {len(spt_xs.size())=}'
    assert len(qry_xs.size()) == 5, f'Error, should be [B, n*k, C, H, W] but got {len(qry_xs.size())=}'
    assert len(spt_ys.size()) == 2, f'Error, should be [B, n*k] but got {len(spt_ys.size())=}'
    assert len(qry_ys.size()) == 2, f'Error, should be [B, n*k] but got {len(qry_ys.size())=}'
    return spt_xs, spt_ys, qry_xs, qry_ys
