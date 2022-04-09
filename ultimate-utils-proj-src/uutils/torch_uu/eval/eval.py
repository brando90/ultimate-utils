import logging
from argparse import Namespace
from typing import Any, Optional

from torch import Tensor, tensor

from uutils.torch_uu.agents.common import Agent


def eval_sl(args: Namespace,
            model: Agent,
            dataloaders,
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
        # batch: Any = next(iter(dataloaders['test']))
        val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(batch, training)
    else:
        # hack for l2l
        from learn2learn.data import TaskDataset
        split: str = 'validation' if split == 'val' else split
        task_dataset: TaskDataset = getattr(args.tasksets, split)
        val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(task_dataset, training)
    return val_loss, val_loss_ci, val_acc, val_acc_ci


def meta_eval(args: Namespace,
              model: Agent,
              dataloaders,
              split: str = 'val',
              # training: bool = True,
              training: bool = False,
              ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """

    assumption: your agent has the .forward interface needed
    """
    if training == False:
        print(f'You sure {training=}? Recall you always want batch stats in BN layer in MetaL so put True.')
        logging.warning(f'You sure {training=}? Recall you always want batch stats in BN layer in MetaL so put True.')
    # - hack for l2l using other maml for 5CNN1024
    from uutils.torch_uu.dataloaders.meta_learning.l2l_to_torchmeta_dataloader import TorchMetaDLforL2L
    if isinstance(dataloaders[split], TorchMetaDLforL2L):
        # dl needs to be in "torchmeta format"
        batch: any = next(iter(dataloaders[split]))
        val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(batch, training)
        # val_loss, val_loss_ci, val_acc, val_acc_ci = model(batch, training)
        return val_loss, val_loss_ci, val_acc, val_acc_ci
    # assert False
    # - l2l
    if hasattr(args, 'tasksets'):
        # hack for l2l
        from learn2learn.data import TaskDataset
        split: str = 'validation' if split == 'val' else split
        task_dataset: TaskDataset = getattr(args.tasksets, split)
        val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(task_dataset, training)
        return val_loss, val_loss_ci, val_acc, val_acc_ci
    # - rfs meta-loader
    from uutils.torch_uu.dataset.rfs_mini_imagenet import MetaImageNet
    if isinstance(dataloaders['val'].dataset, MetaImageNet):
        eval_loader = dataloaders[split]
        if eval_loader is None:  # split is train, rfs code doesn't support that annoying :/
            return tensor(-1), tensor(-1), tensor(-1), tensor(-1)
        batch: tuple[Tensor, Tensor, Tensor, Tensor] = get_meta_batch_from_rfs_metaloader(eval_loader)
        val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(batch, training)
        return val_loss, val_loss_ci, val_acc, val_acc_ci
    # - else normal data loader (so torchmeta, or normal pytorch data loaders)
    if isinstance(dataloaders, dict):
        batch: Any = next(iter(dataloaders[split]))
        # print(batch['train'][0].size())
        # batch: Any = next(iter(dataloaders['test']))
        val_loss, val_loss_ci, val_acc, val_acc_ci = model.eval_forward(batch, training)
        return val_loss, val_loss_ci, val_acc, val_acc_ci
    else:
        raise ValueError(f'Unexpected error, dataloaders is of type {dataloaders=} but expected '
                         f'dict or something else (perhaps train, val, test loader type objects).')


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
