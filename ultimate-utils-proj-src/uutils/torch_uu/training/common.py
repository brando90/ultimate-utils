"""

# - scheduling rate tips
- Do "But the most commonly used method is when the validation loss does not improve for a few epochs." according to https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
- for epochs it seems every epoch step is common: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

ref:
    - https://forums.pytorchlightning.ai/t/what-is-the-recommended-or-most-common-practices-to-using-a-scheduler/1416

"""
import logging
from argparse import Namespace
from typing import Any

import progressbar
from progressbar import ProgressBar
from torch import nn, Tensor, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


# - get data

def get_data(dataloaders,
             split: str = 'val',
             ) -> Any:
    """
    Get data according to the different data loader/taskset api's we've encountered. Cases "documented" by the if
    statements in the code.

    return: either a normal batch (tensor) or a l2l taskdataset.
    """
    if isinstance(dataloaders, dict):
        # - torchmeta data loader for l2l
        from uutils.torch_uu.dataloaders.meta_learning.l2l_to_torchmeta_dataloader import TorchMetaDLforL2L
        if isinstance(dataloaders[split], TorchMetaDLforL2L):
            # dl needs to be in "torchmeta format"
            batch: any = next(iter(dataloaders[split]))
            return batch
        # - rfs meta-loader
        from uutils.torch_uu.dataset.rfs_mini_imagenet import MetaImageNet
        # if isinstance(dataloaders['val'].dataset, MetaImageNet):
        if isinstance(dataloaders[split].dataset, MetaImageNet):
            eval_loader = dataloaders[split]
            if eval_loader is None:  # split is train, rfs code doesn't support that annoying :/
                raise NotImplementedError
            batch: tuple[Tensor, Tensor, Tensor, Tensor] = get_meta_batch_from_rfs_metaloader(eval_loader)
            return batch
        # - else normal data loader (so torchmeta, or normal pytorch data loaders)
        if isinstance(dataloaders, dict):
            batch: Any = next(iter(dataloaders[split]))
            return batch
        # - if all attempts failed raise error that dataloader is weird
        raise ValueError(f'Unexpected error, dataloaders is of type {dataloaders=} but expected '
                         f'dict or something else (perhaps train, val, test loader type objects).')
    else:
        # it is here at the end so that we are not forced to import l2l unless we really are using it, checking earlier forces ppl to install l2l when they might not need it
        from learn2learn.data import TaskDataset
        from learn2learn.vision.benchmarks import BenchmarkTasksets
        if isinstance(dataloaders, BenchmarkTasksets):
            split: str = 'validation' if split == 'val' else split
            task_dataset: TaskDataset = getattr(dataloaders, split)
            return task_dataset
        else:
            raise ValueError(f'Unexpected error, dataloaders is {dataloaders=}.')


# - progress bars

def get_trainer_progress_bar(args: Namespace) -> ProgressBar:
    import uutils
    if args.training_mode == 'fit_single_batch' or args.training_mode == 'meta_train_agent_fit_single_batch':
        bar: ProgressBar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    elif args.training_mode == 'iterations':
        bar: ProgressBar = uutils.get_good_progressbar(max_value=args.num_its)
    elif args.training_mode == 'epochs':
        bar: ProgressBar = uutils.get_good_progressbar(max_value=args.num_epochs)
    elif args.training_mode == 'iterations_train_convergence':
        bar: ProgressBar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    elif args.training_mode == 'epochs_train_convergence':
        bar: ProgressBar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    else:
        raise ValueError(f'Invalid training_mode value, got: {args.training_mode}')
    return bar


def check_halt(args: Namespace) -> bool:
    """
    Check if we should halt depending on training mode.

    For logic for idx:
    If idx + 1 >= n then we are done. idx is zero index, so after the end of the 0th loop's actual logic (e.g. train_step)
    we have done 1 step although the index is zero. So we need to only check for idx + 1 >= n.
    """
    if args.training_mode == 'fit_single_batch' or args.training_mode == 'meta_train_agent_fit_single_batch':
        # halt: bool = train_acc >= acc_tolerance and train_loss <= train_loss_tolerance
        halt: bool = args.convg_meter.check_converged()
    elif args.training_mode == 'iterations':
        #  idx + 1 == n, this means you've done n loops where idx = it or epoch_num
        halt: bool = args.it + 1 >= args.num_its
    elif args.training_mode == 'epochs':
        #  idx + 1 == n, this means you've done n loops where idx = it or epoch_num
        halt: bool = args.epoch_num + 1 >= args.num_epochs
    elif args.training_mode == 'iterations_train_convergence':
        halt: bool = args.convg_meter.check_converged()
    elif args.training_mode == 'epochs_train_convergence':
        halt: bool = args.convg_meter.check_converged()
    # - my hunch is that convergence is better
    elif args.training_mode == 'iterations_train_target_train_acc':
        raise NotImplementedError
    elif args.training_mode == 'epochs_train_target_train_acc':
        raise NotImplementedError
    elif args.training_mode == 'iterations_train_target_train_loss':
        raise NotImplementedError
    elif args.training_mode == 'epochs_train_target_train_loss':
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid training_mode value, got: {args.training_mode}')
    return halt


def gradient_clip(args, meta_opt):
    """
    Do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖
    depending on args it does it per parameter or all parameters together.

    Note:
        - don't do this if you're using Adafactor.
    """
    # do gradient clipping: If ‖g‖ ≥ c Then g := c * g/‖g‖
    if hasattr(args, 'grad_clip_mode'):
        if args.grad_clip_mode is None:
            pass
        elif args.grad_clip_mode == 'no_grad_clip' or args.grad_clip_mode is None:  # for backwards compatibility
            pass
        elif args.grad_clip_mode == 'clip_all_seperately':
            for group_idx, group in enumerate(meta_opt.param_groups):
                for p_idx, p in enumerate(group['params']):
                    nn.utils.clip_grad_norm_(p, args.grad_clip_rate)
        elif args.grad_clip_mode == 'clip_all_together':
            # [y for x in list_of_lists for y in x]
            all_params = [p for group in meta_opt.param_groups for p in group['params']]
            nn.utils.clip_grad_norm_(all_params, args.grad_clip_rate)
        else:
            raise ValueError(f'Invalid, args.grad_clip_mode = {args.grad_clip_mode}')


def optimizer_step(args: Namespace, optimizer, meta_batch_size: int = None):
    raise NotImplementedError
    import cherry
    if isinstance(optimizer, Optimizer):
        optimizer.step()
    elif isinstance(optimizer, cherry.optim.Distributed):
        # Average the accumulated gradients and optimize
        for p in args.agent.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        optimizer.step()  # averages gradients across all workers
    else:
        raise ValueError(f'Optimizer {optimizer=} is not supported when trying to do its optimizer step.')


def scheduler_step(args: Namespace, scheduler: _LRScheduler):
    """

    Note:
        - use the if i % log_freq outside. So it assumes you already decided how often to call this. With this setup it
        means that when you do s.step(val_loss) the step will be with respect to a step/collection of its or epochs.
    """
    assert args.scheduler is scheduler
    if not hasattr(args, 'scheduler'):
        return
    else:
        # args.scheduler.step() if (args.scheduler is not None) else None
        if args.scheduler is None:
            return
        # -- ReduceLROnPlateu
        elif isinstance(args.scheduler, ReduceLROnPlateau):
            val_batch: Any = next(iter(args.dataloaders['val']))
            val_loss, val_loss_ci, val_acc, val_acc_ci = args.mdl.eval_forward(val_batch)
            args.scheduler.step(val_loss)
            raise NotImplementedError
        # -- CosineAnnealingLR
        # based: https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/train_supervised.py#L243
        # rfs seems to call cosine scheduler every epoch, since cosine takes in max step, I assume it must be fine to
        # call it every step, it probably decays with a cosine according to the step num they are on (epoch or it).
        elif isinstance(args.scheduler, optim.lr_scheduler.CosineAnnealingLR):
            args.scheduler.step()
        # -- Error catcher
        else:
            import transformers.optimization
            # from transformers.optimization import AdafactorSchedule
            # -- AdafactorSchedule transformers/hugging face
            if isinstance(args.scheduler, transformers.optimization.AdafactorSchedule):
                args.scheduler.step()
            else:
                raise ValueError(f'Error, invalid Scheduler: the scheduler {args.scheduler=} is not supported.')


# - meter

class ConvergenceMeter:
    """
    Class that keeps track if you have converged or not. If you track val_loss this will be equivalent to
    Early Stopping.
    """

    def __init__(self, name: str, convergence_patience: int = 5, current_lowest: float = float('inf')):
        """

        :param name:
        :param patience: number of checks with no improvement after which training will be stopped. Note that the
        intention is that the training code decides how often to call this, ideally in the if i % log_freq conditional.
        """
        self.name = name
        self.current_lowest = current_lowest
        self.counts_above_current_lowest = 0
        self.convergence_patience = convergence_patience
        self.reset(current_lowest)

    def reset(self, new_lowest: float):
        self.current_lowest = new_lowest
        self.counts_above_current_lowest = 0

    def update(self, val):
        """
        Attempt to update the current lowest. If it is lower set this new lowest to be the lowest.
        """
        # - if you have a loss that decreased the lowest you've seen re-start counting.
        if val < self.current_lowest:
            self.reset(new_lowest=val)
        # - if you were not bellow the lowest, then you've been above the lowest for +1 counts
        else:
            self.counts_above_current_lowest += 1

    def item(self) -> float:
        if type(self.current_lowest) is Tensor:
            return self.current_lowest.item()
        else:
            return float(self.current_lowest)

    def check_converged(self) -> bool:
        """
        Check if you have converged. If the loss/value you are tracking has been above the current lowest you have seen
        enough times, then you have converged.
        """
        # - halt if you've been above the current lowest enough times.
        return self.convergence_patience <= self.counts_above_current_lowest

    def __str__(self):
        """
        ref:
            - .__dict__ or vars( ) ? https://stackoverflow.com/questions/21297203/use-dict-or-vars
        :return:
        """
        return f'ConvergenceMeter({vars(self)}'


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


# - tutorials tests

# - run main

if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    # train_test_()
    print(f"\nSuccess Done!: {report_times(start)}\a\n")
