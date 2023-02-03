from typing import Optional, Union

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR


class _AdafactorSchedulerUU(LambdaLR):
    """
    Copy Pasted from AdafactorSchedule.
    TODO: I think this should work for the torch_optimizer library...
        - perhaps its better to return the hugging face one? i.e. scheduler = AdafactorSchedule(optimizer)
        - https://github.com/jettify/pytorch-optimizer/issues/404
    """

    def __init__(self, optimizer, initial_lr=0.0):
        assert False, 'untested'

        def lr_lambda(_):
            return initial_lr

        for group in optimizer.param_groups:
            group["initial_lr"] = initial_lr

        super().__init__(optimizer, lr_lambda)
        for group in optimizer.param_groups:
            del group["initial_lr"]

    def get_lr(self):
        opt = self.optimizer
        lrs = [
            opt._get_lr(group, opt.state[group["params"][0]])
            for group in opt.param_groups
            if group["params"][0].grad is not None
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs


def get_hugging_face_adafactor_scheduler(optimizer, initial_lr: float = 0.0) -> _LRScheduler:
    from transformers.optimization import AdafactorSchedule
    # scheduler = AdafactorSchedule(optimizer)
    scheduler: _LRScheduler = AdafactorSchedule(optimizer, initial_lr=initial_lr)
    return scheduler


def get_adafactor_scheduler(optimizer: Optimizer, scheduler_opt: Optional[str] = None) -> _LRScheduler:
    """
    refs:
        - https://fairseq.readthedocs.io/en/latest/lr_scheduler.html
    """
    if scheduler_opt is None or scheduler_opt == 'None':
        scheduler = None
    elif scheduler_opt == 'adafactor_scheduler_from_huggingface':
        from uutils.torch_uu.optim_uu.adafactor_uu import get_hugging_face_adafactor_scheduler
        scheduler: _LRScheduler = get_hugging_face_adafactor_scheduler(optimizer)
    elif scheduler_opt == 'fairseq_scheduler':
        # https://fairseq.readthedocs.io/en/latest/lr_scheduler.html
        raise ValueError(f'Invalid value, got: {scheduler_opt=}')
    else:
        raise ValueError(f'Invalid value, got: {scheduler_opt=}')
    return scheduler


def get_uutils_default_adafactor_from_torch_optimizer_and_scheduler_default(mdl: nn.Module,
                                                                            lr: float = 1e-4,
                                                                            scheduler_opt: Optional[str] = None
                                                                            ) -> tuple[Optimizer, _LRScheduler]:
    """
    Gets adafactor with uutils default parameters.

    e.g.:
    >>> import torch_optimizer as optim
    # model = ...
    >>> optimizer = optim.DiffGrad(model.parameters(), lr=0.001)
    >>> optimizer.step()
    refs:
        - https://github.com/jettify/pytorch-optimizer/issues/405
        - https://github.com/huggingface/transformers/issues/14574
        - https://stackoverflow.com/questions/70218565/how-to-have-adafactor-run-a-custom-rfs-resnet12-with-maml-with-the-torch-opt?noredirect=1&lq=1
        - https://stackoverflow.com/questions/70171427/adafactor-from-transformers-hugging-face-only-works-with-transfromers-does-it
    """
    import torch_optimizer as optim

    optimizer: Optimizer = optim.Adafactor(
        mdl.parameters(),
        lr=lr,
        eps2=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    )
    scheduler: _LRScheduler = get_adafactor_scheduler(optimizer, scheduler_opt)
    return optimizer, scheduler


def get_default_uu_adafactor_optimizer_and_scheduler_fairseq(mdl: nn.Module,
                                                             lr: float = 1e-4,
                                                             scheduler_opt: Optional[str] = None
                                                             ) -> tuple[Optimizer, _LRScheduler]:
    """
    Get uutils default adafactor optimizer & scheduler from fairseq.

    Refs:
        - https://github.com/pytorch/fairseq/blob/main/fairseq/optim/adafactor.py
        - https://fairseq.readthedocs.io/en/latest/optim.html#fairseq.optim.adafactor.FairseqAdafactor
    """
    from fairseq import optim

    optimizer = optim.adafactor.Adafactor(
        params=mdl.parameters(),
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    )
    scheduler: _LRScheduler = get_adafactor_scheduler(optimizer, scheduler_opt)
    return optimizer, scheduler


def get_uutils_default_adafactor_and_scheduler_from_huggingface(model: nn.Module,
                                                                lr: Union[None, float] = None
                                                                ) -> tuple[Optimizer, _LRScheduler]:
    """
    Get's Adafactor's default optimizer and scheduler from huggingface.

    ( params https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor
    lr = None
    eps = (1e-30, 0.001)
    clip_threshold = 1.0
    decay_rate = -0.8
    beta1 = None
    weight_decay = 0.0
    scale_parameter = True
    relative_step = True
    warmup_init = False )

    refs:
        - https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor
        - https://stackoverflow.com/questions/70171427/adafactor-from-transformers-hugging-face-only-works-with-transfromers-does-it
    """
    from transformers import Adafactor
    from transformers.optimization import AdafactorSchedule

    optimizer: Optimizer = Adafactor(model.parameters(),
                                     scale_parameter=True,
                                     relative_step=True,
                                     warmup_init=True,
                                     lr=lr)
    # - for details: https://github.com/huggingface/transformers/blob/ec47baeba20379aa148bbc80bbfc31851ceabcf7/src/transformers/optimization.py#L604
    scheduler = AdafactorSchedule(optimizer)
    return optimizer, scheduler


def get_default_adafactor_hps() -> tuple[dict, dict]:
    opt_hps: dict = dict(
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    )
    scheduler_hps: dict = dict(
        initial_lr=0.0
    )
    return opt_hps, scheduler_hps


def get_default_adafactor_opt_fairseq_and_hps_dict(mdl: nn.Module,
                                                   lr=None,
                                                   eps=(1e-30, 1e-3),
                                                   clip_threshold=1.0,
                                                   decay_rate=-0.8,
                                                   beta1=None,
                                                   weight_decay=0.0,
                                                   scale_parameter=True,
                                                   relative_step=True,
                                                   warmup_init=False,
                                                   ) -> tuple[Optimizer, dict]:
    """
    Get the optimizer and scheduler objects for the current model and the hyperparameters (hps) that created it.

    note:
        - set when "AdafactorDefaultFair" is on.
    """
    # - hps
    opt_hps: dict = dict(
        lr=lr,
        eps=eps,
        clip_threshold=clip_threshold,
        decay_rate=decay_rate,
        beta1=beta1,
        weight_decay=weight_decay,
        scale_parameter=scale_parameter,
        relative_step=relative_step,
        warmup_init=warmup_init,
    )

    # - get opt & scheduler from hps
    from fairseq import optim
    optimizer: Optimizer = optim.adafactor.Adafactor(params=mdl.parameters(), **opt_hps)
    return optimizer, opt_hps


def get_default_adafactor_scheduler_fairseq_and_hps_dict(optimizer: Optimizer,
                                                         initial_lr=0.0,
                                                         ) -> tuple[_LRScheduler, dict]:
    """
    Get the optimizer and scheduler objects for the current model and the hyperparameters (hps) that created it.
    """
    # - hps
    scheduler_hps: dict = dict(
        initial_lr=initial_lr
    )
    # - get opt & scheduler from hps
    from transformers.optimization import AdafactorSchedule
    scheduler: _LRScheduler = AdafactorSchedule(optimizer, **scheduler_hps)
    return scheduler, scheduler_hps
