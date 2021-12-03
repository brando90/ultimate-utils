"""
Code for optimizers
"""
from typing import Union, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_uutils_default_adafactor_from_torch_optimizer_and_scheduler_default(mdl: nn.Module, lr: float = 1e-4,
                                                                            scheduler_opt: Optional[str] = None
                                                                            ) -> tuple[Optimizer, _LRScheduler]:
    """
    Gets adafactor with uutils default parameters.

    e.g.:
    >>> iimport torch_optimizer as optim
    # model = ...
    >>> optimizer = optim.DiffGrad(model.parameters(), lr=0.001)
    >>> optimizer.step()
    refs:
        - https://github.com/huggingface/transformers/issues/14574
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
    if scheduler_opt is None or scheduler_opt == 'None':
        scheduler = None
    elif scheduler_opt == 'adafactor_scheduler_from_huggingface':
        from uutils.torch_uu.optim_uu.adafactor_uu import get_hugging_face_adafactor_scheduler
        # scheduler = AdafactorSchedule(optimizer)
        # scheduler = get_hugging_face_adafactor_scheduler(optimizer, initial_lr=0.0)
        scheduler = get_hugging_face_adafactor_scheduler(optimizer)
    else:
        raise ValueError(f'Invalid value, got: {scheduler_opt=}')
    return optimizer, scheduler


def get_default_uu_adafactor_optimizer_and_scheduler_fairseq() -> tuple[Optimizer, _LRScheduler]:
    """
    Get uutils default adafactor optimizer & scheduler from fairseq.

    Refs:
        - https://github.com/pytorch/fairseq/blob/main/fairseq/optim/adafactor.py
        - https://github.com/huggingface/transformers/issues/14574
        - https://stackoverflow.com/questions/70171427/adafactor-from-transformers-hugging-face-only-works-with-transfromers-does-it
    :return:
    """
    pass


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


# -- tests

def one_test():
    pass


if __name__ == '__main__':
    print('Done, success!\a')
