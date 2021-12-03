"""
Code for optimizers
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


def get_uutils_default_adafactor_from_torch_optimizer_default(mdl: nn.Module, lr: float = 1e-4) -> Optimizer:
    """
    Gets adafactor with uutils default parameters.

    e.g.:
    >>> import torch_optimizer as optim
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
    return optimizer


def get_adafactor_fairseq():
    """

    Refs:
        - https://github.com/pytorch/fairseq/blob/main/fairseq/optim/adafactor.py
        - https://github.com/huggingface/transformers/issues/14574
        - https://stackoverflow.com/questions/70171427/adafactor-from-transformers-hugging-face-only-works-with-transfromers-does-it
    :return:
    """
    pass


# -- tests

def one_test():
    pass


if __name__ == '__main__':
    print('Done, success!\a')
