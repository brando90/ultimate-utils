from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_opt_adam_rfs_cifarfs(mdl: nn.Module,
                             lr=1e-4,  # note original rfs did: 0.05
                             weight_decay=0.0005,
                             ) -> tuple[Optimizer, dict]:
    """

    Note: because my rfs code uses lr 1e-4 we are using a 1e-4 learning rate for now. Perhaps change later if
    experiments don't work out well.
    """
    opt_hps: dict = dict(
        lr=lr,
        weight_decay=weight_decay
    )
    optimizer: Optimizer = optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer, opt_hps


def get_cosine_scheduler_adam_rfs_cifarfs(optimizer: Optimizer,
                                     lr=1e-4,  # note original rfs did: 0.05
                                     lr_decay_rate=0.1,
                                     epochs=90
                                     ) -> tuple[_LRScheduler, dict]:
    scheduler_hps: dict = dict(
        lr=lr,
        lr_decay_rate=lr_decay_rate,
        epochs=epochs
    )
    eta_min = lr * (lr_decay_rate ** 3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min, -1)
    return scheduler, scheduler_hps
