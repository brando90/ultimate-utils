from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_opt_sgd_rfs_cifarfs(mdl: nn.Module,
                            lr=1e-4,  # note original rfs did: 0.05
                            weight_decay=0.0005,
                            ) -> tuple[Optimizer, dict]:
    """

    We use SGD optimizer with a momentum of 0.9 and a weight decay of 5e−4. Each batch
    consists of 64 samples. The learning rate is initialized as
    0.05 and decayed with a factor of 0.1 by three times for
    all datasets, except for miniImageNet where we only decay
    twice as the third decay has no effect. We train 100 epochs
    for miniImageNet, 60 epochs for tieredImageNet, and 90
    epochs for both CIFAR-FS and FC100. During distillation,
    we use the same learning schedule and set α = β = 0.5.

    ref:
        - rfs paper: https://arxiv.org/abs/2003.11539
    """
    # args.outer_opt = optim.SGD(args.base_model.parameters(), lr=args.outer_lr, momentum=0.9, weight_decay=5e-4)

    # opt_hps: dict = dict(
    #     lr=lr,
    #     weight_decay=weight_decay
    # )
    # optimizer: Optimizer = optim.Adam(mdl.parameters(), lr=lr, weight_decay=0.0005)
    # return optimizer, opt_hps
    pass


def get_cosine_scheduler_sgd_rfs_cifarfs(optimizer: Optimizer,
                                         lr=1e-4,
                                         lr_decay_rate=0.1,
                                         epochs=90
                                         ) -> tuple[_LRScheduler, dict]:
    """
    The learning rate is initialized as
    0.05 and decayed with a factor of 0.1 by three times for
    all datasets, except for miniImageNet where we only decay
    twice as the third decay has no effect. We train 100 epochs
    for miniImageNet, 60 epochs for tieredImageNet, and 90
    epochs for both CIFAR-FS and FC100.

    ref:
        - rfs paper: https://arxiv.org/abs/2003.11539
    """
    # scheduler_hps: dict = dict(
    #     lr=1e-4,
    #     lr_decay_rate=0.1,
    #     epochs=90
    # )
    # eta_min = lr * (lr_decay_rate ** 3)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min, -1)
    # return scheduler, scheduler_hps
    pass
