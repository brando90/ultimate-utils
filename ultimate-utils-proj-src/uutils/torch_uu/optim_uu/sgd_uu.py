from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def _get_opt_hps_sgd_rfs() -> dict:
    opt_hps: dict = dict(lr=5e-2, momentum=0.9, weight_decay=5e-4)
    return opt_hps


def get_opt_sgd_rfs(mdl: nn.Module,
                    lr,  # rfs 5e-2
                    momentum,  # rfs 0.9
                    weight_decay,  # rfs 5e-4
                    ) -> tuple[Optimizer, dict]:
    """

    From rfs:
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
    optimizer: Optimizer = optim.SGD(mdl.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    opt_hps: dict = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer, opt_hps


def get_cosine_scheduler_sgd_rfs(optimizer: Optimizer,
                                 T_max,  # args.num_epochs // args.log_scheduler_freq, times to decay basically
                                 lr,  # rfs 5e-2
                                 lr_decay_rate,  # rfs 0.1
                                 ) -> tuple[_LRScheduler, dict]:
    """
    This scheduler is a smooth implementation to their "decay three times by 0.1" by using a cosine scheduler.

    From rfs:
    The learning rate is initialized as
    0.05 and decayed with a factor of 0.1 by three times for
    all datasets, except for miniImageNet where we only decay
    twice as the third decay has no effect. We train 100 epochs
    for miniImageNet, 60 epochs for tieredImageNet, and 90
    epochs for both CIFAR-FS and FC100.

    ref:
        - rfs paper: https://arxiv.org/abs/2003.11539
    """
    eta_min = lr * (lr_decay_rate ** 3)  # note, MAML++ uses 1e-5, if you calc it seems rfs uses 5e-5
    scheduler_hps: dict = dict(T_max=T_max, lr_decay_rate=lr_decay_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
    return scheduler, scheduler_hps


def _adjust_learning_rate_rfs(epoch, opt, optimizer):
    """
    Sets the learning rate to the initial LR decayed by decay rate every steep step.

    Note:
        - that they also have a smooth version of this using the cosine annealing rate. We have it too, use it to
        reproduce rfs.

    ref:
        - https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/util.py#L61
    """
    import numpy as np

    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
