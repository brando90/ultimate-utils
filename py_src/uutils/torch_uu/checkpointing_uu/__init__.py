from argparse import Namespace
from typing import Union

from torch.optim.lr_scheduler import _LRScheduler


def try_to_get_scheduler_state_dict(scheduler: _LRScheduler) -> Union[dict, None]:
    try:
        scheduler_state_dict: dict = scheduler.state_dict()
    except Exception as e:
        scheduler_state_dict: None = None
    return scheduler_state_dict


def resume_from_checkpoint(args: Namespace) -> bool:
    """
    Returns true if to resume from checkpoint.
    """
    # - if args does not have path_to_checkpoint, then we don't need to check if we want to start from a checkpoint
    if not hasattr(args, 'path_to_checkpoint'):
        # for backwards compatibility
        return False
    else:
        # - check if some path to the ckpt is set, if yes then make it as an expanded Path.
        resume: bool = args.path_to_checkpoint is not None
    return resume
