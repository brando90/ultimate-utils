"""
Note: you are recommended to follow this
https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
and likely avoid dill due to possible path to objects changes.

Refs:
    - for DDP checkpointing see: https://stackoverflow.com/questions/70386800/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data
"""
from argparse import Namespace
from typing import Optional

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import uutils
from uutils.torch_uu.checkpointing_uu import try_to_get_scheduler_state_dict
from uutils.torch_uu.distributed import is_lead_worker, get_model_from_ddp, is_running_parallel, move_to_ddp


def save_for_supervised_learning(args: Namespace,
                                 ckpt_filename: str = 'ckpt.pt',
                                 ignore_logger: bool = False,
                                 ):
    """
    Save a model checkpoint now (if we are the lead process).

    Implementation details/comments:
    - since you need to create an instance of the model before loading it, you need to save the hyperparameters
    that go into the constructor of the object.
    - when you choose a specific model_option, opt_option, scheduler_option it comes with the very specific
    hyperparameters needed to construct an empty model. One saves that dict and the uses that to construct
    an empty model and then later load it's parameters.

    Warning:
        - if you save with dill but save the actual objects, your this dill unpickling will likely
         needs to know the paths to the code to load the class correctly. So it's safer to load the data e.g.
         weights, biases, optimizer state and then load the data into the objects when YOU instantiate those objects.

    Ref:
        - standard way: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        - https://stackoverflow.com/questions/70129895/why-is-it-not-recommended-to-save-the-optimizer-model-etc-as-pickable-dillable
        - DDP checkpointing: https://stackoverflow.com/questions/70386800/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data
    """
    if is_lead_worker(args.rank):
        print(f'saving checkpoint at: {(args.log_root/ckpt_filename)=}')
        print(f'{type(args.model)=}')
        print(f'{args.model_option=}')
        import pickle
        if not ignore_logger:
            args.logger.save_current_plots_and_stats() if hasattr(args, 'logger') else None

        # - ckpt
        args_pickable: Namespace = uutils.make_args_pickable(args)
        # note not saving any objects, to make sure checkpoint is loadable later with no problems
        torch.save({'training_mode': args.training_mode,
                    'it': args.it,
                    'epoch_num': args.epoch_num,

                    # 'args': args_pickable,  # some versions of this might not have args!
                    # decided only to save the dict version to avoid this ckpt not working, make it args when loading
                    'args_dict': vars(args_pickable),  # some versions of this might not have args!

                    'model_state_dict': get_model_from_ddp(args.model).state_dict(),
                    'model_str': str(args.model),  # added later, to make it easier to check what optimizer was used
                    'model_hps': args.model_hps,
                    'model_option': args.model_option,

                    'opt_state_dict': args.opt.state_dict(),
                    'opt_str': str(args.opt),
                    'opt_hps': args.opt_hps,
                    'opt_option': args.opt_option,

                    'scheduler_str': str(args.scheduler),
                    'scheduler_state_dict': try_to_get_scheduler_state_dict(args.scheduler),
                    'scheduler_hps': args.scheduler_hps,
                    'scheduler_option': args.scheduler_option,
                    },
                   pickle_module=pickle,
                   f=args.log_root / ckpt_filename)
        print(f'saving checkpoint success at {(args.log_root/ckpt_filename)=}')
    return

# todo - make a folder for it in models uutils and put this at the bottom
# def load_model_resnet12_rfs(args: Namespace) -> nn.Module:
#     pass
#     # - get the hps of the model & build the instance
#
#     # - load the parameters into the empty instance
#
#     # - return the loaded instance
#     # return model
