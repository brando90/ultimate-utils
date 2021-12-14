"""
Note: you are recommended to follow this
https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
and likely avoid dill due to possible path to objects changes.
"""
from argparse import Namespace

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler

import uutils
from uutils.torch_uu.checkpointing_uu import try_to_get_scheduler_state_dict
from uutils.torch_uu.distributed import is_lead_worker, get_model_from_ddp


def save_for_supervised_learning(args: Namespace, ckpt_filename: str = 'ckpt.pt'):
    """
    Save a model checkpoint now (if we are the lead process).

    Implementation:
    - since you need to create an instance of the model before loading it, you need to save the hyperparameters
    that go into the constructor of the object.

    Warning:
        - if you save with dill but save the actual objects, your this dill unpickling will likely
         needs to know the paths to the code to load the class correctly. So it's safer to load the data e.g.
         weights, biases, optimizer state and then load the data into the objects when YOU instantiate those objects.

    Ref:
        - standard way: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        - https://stackoverflow.com/questions/70129895/why-is-it-not-recommended-to-save-the-optimizer-model-etc-as-pickable-dillable
    """
    if is_lead_worker(args.rank):
        # import dill
        import pickle
        args.logger.save_current_plots_and_stats()

        # - ckpt
        args_pickable: Namespace = uutils.make_args_pickable(args)
        torch.save({'training_mode': args.training_mode,
                    'it': args.it,
                    'epoch_num': args.epoch_num,

                    'args': args_pickable,  # some versions of this might not have args!

                    'model_state_dict': args.model.state_dict(),  # added later, to make it easier to check what optimizer was used
                    'model_str': str(args.model),  # added later, to make it easier to check what optimizer was used
                    'model_hps_for_cons': args.model_hps_for_cons,  # to create an empty new instance when loading model
                    'model_option': args.model_option,

                    'opt_state_dict': args.opt.state_dict(),
                    'opt_str': str(args.opt),

                    'scheduler_str': str(args.scheduler),
                    'scheduler_state_dict': try_to_get_scheduler_state_dict(args.scheduler)
                    },
                   # pickle_module=dill,
                   pickle_module=pickle,
                   f=args.log_root / ckpt_filename)

def load_model_optimizer_scheduler_from_checkpoint_given_model_type(args: Namespace,
                                                                    model_option: Optional[str] =  None,
                                                                    ) \
        -> tuple[nn.Module, optim.Optimizer, _LRScheduler]:
    """
    Load the standard most important things: model, optimizer, scheduler.

    Ref:
        - standard way: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    """
    ckpt: dict = torch.load(path2ckpt / filename, map_location=torch.device('cpu'))
    # - get model
    if model_option == '5CNN_opt_as_model_for_few_shot':
        model: nn.Module = load_model_5CNN_opt_as_model_for_few_shot(args, ckpt)
    elif model_option == 'resnet12-rfs':
        pass
    else:
        raise ValueError(f'Model type given not found: got {model_option=}')

    # - get optimizer

    # - get scheduler

    # - return
    return model, opt, scheduler


def load_model_5CNN_opt_as_model_for_few_shot(args: Namespace) -> nn.Module:
    pass
    # - get the hps of the model & build the instance
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import Learner
    model: nn.Module = Learner(image_size, bn_eps,
                 bn_momentum,
                 n_classes,
                 filter_size,  # Meta-LSTM & MAML use 32 filters
                 levels = None,
                 spp=False)
    # - load the parameters into the empty instance

    # - return the loaded instance
    return model

def load_model_resnet12_rfs(args: Namespace) -> nn.Module:
    pass
    # - get the hps of the model & build the instance

    # - load the parameters into the empty instance

    # - return the loaded instance
    # return model