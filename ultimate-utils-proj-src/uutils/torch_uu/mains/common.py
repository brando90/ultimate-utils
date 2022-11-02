"""

Note:
    - the move_to_dpp moves things .to(device)
"""
from argparse import Namespace
from typing import Optional, Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint
from uutils.torch_uu.distributed import move_model_to_dist_device_or_serial_device
from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
from uutils.torch_uu.optim_uu.adafactor_uu import get_default_adafactor_opt_fairseq_and_hps_dict, \
    get_default_adafactor_scheduler_fairseq_and_hps_dict

from pdb import set_trace as st


def get_and_create_model_opt_scheduler_for_run(args: Namespace) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    if resume_from_checkpoint(args):
        model, opt, scheduler = load_model_optimizer_scheduler_from_ckpt(args)
    else:
        model, opt, scheduler = get_and_create_model_opt_scheduler_first_time(args)
    return model, opt, scheduler


def get_and_create_model_opt_scheduler_first_time(args: Namespace) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    model_option: str = args.model_option
    model_hps = args.model_hps if hasattr(args, 'model_hps') else {}

    opt_option = args.opt_option
    opt_hps = args.opt_hps if hasattr(args, 'opt_hps') else {}

    scheduler_option = args.scheduler_option
    scheduler_hps = args.scheduler_hps if hasattr(args, 'scheduler_hps') else {}

    _get_and_create_model_opt_scheduler(args,
                                        model_option,
                                        model_hps,

                                        opt_option,
                                        opt_hps,

                                        scheduler_option,
                                        scheduler_hps,
                                        )
    return args.model, args.opt, args.scheduler


def load_model_optimizer_scheduler_from_ckpt(args: Namespace,
                                             path_to_checkpoint: Optional[str] = None,
                                             load_model: bool = True,
                                             load_opt: bool = True,
                                             load_scheduler: bool = True,

                                             ) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    """
    Load the most important things: model, optimizer, scheduler.

    Ref:
        - standard way: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

    Horrrible hack comment:
    - I accidentally did .cls = new_cls in SL code
    """
    # - prepare args from ckpt
    # ckpt: dict = torch.load(args.path_to_checkpoint, map_location=torch.device('cpu'))
    path_to_checkpoint = args.path_to_checkpoint if path_to_checkpoint is None else path_to_checkpoint
    ckpt: dict = torch.load(path_to_checkpoint, map_location=args.device)
    if load_model:
        model_option = ckpt['model_option']
        model_hps = ckpt['model_hps']
    else:
        model_option = 'None'
        model_hps = 'None'

    if load_opt:
        opt_option = ckpt['opt_option']
        opt_hps = ckpt['opt_hps']
    else:
        opt_option = "None"
        opt_hps = "None"

    if load_scheduler:
        scheduler_option = ckpt['scheduler_option']
        scheduler_hps = ckpt['scheduler_hps']
    else:
        scheduler_option = "None"
        scheduler_hps = "None"

    _get_and_create_model_opt_scheduler(args,
                                        model_option,
                                        model_hps,

                                        opt_option,
                                        opt_hps,

                                        scheduler_option,
                                        scheduler_hps,
                                        )

    # - load state dicts
    if load_model:
        model_state_dict: dict = ckpt['model_state_dict']
        args.model.load_state_dict(model_state_dict)
    else:
        assert args.model is None

    if load_opt:
        opt_state_dict: dict = ckpt['opt_state_dict']
        args.opt.load_state_dict(opt_state_dict)
    else:
        assert args.opt is None

    if load_scheduler:
        if hasattr(args.scheduler, 'load_state_dict'):
            scheduler_state_dict: dict = ckpt['scheduler_state_dict']
            args.scheduler.load_state_dict(scheduler_state_dict)
    else:
        assert args.scheduler is None

    # - load last step (it or epoch_num)
    args_from_ckpt = Namespace(**ckpt['args_dict'])
    if args_from_ckpt.training_mode == 'epochs':
        args.epoch_num = ckpt['epoch_num']
    else:
        args.it = ckpt['it']
    # assert next(mdl.parameters()).device == args.device
    return args.model, args.opt, args.scheduler


def _get_and_create_model_opt_scheduler(args: Namespace,

                                        model_option: Optional[str] = None,  # if None use args else "None" pass
                                        model_hps: dict = {},

                                        opt_option: Optional[str] = None,  # if None use args else "None" pass
                                        opt_hps: dict = {},

                                        scheduler_option: Optional[str] = None,  # if None use args else "None" pass
                                        scheduler_hps: dict = {},

                                        ) -> tuple[nn.Module, Optimizer, _LRScheduler]:
    """
    Creates for the first time the model, opt, scheduler needed for the experiment run in main.

    Note:
        - if you don't want to use something, pass "None" to the corresponding option e.g. _option arg.
        E.g. it is useful to pass "None" for the optimizer and scheduler to only get the model.
        - The default is that if it's None then return values from the args field.
    """
    # - get model the empty model from the hps for the cons for the model
    model_option: str = args.model_option if model_option is None else model_option
    if model_option == "None":
        # pass
        args.model, args.model_hps = None, None
    elif model_option == '5CNN_opt_as_model_for_few_shot_sl':
        args.model, args.model_hps = get_default_learner_and_hps_dict(**model_hps)
    elif model_option == 'resnet12_rfs_mi' or model_option == 'resnet12_rfs':  # resnet12_rfs for backward compat
        from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_mi
        args.model, args.model_hps = get_resnet_rfs_model_mi(args.model_option, **model_hps)
    elif model_option == 'resnet12_rfs_cifarfs_fc100':
        from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_cifarfs_fc100
        args.model, args.model_hps = get_resnet_rfs_model_cifarfs_fc100(args.model_option, **model_hps)
    elif model_option == '5CNN_l2l_mi':
        import learn2learn
        model_hps: dict = dict(n_classes=args.n_classes)  # ways or n_classes is the hp
        args.model, args.model_hps = learn2learn.vision.models.MiniImagenetCNN(args.n_classes), model_hps
        raise NotImplementedError
    elif model_option == '4CNN_l2l_cifarfs':
        from uutils.torch_uu.models.l2l_models import cnn4_cifarsfs
        args.model, args.model_hps = cnn4_cifarsfs(**model_hps)
    elif model_option == '3FNN_5_gaussian':  # Added by Patrick 3/14
        from uutils.torch_uu.models.fullyconnected import fnn3_gaussian
        args.model, args.model_hps = fnn3_gaussian(**model_hps)
    elif model_option == 'resnet18_random':
        from models import get_model
        from task2vec import ProbeNetwork
        args.model: ProbeNetwork = get_model('resnet18', pretrained=False, num_classes=args.n_cls)
        # args.model.cls = args.model.classifier
        args.model_hps = None  # for now since we can get the model using task2vec idk if we need this
        raise NotImplementedError
    elif model_option == 'resnet34_random':
        from models import get_model
        from task2vec import ProbeNetwork
        args.model: ProbeNetwork = get_model('resnet34', pretrained=False, num_classes=args.n_cls)
        # args.model.cls = args.model.classifier
        args.model_hps = None  # for now since we can get the model using task2vec idk if we need this
        raise NotImplementedError
    elif model_option == 'resnet12_hdb1_mio':
        # same as the one in MI since Omni has been resized to that size [3, 84, 84].
        from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_mi
        assert model_hps['num_classes'] != 64
        assert model_hps['num_classes'] != 1100
        assert model_hps['num_classes'] == 64 + 1100
        args.model, args.model_hps = get_resnet_rfs_model_mi(args.model_option, **model_hps)
    elif model_option == 'vit_mi':
        from uutils.torch_uu.models.hf_uu.vit_uu import get_vit_get_vit_model_and_model_hps_mi
        args.model, args.model_hps = get_vit_get_vit_model_and_model_hps_mi(**model_hps)
    else:
        raise ValueError(f'Model option given not found: got {model_option=}')
    if model_option is not None:
        args.model = move_model_to_dist_device_or_serial_device(args.rank, args, args.model)

    # - get optimizer
    opt_option: str = args.opt_option if opt_option is None else opt_option
    if opt_option == 'None':
        # pass
        args.opt, args.opt_hps = None, None
    elif opt_option == 'AdafactorDefaultFair':
        args.opt, args.opt_hps = get_default_adafactor_opt_fairseq_and_hps_dict(args.model, **opt_hps)
    elif opt_option == 'Adam_rfs_cifarfs':
        from uutils.torch_uu.optim_uu.adam_uu import get_opt_adam_rfs_cifarfs
        args.opt, args.opt_hps = get_opt_adam_rfs_cifarfs(args.model, **opt_hps)
    elif opt_option == 'Adam_default':
        from uutils.torch_uu.optim_uu.adam_uu import get_opt_adam_default
        args.opt, args.opt_hps = get_opt_adam_default(args.model, **opt_hps)
    elif opt_option == 'adam_mi_old_resnet12_rfs':
        # from uutils.torch_uu.optim_uu.adam_uu import get_opt_adam_rfs_cifarfs
        # args.opt, args.opt_hps = get_opt_adam_rfs_cifarfs(args.model, **opt_hps)
        return None, None
    elif opt_option == 'Sgd_rfs':
        from uutils.torch_uu.optim_uu.sgd_uu import get_opt_sgd_rfs
        args.opt, args.opt_hps = get_opt_sgd_rfs(args.model, **opt_hps)
    else:
        raise ValueError(f'Optimizer option is invalid: got {opt_option=}')

    # - get scheduler
    scheduler_option: str = args.scheduler_option if scheduler_option is None else scheduler_option
    if scheduler_option == 'None':  # None != 'None', obj None means use the ckpt value
        args.scheduler, args.scheduler_hps = None, None
    elif scheduler_option == 'AdafactorSchedule':
        args.scheduler, args.scheduler_hps = get_default_adafactor_scheduler_fairseq_and_hps_dict(args.opt,
                                                                                                  **scheduler_hps)
    elif scheduler_option == 'Adam_cosine_scheduler_rfs_cifarfs':
        from uutils.torch_uu.optim_uu.adam_uu import get_cosine_scheduler_adam_rfs_cifarfs
        args.scheduler, args.scheduler_hps = get_cosine_scheduler_adam_rfs_cifarfs(args.opt, **scheduler_hps)
    elif scheduler_option == 'Cosine_scheduler_sgd_rfs':
        from uutils.torch_uu.optim_uu.sgd_uu import get_cosine_scheduler_sgd_rfs
        args.scheduler, args.scheduler_hps = get_cosine_scheduler_sgd_rfs(args.opt, **scheduler_hps)
    else:
        raise ValueError(f'Scheduler option is invalid: got {scheduler_option=}')
    return args.model, args.opt, args.scheduler


def load_model_ckpt(args: Namespace,
                    path_to_checkpoint: Optional[str] = None,
                    ) -> nn.Module:
    base_model, _, _ = load_model_optimizer_scheduler_from_ckpt(args, path_to_checkpoint,
                                                                load_model=True,
                                                                load_opt=False,
                                                                load_scheduler=False)
    assert args.model is base_model
    return base_model


def meta_learning_type(args: Namespace) -> bool:
    """
    Since we didn't save the agent stuff excplicitly, can't remember if this is a bug or not...
    anyway, you can get the
    """
    return args.ckpt['args_dict']['agent']


def _get_agent(args: Namespace, agent_hps: dict = {}):
    """

    Note:
        - some of these functions might assume you've already loaded .model correct in args.
    """
    if args.agent_opt == 'MAMLMetaLearnerL2L_default':
        from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearnerL2L
        agent = MAMLMetaLearnerL2L(args, args.model, **agent_hps)
    elif args.agent_opt == 'MAMLMetaLearner_default':
        from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
        agent = MAMLMetaLearner(args, args.model, **agent_hps)
    else:
        raise ValueError(f'Invalid meta-learning type, got {meta_learning_type(args)}')
    args.agent = agent
    return args.agent
