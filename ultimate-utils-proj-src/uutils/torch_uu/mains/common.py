"""

Note:
- the move_to_dpp moves things .to(device)
"""
import os
import re
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

    opt_option = args.opt_option if hasattr(args, 'opt_option') else 'AdafactorDefaultFair'
    opt_hps = args.opt_hps if hasattr(args, 'opt_hps') else {}

    scheduler_option = args.scheduler_option if hasattr(args, 'scheduler_option') else 'None'
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

    Note:
        - Note this code assume you've already set the device properly outside of here. If you are using ddp then code
        outside should've move model to device already (according to parallel using rank each proc or serial using gpu:0).

    Horrrible hack comment:
    - I accidentally did .cls = new_cls in SL code
    """
    # - prepare args from ckpt
    path_to_checkpoint = args.path_to_checkpoint if path_to_checkpoint is None else path_to_checkpoint
    # we could do a best effort set_device if args.device is None, e.g. call set_device(args), now no, responsibility is in main script runnning for now
    # ckpt: dict = torch.load(path_to_checkpoint, map_location=torch.device('cpu'))
    ckpt: dict = torch.load(path_to_checkpoint, map_location=args.device)
    # ckpt: dict = torch.load(path_to_checkpoint, map_location=torch.device('cuda:0'))
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
    elif model_option == '5CNN_opt_as_model_for_few_shot_sl' or model_option == '5CNN_opt_as_model_for_few_shot':
        args.model, args.model_hps = get_default_learner_and_hps_dict(**model_hps)
    elif model_option == 'resnet12_rfs_mi' or model_option == 'resnet12_rfs' or model_option == 'resnet24_rfs' or \
            re.match(r'resnet[0-9]+_rfs', model_option):  # resnet12_rfs for backward compat
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
        assert model_hps['num_classes'] == 64 + 1100
        args.model, args.model_hps = get_resnet_rfs_model_mi(args.model_option, **model_hps)
    elif model_option == 'vit_mi':
        from uutils.torch_uu.models.hf_uu.vit_uu import get_vit_get_vit_model_and_model_hps_mi
        args.model, args.model_hps = get_vit_get_vit_model_and_model_hps_mi(**model_hps)
    else:
        raise ValueError(f'Model option given not found: got {model_option=}')
    if model_option is not None:
        args.rank = args.rank if hasattr(args, 'rank') else -1
        args.model = move_model_to_dist_device_or_serial_device(args.rank, args, args.model)

    # - get optimizer
    opt_option: str = args.opt_option if opt_option is None else opt_option
    if opt_option == 'None':
        # pass
        args.opt, args.opt_hps = None, None
    elif opt_option == 'AdafactorDefaultFair':
        args.opt, args.opt_hps = get_default_adafactor_opt_fairseq_and_hps_dict(args.model, **opt_hps)
    # elif opt_option == 'AdafactorHuggingFace':
    #     args.opt, args.opt_hps = get_default_adafactor_opt_fairseq_and_hps_dict(args.model, **opt_hps)
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


def _get_maml_agent(args: Namespace, agent_hps: dict = {}):
    """
    Note:
        - some of these functions might assume you've already loaded .model correct in args.
    """
    if args.agent_opt == 'MAMLMetaLearnerL2L':
        from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearnerL2L
        agent = MAMLMetaLearnerL2L(args, args.model, **agent_hps)
    elif args.agent_opt == 'MAMLMetaLearner':
        from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
        agent = MAMLMetaLearner(args, args.model, **agent_hps)
    else:
        raise ValueError(f'Invalid meta-learning type, got {meta_learning_type(args)}')
    args.agent = agent
    args.meta_learner = agent
    return agent


# -- examples, tests, unit tests, tutorials, etc

def args_5cnn_mdl_size_estimates():
    from pathlib import Path
    # - model
    args: Namespace = Namespace()
    # assert args.filter_size != -1, f'Err: {args.filter_size=}'
    # print(f'--->{args.filter_size=}')
    args.n_cls = 64 + 1100  # mio
    # args.n_cls = 1262  # micod
    args.n_cls = 5  # 5-way
    args.filter_size = 1
    args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size, levels=None, spp=False, in_channels=3)
    # - opt
    args.opt_option = 'Adam_rfs_cifarfs'
    # args.batch_size = 256
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)
    # - scheduler
    args.scheduler_option = 'None'

    # # - data
    # args.data_option = 'hdb1_mio_usl'
    # args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_option = 'hdb4_micod'
    args.n_classes = args.n_cls
    args.data_augmentation = 'hdb4_micod'
    return args


def print_model_size():
    """
    When do we match the resnet12rfs 1.4M params? how many filters do we need for the 5CNN?
    """
    from uutils.torch_uu import count_number_of_parameters
    # - get data loader
    # -- print num filters vs num params
    # num_filters: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # num_filters: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    num_filters: list[int] = [1, 2, 4, 6, 8, 12, 14, 16, 32, 64, 128, 256, 512, 1024]
    num_params: list[int] = []
    d: dict = {}
    for num_filter in num_filters:
        print(f'-- {num_filter=}')
        # - get args
        args: Namespace = args_5cnn_mdl_size_estimates()
        args.model_hps['filter_size'] = num_filter
        # - get fake data
        B = 4
        x = torch.rand(B, 3, 84, 84)
        # - get model
        get_and_create_model_opt_scheduler_for_run(args)
        args.number_of_trainable_parameters = count_number_of_parameters(args.model)
        print(f'{args.number_of_trainable_parameters=}')
        args.model(x)
        # - print number of parameters
        print(f'{args.model.cls.out_features=}')
        print(f'{args.filter_size=}') if hasattr(args, 'filter_size') else None
        # - append
        num_params.append(args.number_of_trainable_parameters)
        d[num_filter] = args.number_of_trainable_parameters
    print(f'{num_filters=}')
    print(f'{num_params=}')
    print(f'{d=}')
    # - make a table from the dict using pandas, key is num filters, value is num params and name them
    import pandas as pd
    df = pd.DataFrame({'Num Filters': list(d.keys()), 'Num Params': list(d.values())})
    print(df)
    # - plot number of filters vs number of params title nums params vs num filters x labl num filters y label num params, using uutils
    import matplotlib.pyplot as plt
    from uutils.plot import plot
    plot(num_filters, num_params, title='Number of Parameters vs Number of Filters', xlabel='Number of Filters',
         ylabel='Number of Parameters', marker='o', color='b')
    plt.axhline(y=1.4e6, color='r', linestyle='-', label='ResNet12RFS (num params)')
    plt.legend()
    plt.show()


# -- run main

if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    # main()
    print_model_size()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
