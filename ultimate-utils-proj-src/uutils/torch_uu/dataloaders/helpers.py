"""

Notes:
    - for augmentation discussions: https://stats.stackexchange.com/questions/320800/data-augmentation-on-training-set-only/320967#320967

"""
from argparse import Namespace
from pathlib import Path

from torch import nn

from pdb import set_trace as st


def replace_final_layer(args: Namespace, n_classes: int, BYPASS_PROTECTION: bool = False):
    """
    WARNING:
    USE AT YOUR OWN DEMISE. YOU WILL REGRET IT.
    THIS IS EVIL CODE. I REGRET USING THIS. THIS CREATEA A DYSIMMETRY IN SL & MAML MODEL LOADING AND MAKES LOADING A
    MODEL FROM A CKPT REALLY CONFUSING. MAKE SURE MAML AND USL CODE BOTH CALL THIS.
    """
    if BYPASS_PROTECTION:
        if hasattr(args.model, 'cls'):  # for resnet12 (but it will activate if the model has a cls layer)
            args.model.cls = nn.Linear(args.model.cls.in_features, n_classes).to(args.device)
        elif hasattr(args.model, 'model'):  # for 5CNN
            args.model.model.cls = nn.Linear(args.model.model.cls.in_features, n_classes).to(args.device)
        elif "CNN4" in str(args.model):  # for l2l CNNs
            # args.model.classifier = nn.Linear(args.model.classifier.in_features, n_classes).to(args.device)
            # args.model.cls = args.model.classifier
            raise NotImplementedError
            # this confuses pytorch's backprop, see https://github.com/pytorch/pytorch/issues/73697
            # a solution could be to dynamically add the decorator (which seems to not confuse pytorch)
        else:
            raise ValueError(
                f'Given model does not have a final cls layer. Check that your model is in the right format:'
                f'{type(args.model)=} do print(args.model) to see what your arch is and fix the error.')
    else:
        raise ValueError('ERROR. DONT USE THIS CODE.')


def get_sl_dataloader(args: Namespace) -> dict:
    if hasattr(args, 'data_augmentation'):
        print(f'----> {args.data_augmentation=}')
    if hasattr(args, 'data_path'):
        args.data_path.expanduser() if isinstance(args.data_path, Path) else args.data_path
        data_path: str = str(args.data_path)
    else:
        args.data_path = ''
    if args.data_option == 'n_way_gaussians_sl':  # next 3 lines added by Patrick 4/26/22. Everything else shouldn't have changed
        from uutils.torch_uu.dataloaders.meta_learning.gaussian_1d_tasksets import \
            get_train_valid_test_data_loader_1d_gaussian
        args.dataloaders: dict = get_train_valid_test_data_loader_1d_gaussian(args)
        print("Got n_way_gaussians_sl as dataset")
    elif args.data_option == 'n_way_gaussians_sl_nd':
        from uutils.torch_uu.dataloaders.meta_learning.gaussian_nd_tasksets import \
            get_train_valid_test_data_loader_nd_gaussian
        args.dataloaders: dict = get_train_valid_test_data_loader_nd_gaussian(args)
        print("Got n_way_gaussians_sl_nd as dataset")
    elif args.data_option == 'hdb1_mio_usl':
        from diversity_src.dataloaders.usl.hdb1_mi_omniglot_usl_dl import hdb1_mi_omniglot_usl_all_splits_dataloaders
        args.dataloaders: dict = hdb1_mi_omniglot_usl_all_splits_dataloaders(args)
        assert args.model.cls.out_features == 64 + 1100, f'hdb1 expects more classes but got {args.model.cls.out_features=},' \
                                                         f'\nfor model {type(args.model)=}'  # hdb1
    elif args.data_option == 'hdb4_micod_usl':
        from uutils.torch_uu.dataloaders.usl import hdb4_micod_usl_all_splits_dataloaders
        args.dataloaders: dict = hdb4_micod_usl_all_splits_dataloaders(args)
        assert args.model.cls.out_features == 64 + 34 + 64 + 1100, f'hdb4 expects more classes but got {args.model.cls.out_features=},' \
                                                                   f'\nfor model {type(args.model)=}'
    elif 'mnist' in data_path:
        from uutils.torch_uu.dataloaders.mnist import get_train_valid_test_data_loader_helper_for_mnist
        args.dataloaders: dict = get_train_valid_test_data_loader_helper_for_mnist(args)
        assert args.n_cls == 10
    elif 'cifar10' in data_path:
        raise NotImplementedError
    elif data_path == 'cifar100':
        from uutils.torch_uu.dataloaders.cifar100 import get_train_valid_test_data_loader_helper_for_cifar100
        args.dataloaders: dict = get_train_valid_test_data_loader_helper_for_cifar100(args)
        assert args.n_cls == 100
    elif 'CIFAR-FS' in data_path:
        from uutils.torch_uu.dataloaders.cifar100fs_fc100 import get_train_valid_test_data_loader_helper_for_cifarfs
        args.dataloaders: dict = get_train_valid_test_data_loader_helper_for_cifarfs(args)
    elif 'miniImageNet_rfs' in data_path:
        from uutils.torch_uu.dataloaders.miniimagenet_rfs import get_train_valid_test_data_loader_miniimagenet_rfs
        args.dataloaders: dict = get_train_valid_test_data_loader_miniimagenet_rfs(args)
    elif args.data_option == 'mds':
        # todo, would be nice to move this code to uutils @patrick so import is from uutils
        from diversity_src.dataloaders.metadataset_batch_loader import get_mds_loader
        args.dataloaders: dict = get_mds_loader(args)
        # assert args.model.cls.out_features == 3144
        # assert args.model.cls.out_features == 712 + 70 + 140 + 33 + 994 + 883 + 241 + 71
    elif 'l2l' in data_path:
        if args.data_option == 'cifarfs_l2l_sl':
            from uutils.torch_uu.dataloaders.cifar100fs_fc100 import get_sl_l2l_cifarfs_dataloaders
            args.dataloaders: dict = get_sl_l2l_cifarfs_dataloaders(args)
        else:
            raise ValueError(f'Invalid data set: got {data_path=} or wrong data option: {args.data_option}')
    else:
        raise ValueError(f'Invalid data set: got {data_path=}')
    return args.dataloaders
