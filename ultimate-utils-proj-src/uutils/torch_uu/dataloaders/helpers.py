"""

Notes:
    - for augmentation discussions: https://stats.stackexchange.com/questions/320800/data-augmentation-on-training-set-only/320967#320967

"""
from argparse import Namespace

from torch import nn


def replace_final_layer(args: Namespace, n_classes: int):
    if hasattr(args.model, 'cls'):
        args.model.cls = nn.Linear(args.model.cls.in_features, n_classes)


def get_sl_dataloader(args: Namespace) -> dict:
    args.path_to_data_set.expanduser()
    path_to_data_set: str = str(args.path_to_data_set).lower()
    if 'mnist' in path_to_data_set:
        from uutils.torch_uu.dataloaders.mnist import get_train_valid_test_data_loader_helper_for_mnist
        args.dataloaders: dict = get_train_valid_test_data_loader_helper_for_mnist(args)
        replace_final_layer(args, n_classes=10)
    elif 'cifar10' in path_to_data_set:
        raise NotImplementedError
    elif path_to_data_set == 'cifar100':
        from uutils.torch_uu.dataloaders.cifar100 import get_train_valid_test_data_loader_helper_for_cifar100
        args.dataloaders: dict = get_train_valid_test_data_loader_helper_for_cifar100(args)
        replace_final_layer(args, n_classes=100)
    elif 'CIFAR-FS' in path_to_data_set:
        from uutils.torch_uu.dataloaders.cifar100 import get_rfs_union_sl_dataloader_cifar100fs
        args.dataloaders: dict = get_rfs_union_sl_dataloader_cifar100fs(args)
        replace_final_layer(args, n_classes=args.n_cls)
    else:
        raise ValueError(f'Invalid data set: got {path_to_data_set=}')
    return args.dataloaders
