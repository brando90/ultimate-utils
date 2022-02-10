"""

Notes:
    - for augmentation discussions: https://stats.stackexchange.com/questions/320800/data-augmentation-on-training-set-only/320967#320967

"""
from argparse import Namespace

from torch import nn


def replace_final_layer(args: Namespace, n_classes: int):
    if hasattr(args.model, 'cls'):  # for resnet12
        args.model.cls = nn.Linear(args.model.cls.in_features, n_classes).to(args.device)
    elif hasattr(args.model, 'model'):  # for 5CNN
        args.model.model.cls = nn.Linear(args.model.model.cls.in_features, n_classes).to(args.device)
    elif "CNN4" in str(args.model):  # for l2l CNNs
        args.model.classifier = nn.Linear(args.model.classifier.in_features, n_classes).to(args.device)
        args.model.cls = args.model.classifier
    else:
        raise ValueError(f'Given model does not have a final cls layer. Check that your model is in the right format:'
                         f'{type(args.model)=} do print(args.model) to see what your arch is and fix the error.')


def get_sl_dataloader(args: Namespace) -> dict:
    args.data_path.expanduser()
    data_path: str = str(args.data_path)
    if 'mnist' in data_path:
        from uutils.torch_uu.dataloaders.mnist import get_train_valid_test_data_loader_helper_for_mnist
        args.dataloaders: dict = get_train_valid_test_data_loader_helper_for_mnist(args)
        replace_final_layer(args, n_classes=args.n_cls)
        assert args.n_cls == 10
    elif 'cifar10' in data_path:
        raise NotImplementedError
    elif data_path == 'cifar100':
        from uutils.torch_uu.dataloaders.cifar100 import get_train_valid_test_data_loader_helper_for_cifar100
        args.dataloaders: dict = get_train_valid_test_data_loader_helper_for_cifar100(args)
        replace_final_layer(args, n_classes=args.n_cls)
        assert args.n_cls == 100
    elif 'CIFAR-FS' in data_path:
        from uutils.torch_uu.dataloaders.cifar100fs_fc100 import get_train_valid_test_data_loader_helper_for_cifarfs
        args.dataloaders: dict = get_train_valid_test_data_loader_helper_for_cifarfs(args)
        replace_final_layer(args, n_classes=args.n_cls)
    elif 'miniImageNet_rfs' in data_path:
        from uutils.torch_uu.dataloaders.miniimagenet_rfs import get_train_valid_test_data_loader_miniimagenet_rfs
        args.dataloaders: dict = get_train_valid_test_data_loader_miniimagenet_rfs(args)
        replace_final_layer(args, n_classes=args.n_cls)
    elif 'l2l' in data_path:
        if args.data_option == 'cifarfs_l2l_sl':
            from uutils.torch_uu.dataloaders.cifar100fs_fc100 import get_sl_l2l_cifarfs_dataloaders
            args.dataloaders: dict = get_sl_l2l_cifarfs_dataloaders(args)
            replace_final_layer(args, n_classes=args.n_cls)
        else:
            raise ValueError(f'Invalid data set: got {data_path=} or wrong data option: {args.data_option}')
    else:
        raise ValueError(f'Invalid data set: got {data_path=}')
    return args.dataloaders
