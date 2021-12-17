"""

Notes:
    - for augmentation discussions: https://stats.stackexchange.com/questions/320800/data-augmentation-on-training-set-only/320967#320967

"""
from argparse import Namespace


def get_sl_dataloader(args: Namespace) -> dict:
    args.data_set_path.exapenduser()
    data_set_path: str = str(args.data_set_path).lower()
    if 'mnist' in data_set_path:
        from uutils.torch_uu.dataloaders.mnist import get_train_valid_test_data_loader_helper_for_mnist
        dataloaders: dict = get_train_valid_test_data_loader_helper_for_mnist(args)
    elif 'cifar10' in data_set_path:
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid data set: got {data_set_path=}')
    args.dataloaders = dataloaders
    return dataloaders
