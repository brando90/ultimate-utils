from argparse import Namespace
from pathlib import Path

from torch import nn


def get_meta_learning_dataloader(args: Namespace) -> dict:
    # Get Meta-Sets for few shot learning
    # if isinstance(args.data_path, Path):
    if args.data_option == 'Path':
        # if path is given use the path
        if 'fully_connected' in str(args.data_path.name):
            from uutils.torch_uu.dataloaders import get_torchmeta_rand_fnn_dataloaders
            args.dataloaders = get_torchmeta_rand_fnn_dataloaders(args)
            raise NotImplementedError
        else:
            raise ValueError(f'Not such task: {args.data_path}')
    else:
        if args.data_option == 'torchmeta_miniimagenet':
            from uutils.torch_uu.dataloaders.meta_learning.torchmeta_ml_dataloaders import \
                get_miniimagenet_dataloaders_torchmeta
            args.dataloaders = get_miniimagenet_dataloaders_torchmeta(args)
            # raise NotImplementedError
        elif args.data_option == 'torchmeta_cifarfs':
            from uutils.torch_uu.dataloaders.meta_learning.torchmeta_ml_dataloaders import \
                get_cifarfs_dataloaders_torchmeta
            args.dataloaders = get_cifarfs_dataloaders_torchmeta(args)
        elif args.data_option == 'torchmeta_sinusoid':
            from uutils.torch_uu.dataloaders import get_torchmeta_sinusoid_dataloaders
            args.dataloaders = get_torchmeta_sinusoid_dataloaders(args)
            raise NotImplementedError
        else:
            raise ValueError(f'Not such task: {args.data_path}')
        return args.dataloaders
