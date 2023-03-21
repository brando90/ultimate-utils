"""
Mosty likely Torchmeta.
"""
from argparse import Namespace
from pathlib import Path

from torch import nn


def get_meta_learning_dataloaders(args: Namespace) -> dict:  # TorchMeta
    """

    Most likely Torchmeta
    """
    args.data_option = None if not hasattr(args, 'data_option') else args.data_option
    # Get Meta-Sets for few shot learning
    # if isinstance(args.data_path, Path):
    if args.data_option == 'Path':
        # if path is given use the path
        if 'fully_connected' in str(args.data_path.name):
            from uutils.torch_uu.dataloaders import get_torchmeta_rand_fnn_dataloaders
            dataloaders = get_torchmeta_rand_fnn_dataloaders(args)
            raise NotImplementedError
        else:
            raise ValueError(f'Not such task: {args.data_path}')
    else:
        if args.data_option == 'torchmeta_miniimagenet':
            from uutils.torch_uu.dataloaders.meta_learning.torchmeta_ml_dataloaders import \
                get_miniimagenet_dataloaders_torchmeta
            dataloaders = get_miniimagenet_dataloaders_torchmeta(args)
            # raise NotImplementedError
        elif args.data_option == 'torchmeta_cifarfs':
            from uutils.torch_uu.dataloaders.meta_learning.torchmeta_ml_dataloaders import \
                get_cifarfs_dataloaders_torchmeta
            dataloaders = get_cifarfs_dataloaders_torchmeta(args)
        elif args.data_option == 'torchmeta_sinusoid':
            from uutils.torch_uu.dataloaders import get_torchmeta_sinusoid_dataloaders
            dataloaders = get_torchmeta_sinusoid_dataloaders(args)
            raise NotImplementedError
        elif args.data_option == 'rfs_meta_learning_miniimagenet':
            from uutils.torch_uu.dataloaders.meta_learning.rfs_meta_learning_data_loaders import \
                get_rfs_meta_learning_mi_dataloader
            dataloaders = get_rfs_meta_learning_mi_dataloader(args)
        elif args.data_option == 'mds':
            # todo, would be nice to move this code to uutils @patrick so import is from uutils
            from diversity_src.dataloaders.metadataset_episodic_loader import get_mds_loaders
            dataloaders: dict = get_mds_loaders(args)
            [setattr(dataloaders[split], 'episodic_batch_2_task_dataset', True) for split in dataloaders.keys()]
        elif 'l2l_data' in str(args.data_path):
            # -- Converts a l2l data set as torchmeta data loader. Returns a batch of tasks, the way that torchmeta would.
            # note: this line is mainly intended for data ananlysis! not meant for ddp, see this if you want that but idk if it will work: https://github.com/learnables/learn2learn/issues/263
            from uutils.torch_uu.dataloaders.meta_learning.l2l_to_torchmeta_dataloader import \
                get_l2l_torchmeta_dataloaders
            dataloaders: dict = get_l2l_torchmeta_dataloaders(args)  # respects normal pytorch dl api afaik
        else:
            raise ValueError(f'Not such task: {args.data_path}')
        return dataloaders
