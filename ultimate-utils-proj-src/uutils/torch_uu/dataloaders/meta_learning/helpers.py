from argparse import Namespace
from pathlib import Path

from torch import nn


def get_meta_learning_dataloader(args: Namespace) -> dict:
    # Get Meta-Sets for few shot learning
    if isinstance(args.path_to_data_set, Path):
        raise NotImplementedError
    else:
        if 'miniimagenet' in str(args.path_to_data_set):
            args.meta_learner.classification()
            args.training_mode = 'iterations'
            from uutils.torch_uu.dataloaders.meta_learning.torchmeta_ml_dataloaders import \
                get_miniimagenet_dataloaders_torchmeta
            args.dataloaders = get_miniimagenet_dataloaders_torchmeta(args)
        elif 'sinusoid' in str(args.path_to_data_set):
            args.training_mode = 'iterations'
            args.criterion = nn.MSELoss()
            args.meta_learner.regression()
            from uutils.torch_uu.dataloaders import get_torchmeta_sinusoid_dataloaders
            args.dataloaders = get_torchmeta_sinusoid_dataloaders(args)
        elif 'fully_connected' in str(args.data_path.name):
            args.training_mode = 'iterations'
            args.criterion = nn.MSELoss()
            args.meta_learner.regression()
            from uutils.torch_uu.dataloaders import get_torchmeta_rand_fnn_dataloaders
            args.dataloaders = get_torchmeta_rand_fnn_dataloaders(args)
        else:
            raise ValueError(f'Not such task: {args.data_path}')
