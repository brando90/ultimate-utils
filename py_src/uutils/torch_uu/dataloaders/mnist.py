"""
Inspired from:
    - https://stackoverflow.com/questions/70400439/what-is-the-proper-way-to-create-training-validation-and-test-set-in-pytorch-or/70400448#70400448
"""
from argparse import Namespace
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from uutils.torch_uu.dataloaders.common import split_inidices, \
    get_serial_or_distributed_dataloaders

# normalization for mnist
NORMALIZE = transforms.Normalize((0.1307,), (0.3081,))  # MNIST


def get_train_valid_test_data_loader_helper_for_mnist(args: Namespace) -> dict:
    train_kwargs = {'data_path': args.data_path,
                    'batch_size': args.batch_size,
                    'batch_size_eval': args.batch_size_eval,
                    'augment_train': args.augment_train,
                    'augment_val': args.augment_val,
                    'num_workers': args.num_workers,
                    'pin_memory': args.pin_memory,
                    'rank': args.rank,
                    'world_size': args.world_size,
                    'merge': None
                    }
    test_kwargs = {'data_path': args.data_path,
                   'batch_size_eval': args.batch_size_eval,
                   'augment_test': args.augment_train,
                   'num_workers': args.num_workers,
                   'pin_memory': args.pin_memory,
                   'rank': args.rank,
                   'world_size': args.world_size,
                   'merge': None
                   }
    train_loader, val_loader = get_train_valid_loader(**train_kwargs)
    test_loader: DataLoader = get_test_loader(**test_kwargs)
    args.n_cls = 10  # all splits same same labels, so it's fine to put here
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders


def get_transform(augment: bool):
    if augment:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NORMALIZE
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            NORMALIZE
        ])
    return transform


def get_train_valid_loader(data_path: Path,
                           batch_size: int = 128,
                           batch_size_eval: int = 64,
                           seed: Optional[int] = None,
                           augment_train: bool = True,
                           augment_val: bool = False,
                           val_size: Optional[float] = 0.2,
                           shuffle: bool = False,  # false for reproducibility, and any split is as good as any other.
                           num_workers: int = -1,
                           pin_memory: bool = False,

                           rank: int = -1,
                           world_size: int = 1,
                           merge: Optional[Callable] = None,
                           ) -> tuple[DataLoader, DataLoader]:
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    """
    # train_kwargs = {'batch_size': args.batch_size}

    # define transforms
    train_transform = get_transform(augment_train)
    val_transform = get_transform(augment_val)

    # load the dataset
    data_path: str = str(Path(data_path).expanduser())
    train_dataset = datasets.MNIST(root=data_path, train=True,
                                   download=True, transform=train_transform)
    val_dataset = datasets.MNIST(root=data_path, train=True,
                                 download=True, transform=val_transform)
    indices = list(range(len(train_dataset)))
    train_indices, val_indices = split_inidices(indices, test_size=val_size, random_state=seed, shuffle=shuffle)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    train_loader, val_loader = get_serial_or_distributed_dataloaders(train_dataset,
                                                                     val_dataset,
                                                                     batch_size,
                                                                     batch_size_eval,
                                                                     rank,
                                                                     world_size,
                                                                     merge,
                                                                     num_workers,
                                                                     pin_memory
                                                                     )
    return train_loader, val_loader


def get_test_loader(data_path,
                    batch_size_eval: int = 64,
                    shuffle: bool = True,
                    augment_test: bool = False,
                    num_workers: int = -1,
                    pin_memory=False,

                    rank: int = -1,
                    world_size: int = 1,
                    merge: Optional[Callable] = None,
                    ) -> DataLoader:
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_path: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.

    Note:
        - it knows it's the test set since train=False in the body when creating the data set.
    """
    # define transform
    test_transform = get_transform(augment_test)

    # load the dataset
    data_path: str = str(Path(data_path).expanduser())
    test_dataset = datasets.MNIST(root=data_path,
                                  train=False,  # ensures its test set
                                  download=True,
                                  transform=test_transform)
    _, test_loader = get_serial_or_distributed_dataloaders(test_dataset,
                                                            test_dataset,
                                                            batch_size_eval,
                                                            batch_size_eval,
                                                            rank,
                                                            world_size,
                                                            merge,
                                                            num_workers,
                                                            pin_memory,
                                                            )
    return test_loader
