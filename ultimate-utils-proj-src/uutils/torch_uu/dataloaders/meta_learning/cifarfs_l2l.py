import os
import random

import learn2learn as l2l
import numpy as np
import torch
import torchvision
from learn2learn.data import MetaDataset
from learn2learn.data.transforms import TaskTransform
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import nn
from torchvision.transforms import Compose

from diversity_src.dataloaders.common import IndexableDataSet, ToRGB, DifferentTaskTransformIndexableForEachDataset

from uutils.torch_uu.dataloaders.cifar100fs_fc100 import get_transform


def get_remaining_transforms_cifarfs(dataset: MetaDataset, ways: int, samples: int) -> list[TaskTransform]:
    from uutils.torch_uu.dataloaders.meta_learning.mini_imagenet_mi_l2l import get_remaining_transforms_mi
    remaining_task_transforms = get_remaining_transforms_mi(dataset, ways=ways, samples=samples)
    return remaining_task_transforms


def get_cifarfs_datasets(
        root='~/data/l2l_data/',
        data_augmentation='hdb2',
        device=None,
        **kwargs,
) -> tuple[MetaDataset, MetaDataset, MetaDataset]:
    """Tasksets for CIFAR-FS benchmarks."""
    if data_augmentation is None:
        train_data_transforms = torchvision.transforms.ToTensor()
        test_data_transforms = torchvision.transforms.ToTensor()
    elif data_augmentation == 'normalize':
        train_data_transforms = Compose([
            lambda x: x / 255.0,
        ])
        test_data_transforms = train_data_transforms
    elif data_augmentation == 'rfs2020' or data_augmentation == 'hdb2':
        train_data_transforms = get_transform(augment=True)
        test_data_transforms = get_transform(augment=False)
    else:
        raise ValueError(f'Invalid data_augmentation argument. Got: {data_augmentation=}')

    train_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                                transform=train_data_transforms,
                                                mode='train',
                                                download=True)
    valid_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                                transform=test_data_transforms,
                                                mode='validation',
                                                download=True)
    test_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                               transform=test_data_transforms,
                                               mode='test',
                                               download=True)
    if device is not None:
        train_dataset = l2l.data.OnDeviceDataset(
            dataset=train_dataset,
            device=device,
        )
        valid_dataset = l2l.data.OnDeviceDataset(
            dataset=valid_dataset,
            device=device,
        )
        test_dataset = l2l.data.OnDeviceDataset(
            dataset=test_dataset,
            device=device,
        )
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    # - add names to be able to get the right task transform for the indexable dataset
    train_dataset.name = 'train_cifarfs'
    valid_dataset.name = 'val_cifarfs'
    test_dataset.name = 'test_cifarfs'

    _datasets = (train_dataset, valid_dataset, test_dataset)
    return _datasets


class Task_transform_cifarfs:
    def __init__(self, ways, samples):
        self.ways = ways
        self.samples = samples

    def __call__(self, dataset):
        from uutils.torch_uu.dataloaders.meta_learning.mini_imagenet_mi_l2l import get_remaining_transforms_mi
        return get_remaining_transforms_mi(dataset, self.ways, self.samples)
