#dataloader for VGG-flower

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

#from uutils.torch_uu.dataloaders.flower import get_transform


def get_remaining_transforms_flower(dataset: MetaDataset, ways: int, samples: int) -> list[TaskTransform]:
    from uutils.torch_uu.dataloaders.meta_learning.mini_imagenet_mi_l2l import get_remaining_transforms_mi
    remaining_task_transforms = get_remaining_transforms_mi(dataset, ways=ways, samples=samples)
    return remaining_task_transforms


def get_flower_datasets(
        root='~/data/l2l_data/',
        data_augmentation='hdb5_vggair',
        device=None,
        **kwargs,
) -> tuple[MetaDataset, MetaDataset, MetaDataset]:
    """Tasksets for VGGFlower benchmarks."""
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
    elif data_augmentation == 'hdb4_micod' or data_augmentation == 'hdb5_vggair':
        size: int = 84
        scale: tuple[float, float] = (0.18, 1.0)
        padding: int = 8
        ratio: tuple[float, float] = (0.75, 1.3333333333333333)
        from uutils.torch_uu.dataset.delaunay_uu import delauny_random_resized_crop_random_crop
        train_data_transforms, _, test_data_transforms = delauny_random_resized_crop_random_crop(
            size=size,
            scale=scale,
            padding=padding,
            ratio=ratio,
        )
    else:
        raise ValueError(f'Invalid data_augmentation argument. Got: {data_augmentation=}')

    train_dataset = l2l.vision.datasets.VGGFlower102(root=root,
                                                transform=train_data_transforms,
                                                mode='train',
                                                download=True)
    valid_dataset = l2l.vision.datasets.VGGFlower102(root=root,
                                                transform=test_data_transforms,
                                                mode='validation',
                                                download=True)
    test_dataset = l2l.vision.datasets.VGGFlower102(root=root,
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
    train_dataset.name = 'train_flower'
    valid_dataset.name = 'val_flower'
    test_dataset.name = 'test_flower'

    _datasets = (train_dataset, valid_dataset, test_dataset)
    return _datasets


class Task_transform_flower:
    def __init__(self, ways, samples):
        self.ways = ways
        self.samples = samples

    def __call__(self, dataset):
        from uutils.torch_uu.dataloaders.meta_learning.mini_imagenet_mi_l2l import get_remaining_transforms_mi
        return get_remaining_transforms_mi(dataset, self.ways, self.samples)
