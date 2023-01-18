from pathlib import Path

import os
import random
from typing import Callable, Union

import learn2learn as l2l
import numpy as np
import torch
from learn2learn.data import MetaDataset, FilteredMetaDataset
from learn2learn.data.transforms import TaskTransform
from learn2learn.vision.benchmarks import BenchmarkTasksets
from torch import nn
from torchvision.transforms import Compose, Normalize, ToPILImage, RandomCrop, ColorJitter, RandomHorizontalFlip, \
    ToTensor, RandomResizedCrop, Resize

from diversity_src.dataloaders.common import IndexableDataSet, ToRGB, DifferentTaskTransformIndexableForEachDataset

from torchvision import transforms
from PIL.Image import LANCZOS

import os
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.omniglot import Omniglot

from uutils import expanduser, report_times, download_and_extract


# can't remeber why I copy pasted but I think its cuz l2l has some weird pickling issue that breaks usl since pytorcm dataloaders do some pickling for some reason
class FullOmniglotUU(Dataset):
    """

    [[Source]]()

    **Description**

    This class provides an interface to the Omniglot dataset.

    The Omniglot dataset was introduced by Lake et al., 2015.
    Omniglot consists of 1623 character classes from 50 different alphabets, each containing 20 samples.
    While the original dataset is separated in background and evaluation sets,
    this class concatenates both sets and leaves to the user the choice of classes splitting
    as was done in Ravi and Larochelle, 2017.
    The background and evaluation splits are available in the `torchvision` package.

    **References**

    1. Lake et al. 2015. “Human-Level Concept Learning through Probabilistic Program Induction.” Science.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**
    ~~~python
    omniglot = l2l.vision.datasets.FullOmniglot(root='./data',
                                                transform=transforms.Compose([
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=True)
    omniglot = l2l.data.MetaDataset(omniglot)
    ~~~

    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # Set up both the background and eval dataset
        omni_background = Omniglot(self.root, background=True, download=download)
        self.len_omni_background_characters = len(omni_background._characters)
        # Eval labels also start from 0.
        # It's important to add 964 to label values in eval so they don't overwrite background dataset.
        omni_evaluation = Omniglot(self.root,
                                   background=False,
                                   download=download,
                                   target_transform=self._target_transform)

        self.dataset = ConcatDataset((omni_background, omni_evaluation))
        self._bookkeeping_path = os.path.join(self.root, 'omniglot-bookkeeping.pkl')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, character_class = self.dataset[item]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _target_transform(self, x):
        return x + self.len_omni_background_characters


def one_minus_x(x):
    """
    This transform only affects omniglot. Switches to a white black background.
    """
    # print('one minus x')
    return 1.0 - x


def get_omniglot_datasets(
        root: str = '~/data/l2l_data/',
        data_augmentation: str = 'hdb1',
        device=None,
        **kwargs,
) -> tuple[MetaDataset, MetaDataset, MetaDataset]:
    """
    Get omniglot data set with the provided re-size function -- since when combining different data sets they have to
    be re-sized to have the same size.
    For example, `hdb1` uses the re-size size of MI.


    Comment on data transform:
        - yes it seems the data transform is this simple (in fact l2l reshapes to 28 by 28). No jitter etc done, see: https://github.com/learnables/learn2learn/blob/0b9d3a3d540646307ca5debf8ad9c79ffe975e1c/learn2learn/vision/benchmarks/omniglot_benchmark.py#L10
        they don't have an if statement based on data gumentation so the code bellow is a correct representation of what l2l does.
    """
    if data_augmentation == 'l2l_original_data_transform':
        data_transforms = transforms.Compose([
            transforms.Resize(28, interpolation=LANCZOS),
            transforms.ToTensor(),
            one_minus_x,
        ])
    elif data_augmentation == 'hdb1':
        data_transforms = transforms.Compose([
            ToRGB(),
            transforms.Resize(84),
            transforms.ToTensor(),
            one_minus_x
        ])
    elif data_augmentation == 'hdb2':
        data_transforms = transforms.Compose([
            ToRGB(),
            transforms.Resize(32),
            transforms.ToTensor(),
            one_minus_x,
            # note: task2vec doesn't have this for mnist, wonder why...just flip background from black to white
        ])
    elif data_augmentation == 'use_random_resized_crop':
        train_data_transforms = transforms.Compose([
            ToRGB(),
            RandomResizedCrop((84, 84), scale=(0.18, 1.0), padding=8),
            transforms.ToTensor(),
            one_minus_x
        ])
        val_data_transforms = transforms.Compose([
            ToRGB(),
            transforms.Resize(84),
            transforms.ToTensor(),
            one_minus_x
        ])
        test_data_transforms = val_data_transforms
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid data transform option for omniglot, got instead {data_augmentation=}')

    # dataset = l2l.vision.datasets.FullOmniglot(
    #     root=root,
    #     transform=data_transforms,
    #     download=True,
    # )
    dataset: Dataset = FullOmniglotUU(root=root,
                                      transform=data_transforms,
                                      download=True,
                                      )
    if device is not None:
        dataset = l2l.data.OnDeviceDataset(dataset, device=device)  # bug in l2l
    # dataset: MetaDataset = l2l.data.MetaDataset(omniglot)

    classes = list(range(1623))
    # random.shuffle(classes)  # todo: wish I wouldn't have copied l2l here and removed this...idk if shuffling this does anything interesting. Doubt it.
    train_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[:1100])
    validation_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1100:1200])
    test_dataset: FilteredMetaDataset = l2l.data.FilteredMetaDataset(dataset, labels=classes[1200:])
    assert isinstance(train_dataset, MetaDataset)
    assert isinstance(validation_dataset, MetaDataset)
    assert isinstance(test_dataset, MetaDataset)
    assert len(train_dataset.labels) == 1100
    assert len(validation_dataset.labels) == 100
    # print(f'{len(classes[1200:])=}')
    assert len(test_dataset.labels) == 423, f'Error, got: {len(test_dataset.labels)=}'

    # - add names to be able to get the right task transform for the indexable dataset
    train_dataset.name = 'train_omniglot'
    validation_dataset.name = 'val_omniglot'
    test_dataset.name = 'test_omniglot'

    _datasets = (train_dataset, validation_dataset, test_dataset)
    return _datasets


def get_remaining_transforms_omniglot(dataset: MetaDataset, ways: int, shots: int) -> list[TaskTransform]:
    """

    Q: todo, what does RandomClassRotation do? https://github.com/learnables/learn2learn/issues/372
    """
    import learn2learn as l2l
    remaining_task_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset, ways, shots),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    return remaining_task_transforms


class Task_transform_omniglot(Callable):
    def __init__(self, ways, samples):
        self.ways = ways
        self.samples = samples

    def __call__(self, dataset):
        return get_remaining_transforms_omniglot(dataset, self.ways, self.samples)
