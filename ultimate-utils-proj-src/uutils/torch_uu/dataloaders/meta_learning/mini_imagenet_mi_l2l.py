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


def mi_img_int_to_img_float(x) -> float:
    return x / 255.0


def get_mi_datasets(
        root='~/data/l2l_data/',
        data_augmentation='lee2019',
        device=None,
        **kwargs,
) -> tuple[MetaDataset, MetaDataset, MetaDataset]:
    """
    Returns MI according to l2l -- note it seems we are favoring the resizing of MI since when unioning datasets they
    have to have the same size.
    """
    # - get download l2l mi data set with my code & data url
    # if should_we_redownload_mi_data_set(root):
    #     url: str = 'https://zenodo.org/record/7311663/files/brandoslearn2learnminiimagenet.zip'
    #     print(f'Redownloading MI data from {url=} since this returned True: {should_we_redownload_mi_data_set(root)=} ')
    #     download_and_extract(url=url,
    #                          path_used_for_zip=root,
    #                          path_used_for_dataset=root,
    #                          rm_zip_file_after_extraction=False,
    #                          force_rewrite_data_from_url_to_file=True,
    #                          clean_old_zip_file=True,
    #                          )
    # -
    if data_augmentation is None:
        train_data_transforms = None
        test_data_transforms = None
    elif data_augmentation == 'normalize':
        train_data_transforms = Compose([
            # lambda x: x / 255.0,
            mi_img_int_to_img_float,
        ])
        test_data_transforms = train_data_transforms
    elif data_augmentation == 'lee2019' or data_augmentation == 'hdb1':
        # print(f'{data_augmentation=}')
        normalize = Normalize(
            mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
            std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0],
        )
        train_data_transforms = Compose([
            ToPILImage(),
            RandomCrop(84, padding=8),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
        # test_data_transforms = Compose([
        #     normalize,
        # ])
        test_data_transforms = Compose([
            ToPILImage(),
            # Resize((84, 84)),
            ToTensor(),
            normalize,
        ])
    elif data_augmentation == 'original_rfs':
        # ref: https://github.com/WangYueFt/rfs/blob/master/dataset/mini_imagenet.py#L22
        from PIL.Image import Image
        normalize = Normalize(
            mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
            std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0],
        )
        train_data_transforms = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])
        test_data_transforms = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise ('Invalid data_augmentation argument.')

    train_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='train',
        download=True,
    )
    valid_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='validation',
        download=True,
    )
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root=root,
        mode='test',
        download=True,
    )
    assert isinstance(train_dataset, Dataset)
    if device is None:
        train_dataset.transform = train_data_transforms
        valid_dataset.transform = test_data_transforms
        test_dataset.transform = test_data_transforms
    else:
        train_dataset = l2l.data.OnDeviceDataset(
            dataset=train_dataset,
            transform=train_data_transforms,
            device=device,
        )
        valid_dataset = l2l.data.OnDeviceDataset(
            dataset=valid_dataset,
            transform=test_data_transforms,
            device=device,
        )
        test_dataset = l2l.data.OnDeviceDataset(
            dataset=test_dataset,
            transform=test_data_transforms,
            device=device,
        )
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)
    assert isinstance(train_dataset, Dataset)

    # - add names to be able to get the right task transform for the indexable dataset
    train_dataset.name = 'train_mi'
    valid_dataset.name = 'val_mi'
    test_dataset.name = 'test_mi'

    _datasets = (train_dataset, valid_dataset, test_dataset)
    return _datasets


def get_remaining_transforms_mi(dataset: MetaDataset, ways: int, samples: int) -> list[TaskTransform]:
    import learn2learn as l2l
    remaining_task_transforms = [
        l2l.data.transforms.NWays(dataset, ways),
        l2l.data.transforms.KShots(dataset, samples),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ]
    return remaining_task_transforms


class Task_transform_mi(Callable):
    def __init__(self, ways, samples):
        self.ways = ways
        self.samples = samples

    def __call__(self, dataset):
        return get_remaining_transforms_mi(dataset, self.ways, self.samples)


# --

def download_mini_imagenet_brandos_download_from_zenodo():
    """
    zeneodo link of data set: https://zenodo.org/record/7311663#.Y21EE-zMJUc

python ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/dataloaders/hdb1_mi_omniglot_l2l.py
    """
    root = '~/data/l2l_data/'
    url: str = 'https://zenodo.org/record/7311663/files/brandoslearn2learnminiimagenet.zip'
    download_and_extract(url=url,
                         path_used_for_zip=root,
                         path_used_for_dataset=root,
                         rm_zip_file_after_extraction=False,
                         force_rewrite_data_from_url_to_file=True,
                         clean_old_zip_file=True,
                         )
    # download_and_extract('https://zenodo.org/record/7311663/files/brandoslearn2learnminiimagenet.zip?download=1',
    #                      '~/data/tmp', '~/data/tmp')
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/tmp/l2l_data', mode='train', download=False)
    [data for data in train_dataset]
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/tmp/l2l_data', mode='validation', download=False)
    [data for data in train_dataset]
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/tmp/l2l_data', mode='test', download=False)
    [data for data in train_dataset]
    for data in train_dataset:
        # print(f'{data=}')
        print(f'{data[0].size()=}')
        print(f'{data[1]=}')
    print('success loop through local data')


def should_we_redownload_mi_data_set(root: Union[str, Path]) -> bool:
    """
    If any of the pickle files is missing or loading the pickle data returned an error,
    return True i.e. yes you need to redownload the data a file is missing or there is a corrupted file.

    mini-imagenet-bookkeeping-{split}.pkl
    mini-imagenet-cache-{split}.pkl
    """
    root: Path = expanduser(root)
    splits: list[str] = ['train', 'validation', 'test']
    # filenames: list[str] = [f'mini-imagenet-bookkeeping-{split}.pkl', f'mini-imagenet-cache-{split}.pkl']
    # for filename in filenames:
    for split in splits:
        # -
        filename1: str = f'mini-imagenet-bookkeeping-{split}.pkl'
        path2file: Path = root / filename1
        if not path2file.exists():
            print(f'This file does NOT exist :{path2file=}, so we are redownloading MI')
            return True
        if not succeeded_opening_pkl_data_mi(path2file):
            return True
        # -
        filename2: str = f'mini-imagenet-cache-{split}.pkl'
        path2file: Path = root / filename2
        if not path2file.exists():
            print(f'This file does NOT exist :{path2file=}, so we are redownloading MI')
            return True
        if not succeeded_opening_pkl_data_mi(path2file):
            return True
    return False


def succeeded_opening_pkl_data_mi(path2file: Union[str, Path]) -> bool:
    path2file: Path = expanduser(path2file)
    try:
        data = torch.load(path2file)
        assert data is not None, f'Err: {data=}'
    except Exception as e:
        import logging
        print(f'Was not able to open the l2l data with torch.load, got error: {e=}')
        logging.warning(e)
        return False
    return True


def loop_through_mi_local():
    # checking the files I will upload to zenodo are good
    import learn2learn as l2l
    # train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/l2l_data', mode='train', download=True)
    # train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/l2l_data', mode='validation', download=True)
    # train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/l2l_data', mode='test', download=True)
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/l2l_data', mode='train', download=False)
    [data for data in train_dataset]
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/l2l_data', mode='validation', download=False)
    [data for data in train_dataset]
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/l2l_data', mode='test', download=False)
    [data for data in train_dataset]
    for data in train_dataset:
        # print(f'{data=}')
        print(f'{data[0].size()=}')
        print(f'{data[1]=}')
    print('success loop through local data')


def download_mini_imagenet_fix_use_gdrive():
    from uutils import download_and_extract
    download_and_extract(None,
                         '~/data/tmp', '~/data/tmp',
                         True,
                         '1wpmY-hmiJUUlRBkO9ZDCXAcIpHEFdOhD', 'mini-imagenet-cache-test.pkl'
                         )
    # download_and_extract('https://www.dropbox.com/s/9g8c6w345s2ek03/mini-imagenet-cache-train.pkl?dl=1',
    #                      '~/data/l2l_data', '~/data/l2l_data')
    # download_and_extract('https://www.dropbox.com/s/ip1b7se3gij3r1b/mini-imagenet-cache-validation.pkl?dl=1',
    #                      '~/data/l2l_data', '~/data/l2l_data')


# --

def mi_test():
    """
python -u ~/ultimate-utils/ultimate-utils-proj-src/uutils/torch_uu/dataloaders/meta_learning/l2l_mini_imagenet_mi.py
    """
    # from argparse import Namespace
    # from pathlib import Path
    #
    # args = Namespace(k_shots=5, k_eval=15, n_classes=5)
    # args.data_option = 'mini-imagenet'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    # args.data_path = Path('~/data/l2l_data/').expanduser()
    # args.data_augmentation = 'lee2019'
    #
    # args.tasksets: BenchmarkTasksets = get_tasksets(
    #     args.data_option,
    #     train_samples=args.k_shots + args.k_eval,
    #     train_ways=args.n_classes,
    #     test_samples=args.k_shots + args.k_eval,
    #     test_ways=args.n_classes,
    #     root=args.data_path,
    #     data_augmentation=args.data_augmentation,
    # )
    # print(args.tasksets)
    pass


def download_mini_imagenet_fix():
    from uutils import download_and_extract
    # download_and_extract('https://www.dropbox.com/s/ye9jeb5tyz0x01b/mini-imagenet-cache-test.pkl?dl=1',
    #                      '~/data/tmp', '~/data/tmp')
    download_and_extract('https://www.dropbox.com/s/ye9jeb5tyz0x01b/mini-imagenet-cache-test.pkl',
                         '~/data/tmp', '~/data/tmp')
    # download_and_extract('https://www.dropbox.com/s/9g8c6w345s2ek03/mini-imagenet-cache-train.pkl?dl=1',
    #                      '~/data/l2l_data', '~/data/l2l_data')
    # download_and_extract('https://www.dropbox.com/s/ip1b7se3gij3r1b/mini-imagenet-cache-validation.pkl?dl=1',
    #                      '~/data/l2l_data', '~/data/l2l_data')


# --

if __name__ == '__main__':
    mi_test()
    print('Done\a\n')
