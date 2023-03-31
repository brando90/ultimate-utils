"""

ref:
    - official SL training from l2l: https://github.com/learnables/learn2learn/blob/master/examples/vision/supervised_pretraining.py
    didn't know about it on time. Alas.
"""
import datasets
from pathlib import Path

import os
from argparse import Namespace
from torch import Tensor

from uutils import expanduser
from torch.utils.data import Dataset

from uutils.torch_uu.dataset.concate_dataset import ConcatDatasetMutuallyExclusiveLabels

from pdb import set_trace as st


# I know sort of ugly, but idk if it requires me to refactor a lot of code, for now this is good enough (and less confusing than not having it) todo fix later as the function bellow too
def hdb1_mi_omniglot_usl_all_splits_dataloaders2(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb1',
        device=None,
) -> dict:
    """ Get the usl data loaders for the hdb1_mio (Meta-L) dataset. """
    from diversity_src.dataloaders.usl.hdb1_mi_omniglot_usl_dl import hdb1_mi_omniglot_usl_all_splits_dataloaders
    return hdb1_mi_omniglot_usl_all_splits_dataloaders(args, root, data_augmentation, device)


def hdb4_micod_usl_all_splits_dataloaders(
        args: Namespace,
        root: str = '~/data/l2l_data/',
        data_augmentation='hdb4_micod',
        device=None,
) -> dict:
    """ Get the usl data loaders for the hdb4_micod (Meta-L) dataset. """
    print(f'----> {data_augmentation=}')
    print(f'{hdb4_micod_usl_all_splits_dataloaders=}')
    # root = os.path.expanduser(root)
    root: Path = expanduser(root)
    from diversity_src.dataloaders.hdb4_micod_l2l import get_hdb4_micod_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_hdb4_micod_list_data_set_splits(root,
                                                                                                         data_augmentation,
                                                                                                         device)
    # - print the number of classes in each split
    # print('-- Printing num classes')
    # from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
    # get_num_classes_l2l_list_meta_dataset(dataset_list_train, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # get_num_classes_l2l_list_meta_dataset(dataset_list_validation, verbose=True)
    # - concat l2l datasets to get usl single dataset
    relabel_filename: str = 'hdb4_micod_train_relabel_usl.pt'
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train, root, relabel_filename)
    relabel_filename: str = 'hdb4_micod_val_relabel_usl.pt'
    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_validation, root, relabel_filename)
    relabel_filename: str = 'hdb4_micod_test_relabel_usl.pt'
    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test, root, relabel_filename)
    assert len(train_dataset.labels) == 64 + 34 + 64 + 1100, f'Err:\n{len(train_dataset.labels)=}'
    # - get data loaders, see the usual data loader you use
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )

    # next(iter(train_loader))
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # next(iter(dataloaders[split]))
    return dataloaders


def get_len_labels_list_datasets(datasets: list[Dataset], verbose: bool = False) -> int:
    if verbose:
        print('--- get_len_labels_list_datasets')
        print([len(dataset.labels) for dataset in datasets])
        print([dataset.labels for dataset in datasets])
        print('--- get_len_labels_list_datasets')
    return sum([len(dataset.labels) for dataset in datasets])


# - from normal l2l to usl

def get_pytorch_dataloaders_from_regular_l2l_tasksets(args) -> dict:
    """ Get the pytorch dataloaders from the l2l tasksets."""
    from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
    from learn2learn.vision.benchmarks import BenchmarkTasksets
    # - get l2l tasksets
    # the k_shots, k_eval aren't needed for usl but the code needs it bellow. Any value is ok because l2l uses them for TaskTransform, which we ignore anyway for usl
    args.k_shots, args.k_eval = 5, 15  # any value is fine, usl doesn't use it anyway
    tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
    train_dataset, valid_dataset, test_dataset = get_dataset_from_l2l_tasksets(tasksets)

    # - some safety defaults
    # args.batch_size_eval = args.batch_size if not hasattr(args, 'batch_size_eval') else args.batch_size_eval
    # args.rank = 0 if not hasattr(args, 'rank') else args.rank

    # - get pytorch dataloaders from l2l tasksets
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    train_loader, val_loader = get_serial_or_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    _, test_loader = get_serial_or_distributed_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        rank=args.rank,
        world_size=args.world_size
    )
    # next(iter(train_loader))
    # st()

    # - get dict of dataloaders
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders


def get_dataset_from_l2l_tasksets(tasksets) -> tuple[Dataset, Dataset, Dataset]:
    """
    Get the datasets from the l2l tasksets.

    (Pdb) type(tasksets.train)
    <class 'learn2learn.data.task_dataset.TaskDataset'>
    (Pdb) type(tasksets.train.dataset)
    <class 'learn2learn.data.meta_dataset.MetaDataset'>
    (Pdb) type(tasksets.train.dataset.dataset)
    <class 'learn2learn.vision.datasets.mini_imagenet.MiniImagenet'>
    """
    from learn2learn.vision.benchmarks import BenchmarkTasksets
    from learn2learn.vision.datasets import MiniImagenet
    from learn2learn.data import TaskDataset, MetaDataset
    # - make type explicit of tasksets
    tasksets: BenchmarkTasksets = tasksets

    # - get the datasets
    if isinstance(tasksets.train, TaskDataset):
        # - get the actual data set object (perhaps inside another dataset object e.g. a MetaDataset)
        # todo: UnionMetaDataset, FilteredMetaDataset, ConcatDataset, ConcatDatasetMutuallyExclusiveLabels
        if isinstance(tasksets.train.dataset, MetaDataset):
            train_dataset = tasksets.train.dataset.dataset
            valid_dataset = tasksets.validation.dataset.dataset
            test_dataset = tasksets.test.dataset.dataset
        # elif isinstance(tasksets.train.dataset, FilteredMetaDataset):
        #     raise NotImplementedError
        # elif isinstance(tasksets.train.dataset, UnionMetaDataset):
        #     raise NotImplementedError
        # elif isinstance(tasksets.train.dataset, ConcatDataset):
        #     raise NotImplementedError
        # elif isinstance(tasksets.train.dataset, ConcatDatasetMutuallyExclusiveLabels):
        #     raise NotImplementedError
        else:
            # raise ValueError(f'not implemented for {type(tasksets.train.dataset)}')
            train_dataset = tasksets.train.dataset
            valid_dataset = tasksets.validation.dataset
            test_dataset = tasksets.test.dataset
        # - fix it if its MI
        print(f'{type(train_dataset)=}')
        if isinstance(train_dataset, MiniImagenet):
            train_dataset = USLDatasetFromL2L(train_dataset)
            valid_dataset = USLDatasetFromL2L(valid_dataset)
            test_dataset = USLDatasetFromL2L(test_dataset)
            print(f'{train_dataset[0][1]=}')
            assert not isinstance(train_dataset[0][1], float)
        return train_dataset, valid_dataset, test_dataset
    else:
        raise ValueError(f'not implemented for {type(tasksets.train)}')


class USLDatasetFromL2L(datasets.Dataset):

    def __init__(self, original_l2l_dataset: datasets.Dataset):
        self.original_l2l_dataset = original_l2l_dataset
        self.transform = self.original_l2l_dataset.transform
        self.original_l2l_dataset.target_transform = label_to_long
        self.target_transform = self.original_l2l_dataset.target_transform

    def __getitem__(self, index: int) -> tuple:
        """ overwrite the getitem method for a l2l dataset. """
        # - get the item
        img, label = self.original_l2l_dataset[index]
        # - transform the item only if the transform does exist and its not a tensor already
        # img, label = self.original_l2l_dataset.x, self.original_l2l_dataset.y
        if self.transform and not isinstance(img, Tensor):
            img = self.transform(img)
        if self.target_transform and not isinstance(label, Tensor):
            label = self.target_transform(label)
        return img, label

    def __len__(self) -> int:
        """ Get the length. """
        return len(self.original_l2l_dataset)


def label_to_long(label: int) -> int:
    """ Convert the label to long. """
    return int(label)
    # convert label to pytorch long
    # return torch.tensor(label, dtype=torch.long)
    # import torch
    # return torch.tensor(label, dtype=torch.long)


# - tests

def loop_through_usl_hdb_and_pass_data_through_mdl():
    print(f'starting {loop_through_usl_hdb_and_pass_data_through_mdl=} test')
    # - for determinism
    import random
    random.seed(0)
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)

    # - args
    args = Namespace(batch_size=8, batch_size_eval=2, rank=-1, world_size=1)

    # - get data loaders
    # dataloaders: dict = hdb1_mi_omniglot_usl_all_splits_dataloaders(args)
    dataloaders: dict = hdb4_micod_usl_all_splits_dataloaders(args)
    print(dataloaders['train'].dataset.labels)
    print(dataloaders['val'].dataset.labels)
    print(dataloaders['test'].dataset.labels)
    n_train_cls: int = len(dataloaders['train'].dataset.labels)
    print('-- got the usl hdb data loaders --')

    # - loop through tasks
    import torch
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # model = get_model('resnet18', pretrained=False, num_classes=n_train_cls).to(device)
    # model = get_model('resnet18', pretrained=True, num_classes=n_train_cls).to(device)
    # from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_mi
    # model, _ = get_resnet_rfs_model_mi('resnet12_rfs', num_classes=n_train_cls)
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
    args.model, args.model_hps = get_default_learner_and_hps_dict()
    # - get model
    model = args.model
    model.to(device)
    from torch import nn
    criterion = nn.CrossEntropyLoss()
    for split, dataloader in dataloaders.items():
        print(f'-- {split=}')
        # next(iter(dataloaders[split]))
        for it, batch in enumerate(dataloaders[split]):
            print(f'{it=}')

            X, y = batch
            print(f'{X.size()=}')
            print(f'{y.size()=}')
            print(f'{y=}')

            y_pred = model(X)
            print(f'{y_pred.size()=}')
            # loss = criterion(y_pred, y)
            # print(f'{loss=}')
            print()
            break
    print('-- end of test --')


if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_through_usl_hdb_and_pass_data_through_mdl()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
