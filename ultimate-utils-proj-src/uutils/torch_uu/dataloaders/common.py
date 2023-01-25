"""

# - creating data train, val, test data sets & data laoders

ref:
    - https://stackoverflow.com/questions/70400439/what-is-the-proper-way-to-create-training-validation-and-test-set-in-pytorch-or

# - data augmentation
Current belief is that augmenting the validation set should be fine, especially if you want to actually encourage
generalization since it makes the val set harder and it allows you to make val split percentage slightly lower since
your validation set was increased size.

For reproducibility of other work, especially for scientific pursues rather than "lets beat state of the art" - to make
it easier to compare results use what they use. e.g. it seems only augmenting the train set is the common thing,
especially when I looked at the augmentation strategies in min-imagenet and mnist.

Test set augmentation helps mostly to make test set harder (so acc should go down) - but it also increases variance
since the data size was increased. If you are reporting results most likely augmenting the data set is a good idea
- especially if you are going to compute test set errors when comparing accuracy with previous work.
Also, the way CI intervals are computed with t_p * std_n / sqrt n, means that the avg test error will be smaller, so
you are fine in general.
Default code I see doesn't augment test set so I most likely won't either.

ref:
    - https://stats.stackexchange.com/questions/320800/data-augmentation-on-training-set-only/320967#320967
    - https://arxiv.org/abs/1809.01442, https://stats.stackexchange.com/a/390470/28986

# - pin_memory
For data loading, passing pin_memory=True to a DataLoader will automatically put the fetched data Tensors in pinned
memory, and thus enables faster data transfer to CUDA-enabled GPUs. Note on pinning:
This is an advanced tip. If you overuse pinned memory, it can cause serious problems when running low on RAM, and
you should be aware that pinning is often an expensive operation. Thus, will leave it's default as False.

ref:
    - on pin_memory: https://pytorch.org/docs/stable/data.html
"""
from argparse import Namespace
from typing import Callable, Optional, Union, Any

import numpy as np
import torch
from numpy.random import RandomState
from torch.utils.data import Dataset, SubsetRandomSampler, random_split, DataLoader, RandomSampler

from pdb import set_trace as st


def get_train_val_split_random_sampler(
        train_dataset: Dataset,
        val_dataset: Dataset,
        val_size: float = 0.2,
        batch_size: int = 128,
        batch_size_eval: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False
        #     random_seed: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Note:
        - this will use different transforms for val and train if the objects you pass have different transforms.
        - note train_dataset, val_dataset whill often be the same data set object but different instances with different
        transforms for each data set.

    Recommended use:
        - this one is recommended when you want the train & val to have different transforms e.g. when doing scientific
        work - instead of beating benchmark work - and the train, val sets had different transforms.

    ref:
        - https://gist.github.com/MattKleinsmith/5226a94bad5dd12ed0b871aed98cb123
    """
    assert 0 <= val_size <= 1.0, f"Error: {val_size} valid_size should be in the range [0, 1]."
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split_idx = int(np.floor(val_size * num_train))

    # I don't think this is needed since later the sampler randomly samples data from a given list
    # if shuffle == True:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)

    train_idx, valid_idx = indices[:split_idx], indices[split_idx:]
    assert len(train_idx) != 0 and len(valid_idx) != 0

    # Samples elements randomly from a given list of indices, without replacement.
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size_eval, sampler=valid_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader


def get_train_val_split_with_split(
        train_dataset: Dataset,
        train_val_split: list[int, int],  # e.g. [50_000, 10_000] for mnist
        batch_size: int = 128,
        batch_size_eval: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False
) -> tuple[DataLoader, DataLoader]:
    """
    Note:
        - this will have the train and val sets have the same transform.

    ref:
        - https://gist.github.com/MattKleinsmith/5226a94bad5dd12ed0b871aed98cb123
        - change transform: https://discuss.pytorch.org/t/changing-transforms-after-creating-a-dataset/64929/4
    """
    train_dataset, valid_dataset = random_split(train_dataset, train_val_split)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size_eval, num_workers=num_workers,
                                               pin_memory=pin_memory)
    return train_loader, valid_loader


def get_serial_or_distributed_dataloaders(train_dataset: Dataset,
                                          val_dataset: Dataset,
                                          batch_size: int = 128,
                                          batch_size_eval: int = 64,
                                          rank: int = -1,
                                          world_size: int = 1,
                                          merge: Optional[Callable] = None,
                                          num_workers: int = -1,  # -1 means its running serially
                                          pin_memory: bool = False,
                                          ):
    """

    """
    from uutils.torch_uu.distributed import is_running_serially
    if is_running_serially(rank):
        train_sampler = RandomSampler(train_dataset)
        val_sampler = RandomSampler(val_dataset)
        num_workers = 4 if num_workers == -1 else num_workers
    else:
        assert (batch_size >= world_size), f'Each worker must get at least one data point, so batch_size >= world_size' \
                                           f'but got: {batch_size}{world_size}'
        from torch.utils.data import DistributedSampler

        # note: shuffle = True by default
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        # set the input num_workers but for ddp 0 is recommended afaik, todo - check
        num_workers = 0 if num_workers == -1 else num_workers
    # get dist dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=merge,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size_eval,
                            sampler=val_sampler,
                            collate_fn=merge,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    # return dataloaders
    # dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    # iter(train_dataloader)  # if this fails its likely your running in pycharm and need to set num_workers flag to 0
    return train_loader, val_loader


def split_inidices(indices: list,
                   test_size: Optional = None,
                   random_state: Optional[Union[int, RandomState, None]] = None,
                   shuffle: bool = False,  # false for reproducibility, and any split is as good as any other.
                   ) -> tuple[list[int], list[int]]:
    import sklearn.model_selection
    # - api: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    train_indices, val_indices = sklearn.model_selection.train_test_split(indices, test_size=test_size,
                                                                          random_state=random_state,
                                                                          shuffle=shuffle)
    return train_indices, val_indices


# - size of data sets from data loader

def get_data_set_size_from_dataloader(dataloader: DataLoader) -> int:
    """
    Note:
        - this is a hacky way to get the size of a dataset from a dataloader
        - todo: fix for mds and indexable data loaders
    """
    return len(dataloader.dataset)


def get_data_set_size_from_taskset(taskset) -> int:
    """
    Note:
        - this is a hacky way to get the size of a dataset from a taskset

    # cases:
        - case 0: SingleDataset
            -
        - case 1: IndexableDataset
            - sum( [ len(taskset.train.dataset.dataset.datasets[i].dataset) for i in range(len(taskset.train.dataset.dataset.datasets))] )
    """
    # - make type explicit
    # already did the .train .validation or .test
    from learn2learn.data import TaskDataset
    taskset: TaskDataset = taskset

    # -- Get size of data set
    # - case 1: IndexableDataset
    try:
        # its easier to just fail this and try to do single than to do them "in order"
        from diversity_src.dataloaders.common import IndexableDataSet
        assert isinstance(taskset.dataset.dataset, IndexableDataSet)
        # return sum([len(taskset.dataset.dataset.datasets[i].dataset) for i in range(len(taskset.dataset.dataset.datasets))])
        datasets = taskset.dataset.dataset.datasets
        size: int = 0
        for metadataset in datasets:  # for i in range(len(taskset.dataset.dataset.datasets)
            # print(metadataset)
            # print(dataset)
            # print(type(metadataset))
            # print(type(dataset)
            dataset = metadataset.dataset
            size += len(dataset)
        return size
    except Exception as e:
        # - case 0: SingleDataset
        # do return len(metadataset.dataset)
        dataset = metadataset.dataset
        return len(dataset)


def get_data_set_sizes_from_dataloaders(dataloaders: dict[str, DataLoader]) -> dict[str, int]:
    """
    Note:
        - this is a hacky way to get the size of a dataset from a dataloader
    """
    return {k: get_data_set_size_from_dataloader(v) for k, v in dataloaders.items()}


def get_data_set_sizes_from_tasksets(tasksets: dict[str, Any]) -> dict[str, int]:
    """
    Note:
        - this is a hacky way to get the size of a dataset from a taskset
    """
    splits = ['train', 'validation', 'test']
    # return {k: get_data_set_size_from_taskset(v) for k, v in tasksets.items()}
    return {split: get_data_set_size_from_taskset(getattr(tasksets, split)) for split in splits}


def get_dataset_size(args: Namespace) -> dict[str, int]:
    split_2_size: dict[str, int] = None
    try:
        dataloaders = args.dataloaders
        split_2_size = get_data_set_size_from_dataloader(dataloaders)
    except Exception as e:
        # print(f'{e=}')
        tasksets = args.tasksets
        split_2_size = get_data_set_sizes_from_tasksets(tasksets)
    return split_2_size


# - get num classes

def get_num_classes_l2l_list_meta_dataset(dataset_list: list, verbose: bool = False) -> dict:
    """ Get the number of classes in a list of l2l meta dataset"""
    from learn2learn.data import MetaDataset
    dataset_list: list[MetaDataset] = dataset_list
    # - get num classes for each data set & total, total is one we really need
    results: dict = dict(total=0)
    for dataset in dataset_list:
        num_classes = len(dataset.labels)
        if hasattr(dataset, 'name'):
            results[dataset.name] = num_classes
        else:
            results[str(type(dataset))] = num_classes
        results['total'] += num_classes
    if verbose:
        print(f'Number of classes: \n {results=}')
        from pprint import pprint
        print(f'Number of classes:')
        pprint(results)
    return results
