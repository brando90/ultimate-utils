"""
Benchmark will be inspired from how meta-data set works. Thus it will work as follows:

Step 0) uniformly sample a dataset D,
Step 1) sample a set of classes C (n-way) from the classes of D assigned to the requested split, and
Step 2) sample support and query examples from C (n-way).

- each episode generated in META-DATASET uses classes from a single dataset
- (Moreover, two of these datasets, Traffic Signs and MSCOCO, are fully reserved for evaluation, meaning that no classes from them participate in the training set.
    - perhaps include two data sets like this at test time?)
- The remaining ones contribute some classes to each of the training, validation and test splits of classes, roughly with 70% / 15% / 15% proportions
- an algorithm that yields realistically imbalanced episodes of variable shots and ways.
- (For datasets without a known class organization, we sample the ‘way’ uniformly from the range [5, MAX-CLASSES], where MAX-CLASSES is either 50 or as many as there are available. Then we sample ‘way’ many classes uniformly at random from the requested class split of the given dataset.
    - remark: number of classes is different at each episode)
- The non-episodic baselines are trained to solve the large classification problem that results from ‘concatenating’ the training classes of all datasets.

discussion: https://github.com/learnables/learn2learn/issues/333
"""

import random
from typing import Callable

import learn2learn as l2l
import numpy as np
import torch
from learn2learn.data import TaskDataset, MetaDataset, DataDescription
from learn2learn.data.transforms import TaskTransform
from torch.utils.data import Dataset


class IndexableDataSet(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, idx: int):
        return self.datasets[idx]


class SingleDatasetPerTaskTransform(Callable):
    """
    Transform that samples a data set first, then creates a task (e.g. n-way, k-shot) and finally
    applies the remaining task transforms.
    """

    def __init__(self, indexable_dataset: IndexableDataSet, cons_remaining_task_transforms: Callable):
        """

        :param: cons_remaining_task_transforms; constructor that builds the remaining task transforms. Cannot be a list
        of transforms because we don't know apriori which is the data set we will use. So this function should be of
        type MetaDataset -> list[TaskTransforms] i.e. given the dataset it returns the transforms for it.
        """
        self.indexable_dataset = MetaDataset(indexable_dataset)
        self.cons_remaining_task_transforms = cons_remaining_task_transforms

    def __call__(self, task_description: list):
        """
        idea:
        - receives the index of the dataset to use
        - then use the normal NWays l2l function
        """
        # - this is what I wish could have gone in a seperate callable transform, but idk how since the transforms take apriori (not dynamically) which data set to use.
        i = random.randint(0, len(self.indexable_dataset) - 1)
        task_description = [DataDescription(index=i)]  # using this to follow the l2l convention

        # - get the sampled data set
        dataset_index = task_description[0].index
        dataset = self.indexable_dataset[dataset_index]
        dataset = MetaDataset(dataset)

        # - use the sampled data set to create task
        remaining_task_transforms: list[TaskTransform] = self.cons_remaining_task_transforms(dataset)
        description = None
        for transform in remaining_task_transforms:
            description = transform(description)
        return description


def sample_dataset(dataset):
    def sample_random_dataset(x):
        print(f'{x=}')
        i = random.randint(0, len(dataset) - 1)
        return [DataDescription(index=i)]
        # return dataset[i]

    return sample_random_dataset


def get_task_transforms(dataset: IndexableDataSet) -> list[TaskTransform]:
    """
    :param dataset:
    :return:
    """
    transforms = [
        sample_dataset(dataset),
        l2l.data.transforms.NWays(dataset, n=5),
        l2l.data.transforms.KShots(dataset, k=5),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ]
    return transforms


def print_datasets(dataset_lst: list):
    for dataset in dataset_lst:
        print(f'\n{dataset=}\n')


def get_indexable_list_of_datasets_mi_and_cifarfs(root: str = '~/data/l2l_data/') -> IndexableDataSet:
    from learn2learn.vision.benchmarks import mini_imagenet_tasksets
    datasets, transforms = mini_imagenet_tasksets(root=root)
    mi = datasets[0].dataset

    from learn2learn.vision.benchmarks import cifarfs_tasksets
    datasets, transforms = cifarfs_tasksets(root=root)
    cifarfs = datasets[0].dataset

    dataset_list = [mi, cifarfs]

    dataset_list = [l2l.data.MetaDataset(dataset) for dataset in dataset_list]
    dataset = IndexableDataSet(dataset_list)
    return dataset


# -- tests

def loop_through_l2l_indexable_datasets_test():
    """
    Generate


    idea: use omniglot instead of letters? It already has a split etc.

    :return:
    """
    # - for determinism
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # - options for number of tasks/meta-batch size
    batch_size: int = 10

    # - create indexable data set
    indexable_dataset: IndexableDataSet = get_indexable_list_of_datasets_mi_and_cifarfs()

    # - get task transforms
    def get_remaining_transforms(dataset: MetaDataset) -> list[TaskTransform]:
        remaining_task_transforms = [
            l2l.data.transforms.NWays(dataset, n=5),
            l2l.data.transforms.KShots(dataset, k=5),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset),
        ]
        return remaining_task_transforms
    task_transforms: TaskTransform = SingleDatasetPerTaskTransform(indexable_dataset, get_remaining_transforms)

    # -
    taskset: TaskDataset = TaskDataset(dataset=indexable_dataset, task_transforms=task_transforms)

    # - loop through tasks
    for task_num in range(batch_size):
        print(f'{task_num=}')
        X, y = taskset.sample()
        print(f'{X.size()=}')
        print(f'{y.size()=}')
        print(f'{y=}')
        print()

    print('-- end of test --')


# -- Run experiment

if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_through_l2l_indexable_datasets_test()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
