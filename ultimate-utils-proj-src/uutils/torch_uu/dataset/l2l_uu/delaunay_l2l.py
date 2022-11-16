"""
Outline of how to create a l2l taskloader:
- for now see https://github.com/learnables/learn2learn/issues/375

note:
    - to create a task loader from a new data set: https://github.com/learnables/learn2learn/issues/375
    - remember we have a l2l to torchmeta loader converter. So for now all code is written in l2l's api.
"""
import os

import learn2learn
from learn2learn.vision.benchmarks import BenchmarkTasksets
from learn2learn.data import MetaDataset
from torch.utils.data import Dataset

from uutils.torch_uu.dataset.delaunay_uu import get_l2l_bm_split_paths, get_delauny_dataset_splits

from learn2learn.data.transforms import TaskTransform


def get_remaining_delauny_l2l_task_transforms(dataset: MetaDataset, ways: int, samples: int) -> list[TaskTransform]:
    """
    Get delauny task transforms to create fsl tasks. Same as mi.

    Same as MI to create fsl tasks: NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels.
    """
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_remaining_transforms_mi
    remaining_task_transforms: TaskTransform = get_remaining_transforms_mi(dataset, ways, samples)
    return remaining_task_transforms


def get_delauny_l2l_datasets_and_task_transforms(train_ways=5,
                                                 train_samples=10,
                                                 test_ways=5,
                                                 test_samples=10,
                                                 root: str = '~/data/delauny_l2l_bm_splits',
                                                 data_augmentation: str = 'delauny_uu',
                                                 device=None,
                                                 **kwargs,
                                                 ):
    """ """
    # - get data set with its data transfroms
    path2train, path2val, path2test = get_l2l_bm_split_paths(root)
    train_dataset, valid_dataset, test_dataset = get_delauny_dataset_splits(path2train, path2val, path2test,
                                                                            data_augmentation)
    if device is None:
        pass  # in the original l2l it hardcodes a side effect to change the data transforms for some reason, we already put the data transforms so do nothing
    else:
        train_dataset = learn2learn.OnDeviceDataset(
            dataset=train_dataset,
            transform=train_dataset.transform,
            device=device,
        )
        valid_dataset = learn2learn.OnDeviceDataset(
            dataset=valid_dataset,
            transform=valid_dataset.transform,
            device=device,
        )
        test_dataset = learn2learn.OnDeviceDataset(
            dataset=test_dataset,
            transform=test_dataset.transform,
            device=device,
        )
    train_dataset: MetaDataset = MetaDataset(train_dataset)
    valid_dataset: MetaDataset = MetaDataset(valid_dataset)
    test_dataset: MetaDataset = MetaDataset(test_dataset)

    # - decided same as MI transforms because I don't acppreciate the omniglot one + delauny seems more similar to mi? (since it has colours...?)
    train_transforms = get_remaining_delauny_l2l_task_transforms(train_dataset, train_ways, train_samples)
    valid_transforms = get_remaining_delauny_l2l_task_transforms(valid_dataset, test_ways, test_samples)
    test_transforms = get_remaining_delauny_l2l_task_transforms(test_dataset, test_ways, test_samples)

    # - add names to be able to get the right task transform for the indexable dataset
    train_dataset.name = 'train_delauny_uu'
    valid_dataset.name = 'val_delauny_uu'
    test_dataset.name = 'test_delauny_uu'

    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms


def get_delauny_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,  # let it be -1 for continual tasks https://github.com/learnables/learn2learn/issues/315
        root='~/data/delauny_l2l_bm_splits',
        device=None,
        **kwargs,
) -> BenchmarkTasksets:
    """ """
    root = os.path.expanduser(root)
    # Load task-specific data and transforms
    datasets, transforms = get_delauny_l2l_datasets_and_task_transforms(train_ways=train_ways,
                                                                        train_samples=train_samples,
                                                                        test_ways=test_ways,
                                                                        test_samples=test_samples,
                                                                        root=root,
                                                                        device=device,
                                                                        **kwargs)
    train_dataset, validation_dataset, test_dataset = datasets
    train_transforms, validation_transforms, test_transforms = transforms

    # Instantiate the tasksets
    train_tasks = learn2learn.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = learn2learn.data.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks,
    )
    test_tasks = learn2learn.data.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks,
    )
    return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks)


# -- Run experiment

def loop_through_delaunay():
    print(f'test: {loop_through_delaunay=}')
    # - for determinism
    import random
    random.seed(0)
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)

    # - options for number of tasks/meta-batch size
    batch_size = 2

    # - get benchmark
    benchmark: BenchmarkTasksets = get_delauny_tasksets()
    splits = ['train', 'validation', 'test']
    tasksets = [getattr(benchmark, split) for split in splits]

    # - loop through tasks
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # from models import get_model
    # model = get_model('resnet18', pretrained=False, num_classes=5).to(device)
    # model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V2")
    # model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
    model, _ = get_default_learner_and_hps_dict()  # 5cnn
    model.to(device)
    from torch import nn
    criterion = nn.CrossEntropyLoss()
    for i, taskset in enumerate(tasksets):
        print(f'-- {splits[i]=}')
        for task_num in range(batch_size):
            print(f'{task_num=}')

            X, y = taskset.sample()
            print(f'{X.size()=}')
            print(f'{y.size()=}')
            print(f'{y=}')

            y_pred = model(X)
            loss = criterion(y_pred, y)
            print(f'{loss=}')
            print()
            break
    print(f'done test: {loop_through_delaunay=}')


if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    loop_through_delaunay()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
