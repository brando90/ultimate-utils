from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from torchvision.transforms import (Compose, ToPILImage, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Normalize)
import torch
import torch.distributions as dist
import torch.utils.data as data
"""
The benchmark modules provides a convenient interface to standardized benchmarks in the literature.
It provides train/validation/test TaskDatasets and TaskTransforms for pre-defined datasets.

This utility is useful for researchers to compare new algorithms against existing benchmarks.
For a more fine-grained control over tasks and data, we recommend directly using `l2l.data.TaskDataset` and `l2l.data.TaskTransforms`.
"""

import os
import numpy as np
import learn2learn as l2l

from collections import namedtuple


# from .omniglot_benchmark import omniglot_tasksets
# from .mini_imagenet_benchmark import mini_imagenet_tasksets
# from .tiered_imagenet_benchmark import tiered_imagenet_tasksets
# from .fc100_benchmark import fc100_tasksets
# from .cifarfs_benchmark import cifarfs_tasksets


def gaussian_1d_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        mu_B = 10,
        sigma_B = 10,
        root='~/data',
        data_augmentation=None,
        device=None,
        **kwargs,
):
    """Tasksets for mini-ImageNet benchmarks."""
    if data_augmentation is None:
        train_data_transforms = None
        test_data_transforms = None
    else:
        raise NotImplementedError

    '''train_dataset = l2l.vision.datasets.MiniImagenet(
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
    )'''

    train_dataset = MiniGaussiannet(mode = 'train', mu_B = mu_B, sigma_B=sigma_B)
    valid_dataset = MiniGaussiannet(mode='validation',mu_B = mu_B, sigma_B=sigma_B)
    test_dataset = MiniGaussiannet(mode='test',mu_B = mu_B, sigma_B=sigma_B)

    if device is None:
        train_dataset.transform = train_data_transforms
        valid_dataset.transform = train_data_transforms
        test_dataset.transform = test_data_transforms
    '''
    else: #For GPU Device?
        train_dataset = l2l.data.OnDeviceDataset(
            dataset=train_dataset,
            transform=train_data_transforms,
            device=device,
        )
        valid_dataset = l2l.data.OnDeviceDataset(
            dataset=valid_dataset,
            transform=train_data_transforms,
            device=device,
        )
        test_dataset = l2l.data.OnDeviceDataset(
            dataset=test_dataset,
            transform=test_data_transforms,
            device=device,
        )
    '''
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        NWays(train_dataset, train_ways),
        KShots(train_dataset, train_samples),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    valid_transforms = [
        NWays(valid_dataset, test_ways),
        KShots(valid_dataset, test_samples),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    test_transforms = [
        NWays(test_dataset, test_ways),
        KShots(test_dataset, test_samples),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]

    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms


__all__ = ['list_tasksets', 'get_tasksets']

BenchmarkTasksets = namedtuple('BenchmarkTasksets', ('train', 'validation', 'test'))

_TASKSETS = {
    # 'omniglot': omniglot_tasksets,
    # 'mini-imagenet': mini_imagenet_tasksets,
    'n_way_gaussians' : gaussian_1d_tasksets
    # 'tiered-imagenet': tiered_imagenet_tasksets,
    # 'fc100': fc100_tasksets,
    # 'cifarfs': cifarfs_tasksets,
}


def list_tasksets():
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/benchmarks/)

    **Description**

    Returns a list of all available benchmarks.

    **Example**
    ~~~python
    for name in l2l.vision.benchmarks.list_tasksets():
        print(name)
        tasksets = l2l.vision.benchmarks.get_tasksets(name)
    ~~~
    """
    return _TASKSETS.keys()


def get_tasksets(
        name,
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        mu_B = 10,
        sigma_B = 10,
        num_tasks=-1,
        #root='~/data',
        device=None,
        **kwargs,
):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/benchmarks/)

    **Description**

    Returns the tasksets for a particular benchmark, using literature standard data and task transformations.

    The returned object is a namedtuple with attributes `train`, `validation`, `test` which
    correspond to their respective TaskDatasets.
    See `examples/vision/maml_miniimagenet.py` for an example.

    **Arguments**

    * **name** (str) - The name of the benchmark. Full list in `list_tasksets()`.
    * **train_ways** (int, *optional*, default=5) - The number of classes per train tasks.
    * **train_samples** (int, *optional*, default=10) - The number of samples per train tasks.
    * **test_ways** (int, *optional*, default=5) - The number of classes per test tasks. Also used for validation tasks.
    * **test_samples** (int, *optional*, default=10) - The number of samples per test tasks. Also used for validation tasks.
    * **num_tasks** (int, *optional*, default=-1) - The number of tasks in each TaskDataset.
    * **device** (torch.Device, *optional*, default=None) - If not None, tasksets are loaded as Tensors on `device`.
    * **root** (str, *optional*, default='~/data') - Where the data is stored.

    **Example**
    ~~~python
    train_tasks, validation_tasks, test_tasks = l2l.vision.benchmarks.get_tasksets('omniglot')
    batch = train_tasks.sample()

    or:

    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot')
    batch = tasksets.train.sample()
    ~~~
    """
    #root = os.path.expanduser(root)

    # Load task-specific data and transforms
    datasets, transforms = _TASKSETS[name](train_ways=train_ways,
                                           train_samples=train_samples,
                                           test_ways=test_ways,
                                           test_samples=test_samples,
                                           mu_B = mu_B,
                                           sigma_B = sigma_B,
                                           #root=root,
                                           device=device,
                                           **kwargs)
    train_dataset, validation_dataset, test_dataset = datasets
    train_transforms, validation_transforms, test_transforms = transforms

    # Instantiate the tasksets
    train_tasks = l2l.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = l2l.data.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks,
    )
    test_tasks = l2l.data.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks,
    )
    return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks)

class MiniGaussiannet(data.Dataset):
    def __init__(self, mode='train', mu_B=10, sigma_B=10):
        """
        Create the three datasets based on mode
        If mode = train, we want to create 64 classes * 600 samples/class
        If mode = test, we create  20 classes * 600 samples/class
        If mode = val, we create 16 classes * 600 samples/class
        """
        self.x = []
        self.y = []

        samples_per_class = 600
        if (mode == 'train'):
            classes = 64
        elif (mode == 'test'):
            classes = 20
        else:
            classes = 16

        # Rho controls the spread of the class distribution (usually <1)
        rho = 0.3

        # Sample distribution of class from N(mu_B, sigma_B)
        task_dist = dist.Normal(mu_B * torch.ones(classes), sigma_B * torch.ones(classes))

        class_mus = task_dist.sample()
        class_sigmas = torch.abs(rho * task_dist.sample())

        for c in range(classes):
            for sample in range(samples_per_class):
                self.y.append(c)
                # Sample from N(mu_class, sigma_class)
                self.x.append(np.random.normal(class_mus[c], class_sigmas[c]))

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        #print(self.x)
        #print(self.y)

    def __getitem__(self, idx):
        data = self.x[idx]
        #if self.transform:
        #    data = self.transform(data)
        return data, self.y[idx]

    def __len__(self):
        return len(self.x)


# --

def gaussian_taskset_test():
    from argparse import Namespace
    from pathlib import Path

    args = Namespace(k_shots=5, k_eval=15, n_classes=5)
    args.data_option = 'n_way_gaussians'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
    #args.data_path = Path('~/data/l2l_data/').expanduser()
    #args.data_augmentation = 'lee2019'

    args.tasksets: BenchmarkTasksets = get_tasksets(
        args.data_option,
        train_samples=args.k_shots + args.k_eval,
        train_ways=args.n_classes,
        test_samples=args.k_shots + args.k_eval,
        test_ways=args.n_classes,
        #root=args.data_path,
        #data_augmentation=args.data_augmentation,
    )

    # try sampling!
    print(args.tasksets.train.sample())
    print(args.tasksets.test.sample())
    print(args.tasksets.validation.sample())


if __name__ == '__main__':
    gaussian_taskset_test()
    print('Done\a\n')
