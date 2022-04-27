from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from torchvision.transforms import (Compose, ToPILImage, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Normalize)
import torch
import torch.distributions as dist
import torch.utils.data as data
from argparse import Namespace
"""
The benchmark modules provides a convenient interface to standardized benchmarks in the literature.
It provides train/validation/test TaskDatasets and TaskTransforms for pre-defined datasets.
This utility is useful for researchers to compare new algorithms against existing benchmarks.
For a more fine-grained control over tasks and data, we recommend directly using `l2l.data.TaskDataset` and `l2l.data.TaskTransforms`.
"""

import os
import numpy as np
import learn2learn as l2l
import pickle

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
        mu_m_B = 10,
        sigma_m_B = 10,
        mu_s_B = 3,
        sigma_s_B = 1,
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

    train_dataset = MiniGaussiannet(mode = 'train', mu_m_B = mu_m_B, sigma_m_B=sigma_m_B, mu_s_B = mu_s_B, sigma_s_B = sigma_s_B)
    valid_dataset = MiniGaussiannet(mode='validation',mu_m_B = mu_m_B, sigma_m_B=sigma_m_B, mu_s_B = mu_s_B, sigma_s_B = sigma_s_B)
    test_dataset = MiniGaussiannet(mode='test',mu_m_B = mu_m_B, sigma_m_B=sigma_m_B,mu_s_B = mu_s_B,  sigma_s_B = sigma_s_B)

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
        mu_m_B=10,
        sigma_m_B=10,
        mu_s_B=3,
        sigma_s_B=1,
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
                                           mu_m_B=mu_m_B,
                                           sigma_m_B=sigma_m_B,
                                           mu_s_B=mu_s_B,
                                           sigma_s_B=sigma_s_B,
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

# See https://github.com/learnables/learn2learn/issues/301
'''
def get_train_valid_test_data_loader_1d_gaussian(args: Namespace) -> dict:
    """
    Trying to do:
        type(args.tasksets.train.dataset.dataset)
        <class 'learn2learn.vision.datasets.cifarfs.CIFARFS'>

    :param tasksets:
    :param split:
    :return:
    """
    tasksets =   get_tasksets(
            args.data_option,
            train_samples=args.k_shots + args.k_eval, #k shots for meta-train, k eval for meta-validaton/eval
            train_ways=args.n_classes,
            test_samples=args.k_shots + args.k_eval,
            test_ways=args.n_classes,
            mu_m_B=args.mu_m_B,
            sigma_m_B=args.sigma_m_B,
            mu_s_B=args.mu_s_B,
            sigma_s_B=args.sigma_s_B,
            #root=args.data_path, #No need for datafile
            #data_augmentation=args.data_augmentation, #TODO: currently not implemented! Do we need to implement?
        )

    # trying to do something like: args.tasksets.train
    taskset: TaskDataset = getattr(tasksets, split)
    # trying to do: type(args.tasksets.train.dataset.dataset)
    dataset: MetaDataset = taskset.dataset
    dataset: Dataset = dataset.dataset
    # assert isinstance(args.tasksets.train.dataset.dataset, Dataset)
    # asser isinstance(tasksets.train.dataset.dataset, Dataset)
    assert isinstance(dataset, Dataset), f'Expect dataset to be of type Dataset but got {type(dataset)=}.'
    return dataset
'''

def get_train_valid_test_data_loader_1d_gaussian(args: Namespace) -> dict:
    train_dataset = MiniGaussiannet(mode='train', mu_m_B=args.mu_m_B, sigma_m_B=args.sigma_m_B, mu_s_B=args.mu_s_B,
                                    sigma_s_B=args.sigma_s_B)
    valid_dataset = MiniGaussiannet(mode='validation', mu_m_B=args.mu_m_B, sigma_m_B=args.sigma_m_B, mu_s_B=args.mu_s_B,
                                    sigma_s_B=args.sigma_s_B)
    test_dataset = MiniGaussiannet(mode='test', mu_m_B=args.mu_m_B, sigma_m_B=args.sigma_m_B, mu_s_B=args.mu_s_B, sigma_s_B=args.sigma_s_B)

    #---DEBUG---#
    #from torch.utils.data import RandomSampler
    #train_sampler = RandomSampler(train_dataset)
    #print(train_sampler)
    #----------#
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
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    print(dataloaders)
    return dataloaders

class MiniGaussiannet(data.Dataset):
    def __init__(self, mode='train', mu_m_B=10, sigma_m_B=10, mu_s_B=3, sigma_s_B=1):
        """
        Create the three datasets, each datasets have 100 classes and 1000 samples perclass
        """

        #4/5 added pickle functionality
        #TODO: Tell brando I made a HUGE HUGE realization - forgot to specify the mode! so basically we were training/testing/validating on the SAME dataset?!?!?!
        dataset_filename = os.path.join(os.path.expanduser('~'),"1d_gaussian_datasets/1d_gaussian_%s_%s_%s_%s_%s.pkl" % (mode, mu_m_B,sigma_m_B,mu_s_B,sigma_s_B))
        print("Trying to find pickled 1d gaussian dataset for current benchmark...")
        try:
            self.x, self.y = pickle.load(open(dataset_filename,"rb"))
            print("Found pickled dataset for benchmark!")
        except (OSError, IOError) as e:
            print("Didn't find pickled dataset for benchmark. Creating one...")
            self.x = []
            self.y = []

            #3/30 changed to 1000 samples/class, 100/100/100 class split
            samples_per_class = 1000#10#1000#600#1000
            if (mode == 'train'):
                classes = 100#10#100#64#100#
            elif (mode == 'test'):
                classes = 100#10#100#20#100#
            else:
                classes = 100#10#100#16 ##

            # Sample mu_class ~ N(mu_m_B, sigma_m_B), sigma_class ~ N(mu_s_B, sigma_s_B)
            task_mu_dist = dist.Normal(mu_m_B * torch.ones(classes), sigma_m_B * torch.ones(classes))
            task_sigma_dist = dist.Normal(mu_s_B * torch.ones(classes), sigma_s_B * torch.ones(classes))

            class_mus = task_mu_dist.sample()
            class_sigmas = torch.abs(task_sigma_dist.sample()) #torch.abs(rho * task_dist.sample())

            #Add a permutation to the classes, e.g. [0,1,2,3,4] => [4,0,1,2,3]
            for c in np.random.permutation(classes):#range(classes):
                class_dist = dist.Normal(class_mus[c], class_sigmas[c])
                for sample in range(samples_per_class):
                    self.y.append(c) #float(c) previously
                    self.x.append(class_dist.sample().numpy().reshape(-1,1,1))

            self.x = np.array(self.x)
            self.y = np.array(self.y,dtype=np.int64)
            print("self.y", self.y)
            pickle.dump((self.x, self.y), open(dataset_filename,"wb"))
            print("Sucessfully created pickled dataset for benchmark!")


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
    args.mu_m_B = 0 #doesn't matter
    args.sigma_m_B = 10
    args.mu_s_B = 2
    args.sigma_s_B = 1
    args.batch_size = 8  # 256 #TODO: How to make learning rates fair comparison with MAML?
    args.batch_size_eval = 8  # TODO: WHat is this?
    args.world_size = 1
    args.rank = -1
    #args.data_path = Path('~/data/l2l_data/').expanduser()
    #args.data_augmentation = 'lee2019'

    args.tasksets: BenchmarkTasksets = get_tasksets(
        args.data_option,
        train_samples=args.k_shots + args.k_eval,
        train_ways=args.n_classes,
        test_samples=args.k_shots + args.k_eval,
        test_ways=args.n_classes,
        mu_m_B=args.mu_m_B,
        sigma_m_B=args.sigma_m_B,
        mu_s_B=args.mu_s_B,
        sigma_s_B=args.sigma_s_B
        #root=args.data_path,
        #data_augmentation=args.data_augmentation,
    )

    # try sampling!
    print(args.tasksets.train.sample())
    print(args.tasksets.test.sample())
    print(args.tasksets.validation.sample())

    #Try sampling from the USL dataloader
    usl_1d_gaussian_tasks = get_train_valid_test_data_loader_1d_gaussian(args)
    #Need to make this not float!
    print(next(iter(usl_1d_gaussian_tasks['train'])))
    print(next(iter(usl_1d_gaussian_tasks['test'])))
    print(next(iter(usl_1d_gaussian_tasks['val'])))


if __name__ == '__main__':
    gaussian_taskset_test()
    print('Done\a\n')