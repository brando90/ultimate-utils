"""
refs:
    - inspired from: https://github.com/WangYueFt/rfs/blob/master/train_supervised.py
    - normalization: https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/dataset/transform_cfg.py#L104
"""
from argparse import Namespace
from pathlib import Path
from typing import Optional, Callable

import torch
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, Compose

import os
import pickle
from PIL import Image
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset

# - cifar100 normalization according to rfs
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
normalize_cifar100 = transforms.Normalize(mean=mean, std=std)


def get_train_valid_test_data_loader_helper_for_cifarfs(args: Namespace) -> dict:
    train_kwargs = {'args': args,
                    'data_path': args.data_path,
                    'batch_size': args.batch_size,
                    'batch_size_eval': args.batch_size_eval,
                    'augment_train': args.augment_train,
                    'augment_val': args.augment_val,
                    'num_workers': args.num_workers,
                    'pin_memory': args.pin_memory,
                    'rank': args.rank,
                    'world_size': args.world_size,
                    'merge': None
                    }
    dataloaders: dict = get_rfs_union_sl_dataloader_cifarfs(**train_kwargs)
    return dataloaders


def get_transform(augment: bool):
    if augment:
        transform = transforms.Compose([
            # lambda x: Image.fromarray(x),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            # lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize_cifar100
        ])
    else:
        transform = transforms.Compose([
            # lambda x: Image.fromarray(x),
            transforms.ToTensor(),
            normalize_cifar100
        ])
    return transform


def get_transform_rfs(augment: bool):
    """
    this won't work for l2l data sets.
    """
    if augment:
        transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize_cifar100
        ])
    else:
        transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
            normalize_cifar100
        ])
    return transform


class CIFAR100(Dataset):
    """support FC100 and CIFAR-FS"""

    def __init__(self, data_root, data_aug,
                 partition='train', pretrain=True,
                 is_sample=False, k=4096,
                 transform=None):
        super(Dataset, self).__init__()
        # self.data_root = args.data_root
        self.data_root = data_root
        self.partition = partition
        # self.data_aug = args.data_aug
        self.data_aug = data_aug
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std = [0.2675, 0.2565, 0.2761]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        if self.pretrain:
            self.file_pattern = '%s.pickle'
        else:
            self.file_pattern = '%s.pickle'
        self.data = {}

        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            labels = data['labels']
            # adjust sparse labels to labels from 0 to n.
            cur_class = 0
            label2label = {}
            for idx, label in enumerate(labels):
                if label not in label2label:
                    label2label[label] = cur_class
                    cur_class += 1
            new_labels = []
            for idx, label in enumerate(labels):
                new_labels.append(label2label[label])
            self.labels = new_labels

        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)

        if not self.is_sample:
            return img, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx

    def __len__(self):
        return len(self.labels)


def get_rfs_union_sl_dataloader_cifarfs(args: Namespace,
                                        data_path: Path,
                                        batch_size: int = 128,
                                        batch_size_eval: int = 64,
                                        augment_train: bool = True,
                                        augment_val: bool = False,
                                        num_workers: int = -1,
                                        pin_memory: bool = False,
                                        rank: int = -1,
                                        world_size: int = 1,
                                        merge: Optional[Callable] = None,
                                        ) -> dict:
    """
    ref:
        - https://github.com/WangYueFt/rfs/blob/master/train_supervised.py
    """
    assert rank == -1 and world_size == 1, 'no DDP yet. Need to change dl but its not needed in (small) sl.'

    # args.num_workers = 2 if args.num_workers is None else args.num_workers
    # args.target_type = 'classification'
    # args.data_aug = True
    data_root: str = str(data_path)

    # -- get SL dataloaders
    train_trans, val_trans = get_transform_rfs(augment_train), get_transform_rfs(augment_val)
    train_loader = DataLoader(CIFAR100(data_root=data_root, data_aug=augment_train, partition='train',
                                       transform=train_trans),
                              batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
    val_loader = DataLoader(CIFAR100(data_root=data_path, data_aug=augment_val, partition='val',
                                     transform=val_trans),
                            batch_size=batch_size_eval, shuffle=True, drop_last=False,
                            num_workers=num_workers)
    # test_loader = None  # note: since we are evaluating with meta-learning not SL it doesn't need to have this
    test_trans = get_transform_rfs(False)
    test_loader = DataLoader(CIFAR100(data_root=data_path, data_aug=test_trans, partition='test',
                                      transform=val_trans),
                             batch_size=batch_size_eval, shuffle=True, drop_last=False,
                             num_workers=num_workers)

    # -- get meta-dataloaders
    # not needed, we will not evaluate while running the model the meta-test error, that is done seperately.

    # - for now since torchmeta always uses the meta-train or meta-val (but not both together) we won't allow the merge
    args.n_cls = 64

    # - return data loaders
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders


# def get_rfs_union_sl_dataloader_fc100(args: Namespace) -> dict:
#     n_cls = 60


# -

def cifarfs_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        root='~/data',
        data_augmentation=None,
        device=None,
        **kwargs,
):
    import torchvision as tv
    import learn2learn as l2l

    from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels

    from torchvision.transforms import (Compose, ToPILImage, ToTensor, RandomCrop, RandomHorizontalFlip,
                                        ColorJitter, Normalize)
    """Tasksets for CIFAR-FS benchmarks."""
    if data_augmentation is None:
        train_data_transforms = tv.transforms.ToTensor()
        test_data_transforms = tv.transforms.ToTensor()
    elif data_augmentation == 'normalize':
        train_data_transforms = Compose([
            lambda x: x / 255.0,
        ])
        test_data_transforms = train_data_transforms
    elif data_augmentation == 'rfs2020':
        train_data_transforms: list[Callable] = get_transform(augment=True)
        test_data_transforms: list[Callable] = get_transform(augment=False)
    else:
        raise ('Invalid data_augmentation argument.')

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


def fc100_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        root='~/data',
        data_augmentation=None,
        device=None,
        **kwargs,
):
    import torchvision as tv
    import learn2learn as l2l

    from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels

    from torchvision.transforms import (Compose, ToPILImage, ToTensor, RandomCrop, RandomHorizontalFlip,
                                        ColorJitter, Normalize)
    """Tasksets for FC100 benchmarks."""
    if data_augmentation is None:
        train_data_transforms = tv.transforms.ToTensor()
        test_data_transforms = tv.transforms.ToTensor()
    elif data_augmentation == 'normalize':
        train_data_transforms = Compose([
            lambda x: x / 255.0,
        ])
        test_data_transforms = train_data_transforms
    elif data_augmentation == 'rfs2020':
        train_data_transforms = get_transform(True)
        test_data_transforms = get_transform(False)
    else:
        raise ('Invalid data_augmentation argument.')

    train_dataset = l2l.vision.datasets.FC100(root=root,
                                              transform=train_data_transforms,
                                              mode='train',
                                              download=True)
    valid_dataset = l2l.vision.datasets.FC100(root=root,
                                              transform=train_data_transforms,
                                              mode='validation',
                                              download=True)
    test_dataset = l2l.vision.datasets.FC100(root=root,
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


_TASKSETS = {
    # 'omniglot': omniglot_tasksets,
    # 'mini-imagenet': mini_imagenet_tasksets,
    # 'tiered-imagenet': tiered_imagenet_tasksets,
    'fc100': fc100_tasksets,
    'cifarfs': cifarfs_tasksets,
}


def get_tasksets(
        name,
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,
        root='~/data',
        data_augmentation=None,
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
    import learn2learn as l2l

    from learn2learn.vision.benchmarks import BenchmarkTasksets
    # - unchanged l2l code, what I changed is what _TASKSETS has
    root = os.path.expanduser(root)

    # Load task-specific data and transforms
    datasets, transforms = _TASKSETS[name](train_ways=train_ways,
                                           train_samples=train_samples,
                                           test_ways=test_ways,
                                           test_samples=test_samples,
                                           root=root,
                                           data_augmentation=data_augmentation,
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


# - get SL from l2l

def get_sl_l2l_datasets(root,
                        data_augmentation: str = 'rfs2020',
                        device=None
                        ) -> tuple:
    if data_augmentation is None:
        train_data_transforms = transforms.ToTensor()
        test_data_transforms = transforms.ToTensor()
        raise ValueError('only rfs2020 augmentation allowed')
    elif data_augmentation == 'normalize':
        train_data_transforms = Compose([
            lambda x: x / 255.0,
        ])
        test_data_transforms = train_data_transforms
        raise ValueError('only rfs2020 augmentation allowed')
    elif data_augmentation == 'rfs2020':
        train_data_transforms = get_transform(True)
        test_data_transforms = get_transform(False)
    else:
        raise ('Invalid data_augmentation argument.')

    import learn2learn
    train_dataset = learn2learn.vision.datasets.CIFARFS(root=root,
                                                        transform=train_data_transforms,
                                                        mode='train',
                                                        download=True)
    valid_dataset = learn2learn.vision.datasets.CIFARFS(root=root,
                                                        transform=train_data_transforms,
                                                        mode='validation',
                                                        download=True)
    test_dataset = learn2learn.vision.datasets.CIFARFS(root=root,
                                                       transform=test_data_transforms,
                                                       mode='test',
                                                       download=True)
    if device is not None:
        train_dataset = learn2learn.data.OnDeviceDataset(
            dataset=train_dataset,
            device=device,
        )
        valid_dataset = learn2learn.data.OnDeviceDataset(
            dataset=valid_dataset,
            device=device,
        )
        test_dataset = learn2learn.data.OnDeviceDataset(
            dataset=test_dataset,
            device=device,
        )
    return train_dataset, valid_dataset, test_dataset


def get_sl_l2l_cifarfs_dataloaders(args: Namespace) -> dict:
    train_dataset, valid_dataset, test_dataset = get_sl_l2l_datasets(root=args.data_path)

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

    args.n_cls = 64
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders


# - tests

def l2l_sl_dl():
    print('starting...')
    args = Namespace(data_path='~/data/l2l_data/', batch_size=8, batch_size_eval=2, rank=-1, world_size=1)
    args.data_path = Path('~/data/l2l_data/').expanduser()
    dataloaders = get_sl_l2l_cifarfs_dataloaders(args)
    max_val = torch.tensor(-1)
    for i, batch in enumerate(dataloaders['train']):
        # print(batch[1])
        # print(batch[0])
        max_val = max(list(batch[1]) + [max_val])
        # print(f'{max_val}')
        # if 63 in batch[1]:
        #     break
    print(f'--> TRAIN FINAL: {max_val=}')
    assert max_val == len(dataloaders['train'].dataset)

    max_val = torch.tensor(-1)
    for i, batch in enumerate(dataloaders['val']):
        # print(batch[1])
        max_val = max(list(batch[1]) + [max_val])
        # print(f'{max_val}')
        # if 15 in batch[1]:
        #     break
    print(f'--> VAL FINAL: {max_val=}')
    assert max_val == len(dataloaders['val'].dataset)

    max_val = torch.tensor(-1)
    for i, batch in enumerate(dataloaders['test']):
        # print(batch[1])
        max_val = max(list(batch[1]) + [max_val])
        # print(f'{max_val}')
        # if 19 in batch[1]:
        #     break
    print(f'--> TEST FINAL: {max_val=}')

    assert max_val == len(dataloaders['test'].dataset)


def rfs_test():
    args = Namespace()
    # args = lambda x: None
    # args.n_ways = 5
    # args.n_shots = 1
    # args.n_queries = 12
    # args.data_root = 'data'
    args.data_root = Path('~/data/CIFAR-FS/').expanduser()
    args.data_aug = True
    # args.n_test_runs = 5
    # args.n_aug_support_samples = 1
    imagenet = CIFAR100(args.data_root, args.data_aug, 'train')
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)

    # metaimagenet = MetaCIFAR100(args, 'train')
    # print(len(metaimagenet))
    # print(metaimagenet.__getitem__(500)[0].size())
    # print(metaimagenet.__getitem__(500)[1].shape)
    # print(metaimagenet.__getitem__(500)[2].size())
    # print(metaimagenet.__getitem__(500)[3].shape)


if __name__ == '__main__':
    l2l_sl_dl()
    print('Done!\a\n')
