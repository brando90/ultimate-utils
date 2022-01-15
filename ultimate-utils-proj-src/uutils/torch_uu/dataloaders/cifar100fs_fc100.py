"""
refs:
    - inspired from: https://github.com/WangYueFt/rfs/blob/master/train_supervised.py
    - normalization: https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/dataset/transform_cfg.py#L104
"""
from argparse import Namespace
from pathlib import Path
from typing import Optional, Callable

from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

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
                    'path_to_data_set': args.path_to_data_set,
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
                                        path_to_data_set: Path,
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
    data_root: str = str(path_to_data_set)

    # -- get SL dataloaders
    train_trans, val_trans = get_transform(augment_train), get_transform(augment_val)
    train_loader = DataLoader(CIFAR100(data_root=data_root, data_aug=augment_train, partition='train',
                                       transform=train_trans),
                              batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
    val_loader = DataLoader(CIFAR100(data_root=path_to_data_set, data_aug=augment_val, partition='val',
                                     transform=val_trans),
                            batch_size=batch_size_eval, shuffle=True, drop_last=False,
                            num_workers=num_workers)
    test_loader = None  # note: since we are evaluating with meta-learning not SL it doesn't need to have this

    # -- get meta-dataloaders
    # not needed, we will not evaluate while running the model the meta-test error, that is done seperately.

    # - for now since torchmeta always uses the meta-train or meta-val (but not both together) we won't allow the merge
    args.n_cls = 64

    # - return data loaders
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders


def get_rfs_union_sl_dataloader_fc100(args: Namespace) -> dict:
    n_cls = 60


if __name__ == '__main__':
    args = lambda x: None
    # args.n_ways = 5
    # args.n_shots = 1
    # args.n_queries = 12
    # args.data_root = 'data'
    args.data_root = Path('~/data/CIFAR-FS/').expanduser()
    args.data_aug = True
    # args.n_test_runs = 5
    # args.n_aug_support_samples = 1
    imagenet = CIFAR100(args, 'train')
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)

    # metaimagenet = MetaCIFAR100(args, 'train')
    # print(len(metaimagenet))
    # print(metaimagenet.__getitem__(500)[0].size())
    # print(metaimagenet.__getitem__(500)[1].shape)
    # print(metaimagenet.__getitem__(500)[2].size())
    # print(metaimagenet.__getitem__(500)[3].shape)
