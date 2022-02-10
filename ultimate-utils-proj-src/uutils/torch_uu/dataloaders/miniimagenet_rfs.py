"""
Adapted from: https://github.com/WangYueFt/rfs
"""
import os
import pickle
from argparse import Namespace
from typing import Optional, Callable

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from pathlib import Path

# - mini-imagenet normalization according to rfs
mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
normalize = transforms.Normalize(mean=mean, std=std)


def get_train_valid_test_data_loader_miniimagenet_rfs(args: Namespace) -> dict:
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
    dataloaders: dict = get_mini_imagenet_rfs_sl_dataloader(**train_kwargs)
    return dataloaders


# todo - fix later? did this padding=8 make a difference?
# def get_transform_rfs():
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     data_augmentation_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(84),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(
#             brightness=0.4,
#             contrast=0.4,
#             saturation=0.4,
#             hue=0.2),
#         transforms.ToTensor(),
#         normalize])
#     return data_augmentation_transforms

def get_transform(augment: bool):
    if augment:
        transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.ToTensor(),
            normalize
        ])
    return transform


def get_mini_imagenet_rfs_sl_dataloader(args: Namespace,
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
    assert rank == -1 and world_size == 1, 'no DDP yet. Need to change dl but its not needed in (small) sl.'

    # args.num_workers = 2 if args.num_workers is None else args.num_workers
    # args.target_type = 'classification'
    # args.data_aug = True
    data_root: str = str(path_to_data_set)

    # -- get SL dataloaders
    train_trans, val_trans = get_transform(augment_train), get_transform(augment_val)
    train_loader = DataLoader(ImageNet(data_root=data_root, data_aug=augment_train, partition='train',
                                       transform=train_trans),
                              batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
    val_loader = DataLoader(ImageNet(data_root=path_to_data_set, data_aug=augment_val, partition='val',
                                     transform=val_trans),
                            batch_size=batch_size_eval, shuffle=True, drop_last=False,
                            num_workers=num_workers)
    # test_loader = None  # note: since we are evaluating with meta-learning not SL it doesn't need to have this
    test_trans = get_transform(augment=False)
    test_loader = DataLoader(ImageNet(data_root=path_to_data_set, data_aug=test_trans, partition='test',
                                      transform=val_trans),
                             batch_size=batch_size_eval, shuffle=True, drop_last=False,
                             num_workers=num_workers)

    # if opt.use_trainval:
    #     n_cls = 80
    # else:
    #     n_cls = 64
    args.n_cls = 64
    dataloaders: dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders


class ImageNet(Dataset):
    def __init__(self, data_root, data_aug,
                 partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.partition = partition
        self.data_aug = data_aug
        """
        mean
        [0.47214064400000005, 0.45330829125490196, 0.4099612805098039]
        std
        [0.2771838538039216, 0.26775040952941176, 0.28449041290196075]
        """
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    Image.fromarray,
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    np.asarray,
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    Image.fromarray,
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        if self.pretrain:
            self.file_pattern = 'miniImageNet_category_split_train_phase_%s.pickle'
        else:
            self.file_pattern = 'miniImageNet_category_split_%s.pickle'
        self.data = {}
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            self.labels = data['labels']

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
        # target = self.labels[item] - min(self.labels)
        target = self.labels[item]

        if not self.is_sample:
            return img, target, item
        else:
            # pos_idx = item
            # replace = True if self.k > len(self.cls_negative[target]) else False
            # neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            # sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            # return img, target, item, sample_idx
            raise ValueError('NCE not implemented yet')

    def __len__(self):
        return len(self.labels)


class MetaImageNet(ImageNet):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaImageNet, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_classes = args.n_classes
        self.k_shots = args.k_shots
        self.k_eval = args.k_eval
        self.classes = list(self.data.keys())
        self.n_test_runs = args.eval_iters
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose([
                Image.fromarray,
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                np.asarray,
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                Image.fromarray,
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_classes, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.k_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.k_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.k_eval, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, k_eval_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * k_eval_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * k_eval_per_way,))

        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs


if __name__ == '__main__':
    args = lambda x: None
    args.n_classes = 5
    args.k_shots = 1
    args.k_eval = 12
    args.data_root = Path('~/data/miniImageNet_rfs/miniImageNet/').expanduser()
    args.data_aug = True
    args.n_test_runs = 5  # same as eval_iters
    args.n_aug_support_samples = 1
    print('SL imagenet unit test')
    for split in ['train', 'val', 'test']:
        print(split)
        imagenet = ImageNet(args, split)
        print(len(imagenet))
        print(imagenet.__getitem__(500)[0].shape)
    print('done with imagenet')

    # for split in ['train', 'val', 'test']:
    for split in ['val', 'test']:
        metaimagenet = MetaImageNet(args, split)
        print(len(metaimagenet))
        print(metaimagenet.__getitem__(500)[0].size())
        print(metaimagenet.__getitem__(500)[1].shape)
        print(metaimagenet.__getitem__(500)[2].size())
        print(metaimagenet.__getitem__(500)[3].shape)
    print('done with metaimagenet tests')
