from argparse import Namespace

import time
from pathlib import Path

import uutils.torch_uu
from meta_learning.datasets.mini_imagenet import MetaImageNet, ImageNet

import torch
from meta_learning.datasets.rand_fnn import RandFNN
from torch import nn, nn as nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torchmeta.toy.helpers import sinusoid
from torchmeta.transforms import ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision import transforms as transforms

import urllib.request

from pathlib import Path



def process_batch_sl(args, batch):
    batch_x, batch_y = batch
    if torch.cuda.is_available():
        if args.device:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
        else:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
    return batch_x, batch_y

def get_torchmeta_meta_data_batch(args: Namespace) -> tuple[Tensor, Tensor]:
    """
    Conceptually, return [B, n_c*k, C, H, W] tensor with B classification tasks, with n_c classes and
    k examples for each class. C, H, W are the usual image dims values for images.

    return: qry_x [B, n_c*k, C, H, W], qry_y [B, ...]
    """
    from uutils.torch_uu import process_meta_batch
    meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_miniimagenet_dataloaders_torchmeta(args)
    batch: dict = next(iter(meta_val_dataloader))
    spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
    return qry_x, qry_y

def get_torchmeta_meta_data_images(args: Namespace, torchmeta_dataloader) -> Tensor:
    """
    Conceptually, return [B, n_c*k, C, H, W] tensor with B classification tasks, with n_c classes and
    k examples for each class. C, H, W are the usual image dims values for images.

    :param torchmeta_dataloader:
    :return: qry_x [B, n_c*k, C, H, W]
    """
    from uutils.torch_uu import process_meta_batch
    batch: dict = next(iter(torchmeta_dataloader))
    spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
    return qry_x

def get_rfs_sl_dataloader(args):
    args.num_workers = 2 if args.num_workers is None else args.num_workers
    args.target_type = 'classification'
    args.trainin_with_epochs = False
    args.data_root = Path('~/data/miniImageNet_rfs/miniImageNet/').expanduser()
    args.data_aug = True
    args.criterion = nn.CrossEntropyLoss()
    # -- get SL dataloaders
    # train_trans, test_trans = transforms_options[opt.transform]
    # train_sl_loader = DataLoader(ImageNet(args=args, partition=args.split),
    #                              batch_size=args.batch_size, shuffle=True, drop_last=True,
    #                              num_workers=args.num_workers)
    train_sl_loader = DataLoader(ImageNet(args=args, partition='train'),
                                 batch_size=args.batch_size, shuffle=True, drop_last=True,
                                 num_workers=args.num_workers)
    val_sl_loader = DataLoader(ImageNet(args=args, partition='val'),
                               batch_size=args.batch_size // 2, shuffle=False, drop_last=False,
                               num_workers=args.num_workers // 2)
    # -- get meta-dataloaders
    args.n_aug_support_samples = 5  # default from rfs
    # meta_testloader = DataLoader(MetaImageNet(args=args, partition='test'),
    #                              batch_size=args.test_batch_size, shuffle=False, drop_last=False,
    #                              num_workers=args.num_workers)
    meta_valloader = DataLoader(MetaImageNet(args=args, partition='val'),
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)
    # if opt.use_trainval:
    #     n_cls = 80
    # else:
    #     n_cls = 64
    # - note: does not return dict because it's code borrowed from rfs paper
    return train_sl_loader, val_sl_loader, meta_valloader

def get_miniimagenet_datasets_torchmeta(args: Namespace) -> dict:
    from torchmeta.datasets.helpers import miniimagenet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_augmentation_transforms = transforms.Compose([
        transforms.RandomResizedCrop(84),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize])
    print('here')
    dataset_train = miniimagenet(args.data_path,
                                 transform=data_augmentation_transforms,
                                 ways=args.n_classes, shots=args.k_shots, test_shots=args.k_eval,
                                 meta_split='train', download=True)
    print('got train')
    dataset_val = miniimagenet(args.data_path, ways=args.n_classes, shots=args.k_shots, test_shots=args.k_eval,
                               meta_split='val', download=True)
    dataset_test = miniimagenet(args.data_path, ways=args.n_classes, shots=args.k_shots, test_shots=args.k_eval,
                                meta_split='test', download=True)
    # - return data sets
    datasets = {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}
    return datasets

def get_miniimagenet_dataloaders_torchmeta(args: Namespace) -> dict:
    args.trainin_with_epochs = False
    args.data_path = Path('~/data/').expanduser()  # for some datasets this is enough
    args.criterion = nn.CrossEntropyLoss()
    # args.image_size = 84  # do we need this?
    # - get data sets
    datasets = get_miniimagenet_datasets_torchmeta(args)
    # - get dataloaders
    meta_train_dataloader = BatchMetaDataLoader(datasets['train'],
                                                batch_size=args.meta_batch_size_train,
                                                num_workers=args.num_workers)
    meta_val_dataloader = BatchMetaDataLoader(datasets['val'],
                                              batch_size=args.meta_batch_size_eval,
                                              num_workers=args.num_workers)
    meta_test_dataloader = BatchMetaDataLoader(datasets['val'],
                                               batch_size=args.meta_batch_size_eval,
                                               num_workers=args.num_workers)
    #- return data laoders
    dataloaders = {'train': meta_train_dataloader, 'val': meta_val_dataloader, 'test': meta_test_dataloader}
    return dataloaders

def get_distributed_dataloader_miniimagenet_torchmeta(args: Namespace) -> dict:
    """
    Get a distributed data laoder for mini-imagenet from torch meta.

    Note:
        - in pytorch dataloaders get distributed samplers and thats how they become distributed.
        - DDP is for models not for data loaders (confusing but accept it :) ).
    :param args:
    :return:
    """
    from uutils.torch_uu.distributed import create_distributed_dataloaders_from_torchmeta_datasets
    print('about to get datasets')
    datasets: dict[str, Dataset] = get_miniimagenet_datasets_torchmeta(args)
    print('got datasets')
    dataloaders: dict[str, DataLoader] = create_distributed_dataloaders_from_torchmeta_datasets(args,
                                                                                                args.rank,
                                                                                                args.world_size,
                                                                                                datasets)
    print('created distributed dataloaders')
    # -- return
    return dataloaders


def get_transforms_mini_imagenet(args):
    # get transforms for images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize])
    val_transform = transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize])
    test_transform = transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize])
    return train_transform, val_transform, test_transform

def get_torchmeta_sinusoid_dataloaders(args):
    # tran = transforms.Compose([torch_uu.tensor])
    # dataset = sinusoid(shots=args.k_eval, test_shots=args.k_shots, transform=tran)
    dataset = sinusoid(shots=args.k_eval, test_shots=args.k_eval)
    train_dataloader = BatchMetaDataLoader(dataset, batch_size=args.meta_batch_size_train,
                                                num_workers=args.num_workers)
    val_dataloader = BatchMetaDataLoader(dataset, batch_size=args.meta_batch_size_eval,
                                              num_workers=args.num_workers)
    test_dataloader = BatchMetaDataLoader(dataset, batch_size=args.meta_batch_size_eval,
                                               num_workers=args.num_workers)
    # - return dataloaders
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataloaders

def get_torchmeta_rand_fnn_dataloaders(args):
    # get data
    dataset_train = RandFNN(args.data_path, 'train')
    dataset_val = RandFNN(args.data_path, 'val')
    dataset_test = RandFNN(args.data_path, 'test')
    # get meta-sets
    metaset_train = ClassSplitter(dataset_train,
                                  num_train_per_class=args.k_shots,
                                  num_test_per_class=args.k_eval,
                                  shuffle=True)
    metaset_val = ClassSplitter(dataset_val, num_train_per_class=args.k_shots,
                                num_test_per_class=args.k_eval,
                                shuffle=True)
    metaset_test = ClassSplitter(dataset_test, num_train_per_class=args.k_shots,
                                 num_test_per_class=args.k_eval,
                                 shuffle=True)
    # get meta-dataloader
    train_dataloader = BatchMetaDataLoader(metaset_train,
                                                batch_size=args.meta_batch_size_train,
                                                num_workers=args.num_workers)
    val_dataloader = BatchMetaDataLoader(metaset_val,
                                              batch_size=args.meta_batch_size_eval,
                                              num_workers=args.num_workers)
    test_dataloader = BatchMetaDataLoader(metaset_test,
                                               batch_size=args.meta_batch_size_eval,
                                               num_workers=args.num_workers)
    # - return dataloaders
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataloaders

def get_dataloaders(args, rank, world_size, merge, dataset):
    """

    todo - figure out what is the number of workers for a DDP dataloader. 1) check pytorch forum for it
    :param args:
    :param rank:
    :param world_size:
    :param merge:
    :param dataset:
    :return:
    """
    from uutils.torch_uu.distributed import is_running_serially

    train_dataset = dataset(args, split='train')
    val_dataset = dataset(args, split='val')
    test_dataset = dataset(args, split='test')
    if is_running_serially(rank):
        train_sampler, val_sampler, test_sampler  = None, None, None
        # args.num_workers = args.num_workers if hasattr(args, 'num_workers') else 4
        args.num_workers = 4
    else:
        # get dist samplers
        assert (args.batch_size >= world_size)
        from torch.utils.data import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        # todo - figure out what is the best for ddp. But my guess is that 0 is fine as hardcoded value & only overwrite if args.num_wokers has a none -1 or none else use my hardcoded default
        args.num_workers = 0
    # get dist dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=merge,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=val_sampler,
                                collate_fn=merge,
                                num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 sampler=test_sampler,
                                 collate_fn=merge,
                                 num_workers=args.num_workers)
    # - return dataloaders
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataloaders

def get_dataset(dataloaders: dict) -> dict:
    """
    From dictionary of dataloaders to dictionary of datasets
    """
    datasets = {split: dataloader.dataset for split,dataloader in dataloaders}
    return datasets

def _get_minimum_args_for_mini_imagenet_from_torchmeta(args: Namespace) -> Namespace:
    # note this is hardcoded in get_miniimagenet_dataloaders_torchmeta
    args.data_path = Path('~/data/').expanduser()  # for some datasets this is enough
    args.n_classes = 5
    args.k_shots = 5
    args.k_eval = 15
    args.meta_batch_size_train = 2
    args.meta_batch_size_eval = 2
    args.num_workers = 0
    return args

def load_all_parmas_for_torchmeta_mini_imagenet_dataloader_into_args(args: Namespace, k_eval: int = 15,
                                              k_shots: int = 5, n_classes: int = 5, meta_batch_size: int = 2,
                                              num_workers: int = 0) -> Namespace:
    """
    """
    args.data_path = Path('~/data/').expanduser()  # for some datasets this is enough
    args.n_classes = n_classes
    args.k_shots = k_shots
    args.k_eval = k_eval
    args.meta_batch_size_train = meta_batch_size
    args.meta_batch_size_eval = meta_batch_size
    args.num_workers = num_workers
    return args

def get_minimum_args_for_torchmeta_mini_imagenet_dataloader(data_path: Path = Path('~/data/'),
                                                            k_eval: int = 15, k_shots: int = 5,
                                                            n_classes: int = 5,
                                                            meta_batch_size_train: int = 2,
                                                            meta_batch_size_eval: int = 2,
                                                            num_workers: int = 0) -> Namespace:
    """
    Gets the minimum args for torchmeta mini imagenet dataloader to work.

    Note, you can update default values if you want.
    """
    args: Namespace = Namespace()
    args.data_path = data_path.expanduser()  # for some datasets this is enough
    args.n_classes = n_classes
    args.k_shots = k_shots
    args.k_eval = k_eval
    args.meta_batch_size_train = meta_batch_size_train
    args.meta_batch_size_eval = meta_batch_size_eval
    args.num_workers = num_workers
    args.device = uutils.torch_uu.get_device()
    return args

def get_set_of_examples_from_mini_imagenet(k_eval: int = 15) -> torch.Tensor:
    from uutils.torch_uu import process_meta_batch
    args: Namespace = Namespace()
    args.data_path = Path('~/data/').expanduser()  # for some datasets this is enough
    args.n_classes = 5
    args.k_shots = 5
    args.k_eval = k_eval
    args.meta_batch_size_train = 2
    args.meta_batch_size_eval = 2
    args.num_workers = 0
    args.device = uutils.torch_uu.get_device()
    # args = get_minimum_args_for_mini_imagenet_from_torchmeta(args)
    meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_miniimagenet_dataloaders_torchmeta(args)
    meta_batch: dict = next(iter(meta_val_dataloader))
    spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, meta_batch)
    # - to get the first task (since it will retrun a batch of tasks)
    X: torch.Tensor = qry_x[0].squeeze()  # [1, num_classes * k_eval, C, H, W] -> [num_classes * k_eval, C, H, W]
    return X

def get_default_mini_imagenet():
    from torchmeta.datasets.helpers import miniimagenet
    from torchmeta.utils.data import BatchMetaDataLoader
    dataset = miniimagenet("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
    dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
    return dataloader

# ---- teats ----

def get_args_for_mini_imagenet():
    from types import SimpleNamespace

    args = SimpleNamespace()
    # Config for
    args.mode = "meta-train"
    #args.mode = "meta-test"
    args.k_shot = 5
    args.k_eval = 15
    args.n_classes = 5
    args.grad_clip = None # does no gradient clipping if None
    args.grad_clip_mode = None # more specific setting of the crad clipping mode
    # Episodes params
    args.episodes = 60000
    args.episodes_val = 100
    args.episodes_test = 100
    #args.log_train_freq = 100 if not args.debug else 1
    #args.log_val_freq = 10 if not args.debug else 1
    # careful to have these larger than the size of the meta-set
    args.meta_batch_size_train = 25
    args.meta_batch_size_eval = 4
    # Inner loop adaptation params
    args.nb_inner_train_steps = 10
    args.track_higher_grads = True # set to false only during meta-testing, but code sets it automatically only for meta-test
    args.copy_initial_weights = False # set to false only if you do not want to train base model's initialization
    # MAML
    # args.fo = False
    # args.inner_lr = 1e-1
    # args.meta_learner = 'maml_fixed_inner_lr'
    # Learner/base model options
    args.bn_momentum = 0.95
    args.bn_eps = 1e-3
    args.base_model_mode = 'child_mdl_from_opt_as_a_mdl_for_few_shot_learning_paper'
    # miniImagenet options
    args.data_root = Path("~/automl-meta-learning/data/miniImagenet").expanduser()
    args.n_workers = 4
    args.pin_memory = False # it is generally not recommended to return CUDA tensors in multi-process loading because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing (see CUDA in multiprocessing). Instead, we recommend using automatic memory pinning (i.e., setting pin_memory=True), which enables fast data transfer to CUDA-enabled GPUs. https://pytorch.org/docs/stable/data.html
    args.criterion = nn.CrossEntropyLoss()
    args.image_size = 84
    return args

def mini_imagenet_loop():
    from uutils.torch_uu import process_meta_batch

    args = get_minimum_args_for_torchmeta_mini_imagenet_dataloader()
    dataloader = get_miniimagenet_dataloaders_torchmeta(args)

    print(f'{len(dataloader)}')
    for batch_idx, batch in enumerate(dataloader['train']):
        print(f'{batch_idx=}')
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
        print(f'Train inputs shape: {spt_x.size()}')  # (2, 25, 3, 28, 28)
        print(f'Train targets shape: {spt_y.size()}'.format(spt_y.shape))  # (2, 25)

        print(f'Test inputs shape: {qry_x.size()}')  # (2, 75, 3, 28, 28)
        print(f'Test targets shape: {qry_y.size()}')  # (2, 75)
        break


if __name__ == '__main__':
    from uutils import report_times
    start = time.time()
    mini_imagenet_loop()
    print(f'Time passed: {report_times(start)}')