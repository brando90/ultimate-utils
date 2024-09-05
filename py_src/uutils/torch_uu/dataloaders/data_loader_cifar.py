"""
Create train, val, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4

https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""

import torch
import numpy as np

from pathlib import Path

#from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils.utils import get_random_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CIFAR_DATA_PATH = Path('~/predicting_generalization/automl/data/').expanduser()

def get_train_val_loader(data_dir,
                           batch_size,
                           augment,
                           device,
                           random_seed,
                           val_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4):
    '''
    Utility function for loading and returning train and val
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - device: device
    - random_seed: fix seed for reproducibility.
    - val_size: percentage split of the training set used for
      the valation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/valation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.

    pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - val_loader: valation set iterator.
    '''
    pin_memory = False
    if 'cuda' in device.type:
        pin_memory = True

    error_msg = "[!] val_size should be in the range [0, 1]."
    assert ((val_size >= 0) and (val_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    #normalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

    # define transforms
    val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    val_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=val_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        # plot_images(X, labels)

    return (train_loader, val_loader)


def get_test_loader(data_dir,
                    batch_size,
                    device,
                    shuffle=True,
                    num_workers=4):
    '''
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - device: device
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.

    pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    '''
    pin_memory = False
    if 'cuda' in device.type:
        pin_memory = True

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    #normalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def get_cifar10_for_data_point_mdl_gen(batch_size=512, num_workers=4, augment=False, random_seed=None):
    '''
    Gets the data sets for training the models that will be data points
    '''
    #data_dir = data_dir if data_dir is not None else CIFAR_DATA_PATH
    data_dir = str(CIFAR_DATA_PATH)
    #print(f'--> {data_dir}')
    ##
    val_size = 0.1
    random_seed = get_random_seed() if random_seed is None else random_seed
    #st()
    trainloader, valloader = get_train_val_loader(data_dir, batch_size, augment, device, random_seed,
                            val_size=val_size, shuffle=True, show_sample=False, num_workers=num_workers)
    #st()
    ##
    batch_size_test = 1024
    testloader = get_test_loader(data_dir, batch_size_test, device, shuffle=True, num_workers=num_workers)
    return trainloader, valloader, testloader

def get_cifar10_for_automl(batch_size=512, num_workers=4, augment=False, random_seed=None, val_size=0.5):
    """

    Args:
        batch_size:
        num_workers:
        augment:
        random_seed:
        val_size:

    Returns:

    """
    #data_dir = data_dir if data_dir is not None else CIFAR_DATA_PATH
    data_dir = str(CIFAR_DATA_PATH)
    ##
    if random_seed is None:
        random_seed = get_random_seed()
    trainloader, valloader = get_train_val_loader(data_dir, batch_size, augment, device, random_seed,
                            val_size=val_size, shuffle=True, show_sample=False, num_workers=num_workers)
    ##
    batch_size_test = 1024
    testloader = get_test_loader(data_dir, batch_size_test, device, shuffle=True, num_workers=num_workers)
    return trainloader, valloader, testloader

def _dont_get_cifar10(batch_size=256, num_workers=4):
    '''
    DONT USE!!!!!!

    get cifar10 data set
    TODO: make batch, num_workers etc parameters
    '''
    error_msg = "---> ERROR: YOU ARE USING A FORBIDDEN METHOD!!!! STOP! \a"
    print(error_msg)
    raise ValueError(error_msg)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return transform, trainset, trainloader, testset, testloader
