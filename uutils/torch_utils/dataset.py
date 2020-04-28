from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

__all__ = ["get_dataset"]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_DICT = {}


def get_dataset(
    dataset, train_path, val_path, batch_size, num_workers, download=False
):
    # TODO Change to accommodate for distributed training
    """
    Fetches data loaders for training and validation

    Args:
        dataset: name of the dataset => ["CIFAR10", "CIFAR100", "ImageNet"]
        train_path: path to training data
        val_path: path to validation data
        batch_size: batch size for data loaders
        num_workers: num of workers for data loaders
        download: downloads the dataset from the internet if True

    Returns:
        A tuple of two separate data loaders for training and validation.
    """
    dataset, (train_transform, valid_transform) = DATASET_DICT[dataset]
    train_dataset = dataset(train_path, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=None,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
    )
    val_dataset = dataset(val_path, transform=valid_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=None,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
    )

    return train_loader, val_loader


class DatasetDecorator:
    """
    Simple decorator class that registers training and validation transforms
    specific for each dataset. Implemented so that the actual dictionary that
    contains all the transforms functions and the dataset classes can be
    exposed in the top of the codebase.

    Args:
        name: name of the dataset
        dataset_f: dataset function from torchvision.datasets
    """

    def __init__(self, name, dataset_f):
        self.name = name
        self.dataset_f = dataset_f

    def __call__(self, transforms_f):
        DATASET_DICT[self.name] = (self.dataset_f, transforms_f())


@DatasetDecorator("CIFAR10", datasets.CIFAR10)
def CIFAR10_transform():
    """
    transforms function for CIFAR10

    Returns:
        A tuple of two transforms.Compose classes for training and validation.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )

    return train_transform, valid_transform


@DatasetDecorator("CIFAR100", datasets.CIFAR10)
def CIFAR100_transform():
    """
    transforms function for CIFAR100

    Returns:
        A tuple of two transforms.Compose classes for training and validation.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    valid_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    return train_transform, valid_transform


@DatasetDecorator("ImageNet", datasets.ImageNet)
def ImageNet_transform():
    """
    transforms function for ImageNet

    Returns:
        A tuple of two transforms.Compose classes for training and validation.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, valid_transform
