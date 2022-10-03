"""
todo: check the two common interfacses for labels normal l2l uses l2l comment on normal dataset

.indices
.labels

right?

function get_labels() then indices the right one?

mapping algorithm

do checks, loop through all data points, create counts for each label how many data points there are
do this for MI only

then check union and ur implementation?
compare the mappings of one & the other?

actually it's easy, just add the cummulative offset and that's it. :D the indices are already -1 indexed.

assert every image has a label between 0 --> n1+n2+... and every bin for each class is none empty

for it to work with any standard pytorch data set I think the workflow would be:
```
pytorch dataset -> l2l meta data set -> union data set -> .dataset field -> data loader
```
for l2l data sets:
```
l2l meta data set -> union data set -> .dataset field -> data loader
```
but the last one might need to make sure .indices or .labels is created or a get labels function that checking the attribute
gets the right .labels or remaps it correctly
"""
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from uutils.torch_uu.dataloaders.cifar100 import get_test_loader_for_cifar100


class ConcatDataset(Dataset):
    """
    ref:
        - https://stackoverflow.com/questions/73913522/why-dont-the-images-align-when-concatenating-two-data-sets-in-pytorch-using-tor
        - https://discuss.pytorch.org/t/concat-image-datasets-with-different-size-and-number-of-channels/36362/12
    """

    def __init__(self, datasets: list[Dataset]):
        """
        """
        # I think concat is better than passing data to a self.data = x obj since concat likely using the getitem method of the passed dataset and thus if the passed dataset doesnt put all the data in memory concat won't either
        self.concat_datasets = torch.utils.data.ConcatDataset(datasets)
        # maps a class label to a list of sample indices with that label.
        self.labels_to_indices = defaultdict(list)
        # maps a sample index to its corresponding class label.
        self.indices_to_labels = defaultdict(None)
        # - do the relabeling
        img2tensor: Callable = torchvision.transforms.ToTensor()
        offset: int = 0
        new_idx: int = 0
        for dataset_idx, dataset in enumerate(datasets):
            assert len(dataset) == len(self.concat_datasets.datasets[dataset_idx])
            assert dataset == self.concat_datasets.datasets[dataset_idx]
            for data_idx, (x, y) in enumerate(dataset):
                y = int(y)
                # - get data point from concataned data set (to compare with the data point from the data set list)
                _x, _y = self.concat_datasets[new_idx]
                _y = int(_y)
                # - sanity check concatanted data set aligns with the list of datasets
                # assert y == _y
                # from PIL import ImageChops
                # diff = ImageChops.difference(x, _x)  # https://stackoverflow.com/questions/35176639/compare-images-python-pil
                # assert diff.getbbox(), f'comparison of imgs failed: {diff.getbbox()=}'
                # assert list(x.getdata()) == list(_x.getdata()), f'\n{list(x.getdata())=}, \n{list(_x.getdata())=}'
                # tensor comparison
                x, _x = img2tensor(x), img2tensor(_x)
                print(f'{data_idx=}, {x.norm()=}, {_x.norm()=}')
                assert torch.equal(x, _x), f'Error for some reason, got: {data_idx=}, {x.norm()=}, {_x.norm()=}, {x=}, {_x=}'
                # - relabling
                new_label = y + offset
                self.indices_to_labels[new_idx] = new_label
                self.labels_to_indices[new_label] = new_idx
            num_labels_for_current_dataset: int = max([y for _, y in dataset])
            offset += num_labels_for_current_dataset
            new_idx += 1
        assert len(self.indices_to_labels.keys()) == len(self.concat_datasets)
        # contains the list of labels from 0 - total num labels after concat
        self.labels = range(offset)
        self.target_transform = lambda data: torch.tensor(data, dtype=torch.int)

    def __len__(self):
        return len(self.concat_datasets)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.concat_datasets[idx]
        y = self.indices_to_labels[idx]
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

class USLDataset(Dataset):
    """Generates a Union Supervised Learning (basically concat) like Meta-Data set:
        > The non-episodic baselines are trained to solve the
        > large classification problem that results from ‘concatenating’ the training classes of all datasets.
    """

    def __init__(self, datasets: list[Dataset]):
        """
        """
        self.datasets = datasets
        # todo do we really need the l2l ones?
        # maps a class label to a list of sample indices with that label.
        self.labels_to_indices = defaultdict(list)
        # maps a sample index to its corresponding class label.
        self.indices_to_labels = defaultdict(None)
        # - do the relabeling
        self.data = []
        dataset: Dataset
        label_cum_sum: int = 0
        new_idx: int = 0
        for dataset in datasets:
            for x, y in dataset:
                y = int(y)
                # print(f'{(x, y)=}')
                new_label = y + label_cum_sum
                self.indices_to_labels[new_idx] = new_label
                new_idx += 1
                self.labels_to_indices[new_label].append(new_idx)
                # -
                self.data.append(x)
            label_cum_sum += len(dataset.labels)
        assert len(self.data) == sum([len(dataset) for dataset in self.datasets])
        assert len(self.labels) == cum_sum
        print()
        self.labels = list(self.labels_to_indices.keys()).sort()
        # todo: 1. do bisect function to index to union, make sure it works with getitem, might be needed for meta-data set?
        # todo: 2. other option is to do what mnist does:
        # self.data, self.targets = self._load_data()
        del self.datasets
        self.target_transform = lambda data: torch.tensor(data, dtype=torch.int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # todo: do bisect function to index to union, make sure it works with getitem
        x = self.data[idx]
        y = self.indices_to_labels[idx]
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


def assert_dataset_is_pytorch_dataset(datasets: list, verbose: bool = False):
    """ to do 1 data set wrap it in a list"""
    for dataset in datasets:
        if verbose:
            print(f'{type(dataset)=}')
            print(f'{type(dataset.dataset)=}')
        assert isinstance(dataset, Dataset), f'Expect dataset to be of type Dataset but got {type(dataset)=}.'


def get_relabling_counts(dataset: Dataset) -> dict:
    """
    counts[new_label] -> counts/number of data points for that new label
    """
    assert isinstance(dataset, Dataset), f'Expect dataset to be of type Dataset but got {type(dataset)=}.'
    counts: dict = {}
    iter_dataset = iter(dataset)
    for datapoint in iter_dataset:
        x, y = datapoint
        # assert isinstance(x, torch.Tensor)
        # assert isinstance(y, int)
        if y not in counts:
            counts[y] = 0
        else:
            counts[y] += 1
    return counts


def assert_relabling_counts(counts: dict, labels: int = 100, counts_per_label: int = 600):
    """
    default values are for MI.

    - checks each label/class has the right number of expected images per class
    - checks the relabels start from 0 and increase by 1
    - checks the total number of labels after concat is what you expect

    ref: https://openreview.net/pdf?id=rJY0-Kcll
    Because the exact splits used in Vinyals et al. (2016)
    were not released, we create our own version of the Mini-Imagenet dataset by selecting a random
    100 classes from ImageNet and picking 600 examples of each class. We use 64, 16, and 20 classes
    for training, validation and testing, respectively.
    """
    # - check each image has the right number of total images
    seen_labels: list[int] = []
    for label, count in counts.items():
        seen_labels.append(label)
        assert counts[label] == counts_per_label
    # - check all labels are there and total is correct
    seen_labels.sort()
    prev_label = -1
    for label in seen_labels:
        diff = label - prev_label
        assert diff == 1
        assert prev_label < label
    # - checks the final label is the total number of labels
    assert label == labels - 1


def check_entire_data_via_the_dataloader(dataloader: DataLoader) -> dict:
    counts: dict = {}
    for it, batch in enumerate(dataloader):
        xs, ys = batch
        for y in ys:
            if y not in counts:
                counts[y] = 0
            else:
                counts[y] += 1
    return counts


# - tests

def loop_through_mnist():
    root = Path('~/data/').expanduser()
    import torch
    import torchvision
    mnist = torchvision.datasets.MNIST(root=root, download=True, transform=torchvision.transforms.ToTensor())
    # iter(torch.utils.data.DataLoader(mnist)).next()
    for x, y in mnist:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, int)
        pass
    for x, y in iter(mnist):
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, int)
        pass
    assert_dataset_is_pytorch_dataset([mnist])

# def l2l_cirfar100_example_union():
#     import learn2learn as l2l
#     train = torchvision.datasets.CIFARFS(root="/tmp/mnist", mode="train")
#     train = l2l.data.MetaDataset(train)
#     valid = torchvision.datasets.CIFARFS(root="/tmp/mnist", mode="validation")
#     valid = l2l.data.MetaDataset(valid)
#     test = torchvision.datasets.CIFARFS(root="/tmp/mnist", mode="test")
#     test = l2l.data.MetaDataset(test)
#     from learn2learn.data import UnionMetaDataset
#     union = UnionMetaDataset([train, valid, test])
#     assert len(union.labels) == 100

def check_cifar100_is_100_in_usl():
    """not worth debuging my cifar100 code."""
    # https://github.com/learnables/learn2learn/issues/357
    from pathlib import Path
    import learn2learn as l2l

    root = Path("~/data/").expanduser()
    # root = Path(".").expanduser()
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    train = l2l.data.MetaDataset(train)
    print(f'{len(train.labels)=}')
    # valid = torchvision.datasets.CIFAR100(root="/tmp/mnist", mode="validation")
    # valid = l2l.data.MetaDataset(valid)
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    test = l2l.data.MetaDataset(test)
    print(f'{len(test.labels)=}')

    from learn2learn.data import UnionMetaDataset
    # union = UnionMetaDataset([train, valid, test])
    union = UnionMetaDataset([train, test])
    assert isinstance(union, Dataset)
    # assert len(union.labels) == 100, f'Error, got instead: {len(union.labels)=}.'

    # - test looping (to eventually see why data loader of union fails)
    print(f'{union[0]}')
    print(f'{next(iter(union))=}')
    for data_point in union:
        print(f'{data_point=}')
        break
    for x, y in union:
        print(f'{(x, y)=}')
        break
    # -
    usl = ConcatDataset([train, test])
    for d1, d2 in zip(union, usl):
        x1, y1 = d1
        x2, y2 = d2
        print(x1, x2)
        print(y1, y2)
        assert x1 == x2
        assert y1 == y2
    assert len(usl.labels) == len(union.labels)
    assert len(usl) == len(union)

    # from torch.utils.data import DataLoader
    # union_dl: DataLoader = DataLoader(union)
    # for x, y in union_dl:
    #     print(f'{(x, y)=}')
    #     break

    # - get normal pytoch data loaders
    # from uutils.torch_uu.dataloaders.cifar100 import get_train_valid_loader_for_cirfar100
    # train_loader, val_loader = get_train_valid_loader_for_cirfar100(root)
    # test_loader: DataLoader = get_test_loader_for_cifar100(root)
    # train, valid, test = train_loader.dataset, val_loader.dataset, test_loader.dataset
    # union = USLDataset([train, valid, test])
    # union_loader = DataLoader(union)
    # next(iter(union_loader))
    # pass


# def check_mi_omniglot_in_usl():
#     from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_and_omniglot_list_data_set_splits
#     dataset_list_train, dataset_list_validation, dataset_list_test = get_mi_and_omniglot_list_data_set_splits()
#     # assert isinstance(dataset, Dataset), f'Expect dataset to be of type Dataset but got {type(dataset)=}.'
#     assert_dataset_is_pytorch_dataset(dataset_list_train)
#     assert isinstance(dataset_list_train[0].dataset, Dataset)
#     train_dataset = USLDataset(dataset_list_train)
#     valid_dataset = USLDataset(dataset_list_validation)
#     test_dataset = USLDataset(dataset_list_test)
#     assert len(train_dataset.labels) == 64 + 1100, f'mi + omnigloat should be number of labels 1164.'
#     assert len(valid_dataset.labels) == 16 + 100, f'mi + omnigloat should be number of labels 116.'
#     assert len(test_dataset.labels) == 20 + 423, f'mi + omnigloat should be number of labels 443.'
#     # next(iter(union_loader))


# def check_mi_usl():
#     """concat data sets should have 100 labels. """
#     # - loop through mnist (normal pytorch data set, sanity check, checking api)
#     # loop_through_mnist()
#
#     # - get mi data set
#     from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_datasets
#     train_dataset, validation_dataset, test_dataset = get_mi_datasets()
#     assert_dataset_is_pytorch_dataset([train_dataset, validation_dataset, test_dataset])
#
#     # - create usl data set
#     from learn2learn.data import UnionMetaDataset
#     union = USLDataset([train_dataset, validation_dataset, test_dataset])
#     # from learn2learn.data import OnDeviceDataset
#     # union = OnDeviceDataset(union)
#     assert_dataset_is_pytorch_dataset([union])
#     assert len(union) == 100 * 600, f'got {len(union)=}'
#     assert len(union.labels) == 100, f'got {len(union.labels)=}'
#
#     # - create dataloader
#     from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
#     union_loader, _ = get_serial_or_distributed_dataloaders(train_dataset=union, val_dataset=union)
#
#     # - assert the relabling worked
#     relabling_counts: dict = get_relabling_counts(union)
#     assert len(relabling_counts.keys()) == 100
#     assert_relabling_counts(relabling_counts)
#     relabling_counts: dict = check_entire_data_via_the_dataloader(union_loader)
#     assert len(relabling_counts.keys()) == 100
#     assert_relabling_counts(relabling_counts)


def concat_data_set_mi():
    """
    note test had to be in MI where train, val, test have disjount/different labels. In cifar100 classic the labels
    in train, val and test are shared from 0-99 instead of being different/disjoint.
    :return:
    """
    # - get mi data set
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_datasets
    train_dataset, validation_dataset, test_dataset = get_mi_datasets()
    assert_dataset_is_pytorch_dataset([train_dataset, validation_dataset, test_dataset])
    train_dataset, validation_dataset, test_dataset = train_dataset.dataset, validation_dataset.dataset, test_dataset.dataset

    # - create usl data set
    union = ConcatDataset([train_dataset, validation_dataset, test_dataset])
    assert_dataset_is_pytorch_dataset([union])
    assert len(union) == 100 * 600, f'got {len(union)=}'
    assert len(union.labels) == 100, f'got {len(union.labels)=}'

    # # - create dataloader
    # from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    # union_loader, _ = get_serial_or_distributed_dataloaders(train_dataset=union, val_dataset=union)
    #
    # # - assert the relabling worked
    # relabling_counts: dict = get_relabling_counts(union)
    # assert len(relabling_counts.keys()) == 100
    # assert_relabling_counts(relabling_counts)
    # relabling_counts: dict = check_entire_data_via_the_dataloader(union_loader)
    # assert len(relabling_counts.keys()) == 100
    # assert_relabling_counts(relabling_counts)

def check_xs_align_cifar100():
    from pathlib import Path
    import torchvision
    # from typing import Callable

    root = Path("~/data/").expanduser()
    # root = Path(".").expanduser()
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    # train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=torchvision.transforms.ToTensor())
    # test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=torchvision.transforms.ToTensor())

    concat = ConcatDataset([train, test])
    print(f'{len(concat)=}')
    print(f'{len(concat.labels)=}')

if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    # check_cifar100_is_100_in_usl()
    # check_mi_omniglot_in_usl()
    # check_mi_usl()
    # concat_data_set_mi()
    check_xs_align_cifar100()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
