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

import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from uutils.torch_uu.dataloaders.cifar100 import get_test_loader_for_cifar100


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

def check_cifar100_is_100_in_usl():
    """not worth debuging my cifar100 code."""
    from uutils.torch_uu.dataloaders.cifar100 import get_train_valid_loader_for_cirfar100
    # import learn2learn as l2l
    # root = Path("~/data/").expanduser()
    # train = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    # train = l2l.data.MetaDataset(train)
    # valid = torchvision.datasets.CIFAR100(root="/tmp/mnist", mode="validation")
    # valid = l2l.data.MetaDataset(valid)
    # test = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    # test = l2l.data.MetaDataset(test)
    # union = UnionMetaDataset([train, valid, test])
    # train_loader, val_loader = get_train_valid_loader_for_cirfar100(root)
    # test_loader: DataLoader = get_test_loader_for_cifar100(root)
    # train, valid, test = train_loader.dataset, val_loader.dataset, test_loader.dataset
    # union = USLDataset([train, valid, test])
    # assert len(union.labels) == 100
    # union_loader = DataLoader(union)
    # next(iter(union_loader))
    pass


def check_mi_omniglot_in_usl():
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_and_omniglot_list_data_set_splits
    dataset_list_train, dataset_list_validation, dataset_list_test = get_mi_and_omniglot_list_data_set_splits()
    # assert isinstance(dataset, Dataset), f'Expect dataset to be of type Dataset but got {type(dataset)=}.'
    assert_dataset_is_pytorch_dataset(dataset_list_train)
    assert isinstance(dataset_list_train[0].dataset, Dataset)
    train_dataset = USLDataset(dataset_list_train)
    valid_dataset = USLDataset(dataset_list_validation)
    test_dataset = USLDataset(dataset_list_test)
    assert len(train_dataset.labels) == 64 + 1100, f'mi + omnigloat should be number of labels 1164.'
    assert len(valid_dataset.labels) == 16 + 100, f'mi + omnigloat should be number of labels 116.'
    assert len(test_dataset.labels) == 20 + 423, f'mi + omnigloat should be number of labels 443.'
    # next(iter(union_loader))


def check_mi_usl():
    """concat data sets should have 100 labels. """
    # - loop through mnist (normal pytorch data set, sanity check, checking api)
    # loop_through_mnist()

    # - get mi data set
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_datasets
    train_dataset, validation_dataset, test_dataset = get_mi_datasets()
    assert_dataset_is_pytorch_dataset([train_dataset, validation_dataset, test_dataset])

    # - create usl data set
    from learn2learn.data import UnionMetaDataset
    union = USLDataset([train_dataset, validation_dataset, test_dataset])
    # from learn2learn.data import OnDeviceDataset
    # union = OnDeviceDataset(union)
    assert_dataset_is_pytorch_dataset([union])
    assert len(union) == 100 * 600, f'got {len(union)=}'
    assert len(union.labels) == 100, f'got {len(union.labels)=}'

    # - create dataloader
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    union_loader, _ = get_serial_or_distributed_dataloaders(train_dataset=union, val_dataset=union)

    # - assert the relabling worked
    relabling_counts: dict = get_relabling_counts(union)
    assert len(relabling_counts.keys()) == 100
    assert_relabling_counts(relabling_counts)
    relabling_counts: dict = check_entire_data_via_the_dataloader(union_loader)
    assert len(relabling_counts.keys()) == 100
    assert_relabling_counts(relabling_counts)


if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    # check_cifar100_is_100_in_usl()
    # check_mi_omniglot_in_usl()
    check_mi_usl()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
