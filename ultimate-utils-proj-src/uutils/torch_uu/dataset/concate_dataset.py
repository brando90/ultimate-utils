"""

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
from typing import Callable, Optional

import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class ConcatDatasetMutuallyExclusiveLabels(Dataset):
    """
    Useful attributes:
        - self.labels: contains all new USL labels i.e. contains the list of labels from 0 - total num labels after concat.
        - len(self): gives number of images after all images have been concatenated
        - self.indices_to_labels: maps the new concat idx to the new label after concat.

    ref:
        - https://stackoverflow.com/questions/73913522/why-dont-the-images-align-when-concatenating-two-data-sets-in-pytorch-using-tor
        - https://discuss.pytorch.org/t/concat-image-datasets-with-different-size-and-number-of-channels/36362/12
    """

    def __init__(self, datasets: list[Dataset],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 compare_imgs_directly: bool = False,
                 ):
        """
        Concatenates different data sets assuming the labels are mutually exclusive in the data sets.

        compare_imgs_directly: adds the additional test that imgs compare at the PIL imgage level.
        """
        self.datasets = datasets
        self.transform = transform
        self.target_transform = target_transform
        # I think concat is better than passing data to a self.data = x obj since concat likely using the getitem method of the passed dataset and thus if the passed dataset doesnt put all the data in memory concat won't either
        self.concat_datasets = torch.utils.data.ConcatDataset(datasets)
        # maps a class label to a list of sample indices with that label.
        self.labels_to_indices = defaultdict(list)
        # maps a sample index to its corresponding class label.
        self.indices_to_labels = defaultdict(None)
        # - do the relabeling
        self._re_label_all_dataset(datasets, compare_imgs_directly)

    def __len__(self):
        return len(self.concat_datasets)

    def _re_label_all_dataset(self, datasets: list[Dataset], compare_imgs_directly: bool = False):
        """
        Relabels according to a blind (mutually exclusive) assumption.

        Relabling Algorithm:
        The zero index of the label starts at the number of labels collected so far. So when relabling we do:
            y =  y + total_number_labels
            total_number_labels += max label for current data set
        where total_number_labels always has the + 1 to correct for the zero indexing.

        :param datasets:
        :param compare_imgs_directly:
        :return:
        """
        self.img2tensor: Callable = torchvision.transforms.ToTensor()
        self.int2tensor: Callable = lambda data: torch.tensor(data, dtype=torch.int)
        total_num_labels_so_far: int = 0
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
                assert y == _y
                if compare_imgs_directly:
                    # from PIL import ImageChops
                    # diff = ImageChops.difference(x, _x)  # https://stackoverflow.com/questions/35176639/compare-images-python-pil
                    # assert diff.getbbox(), f'comparison of imgs failed: {diff.getbbox()=}' # doesn't work :/
                    assert list(x.getdata()) == list(_x.getdata()), f'\n{list(x.getdata())=}, \n{list(_x.getdata())=}'
                    # tensor comparison
                if not isinstance(x, Tensor):
                    x, _x = self.img2tensor(x), self.img2tensor(_x)
                if isinstance(y, int):
                    y, _y = self.int2tensor(y), self.int2tensor(_y)
                assert torch.equal(x,
                                   _x), f'Error for some reason, got: {new_idx=}, {data_idx=}, {x.norm()=}, {_x.norm()=}, {x=}, {_x=}'
                # - relabling
                new_label = y + total_num_labels_so_far
                self.indices_to_labels[new_idx] = new_label
                self.labels_to_indices[new_label].append(new_idx)
                new_idx += 1
            num_labels_for_current_dataset: int = int(max([y for _, y in dataset])) + 1
            # - you'd likely resolve unions if you wanted a proper union, the addition assumes mutual exclusivity
            total_num_labels_so_far += num_labels_for_current_dataset
        assert len(self.indices_to_labels.keys()) == len(self.concat_datasets)
        # contains the list of labels from 0 - total num labels after concat, assume mutually exclusive
        self.labels = range(total_num_labels_so_far)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Get's the data point and it's new label according to a mutually exclusive concatenation.

        For later?
        to do the relabling on the fly we'd need to figure out which data set idx corresponds to and to compute the
        total_num_labels_so_far. Something like this:
            current_data_set_idx = bisect_left(idx)
            total_num_labels_so_far = sum(max(_, y in dataset)+1 for dataset_idx, dataset in enumerate(self.datasets) if dataset_idx <= current_data_set_idx)
            new_y = total_num_labels_so_far
            self.indices_to_labels[idx] = new_y
        :param idx:
        :return:
        """
        x, _y = self.concat_datasets[idx]
        y = self.indices_to_labels[idx]
        # for the first data set they aren't re-labaled so can't use assert
        # assert y != _y, f'concat dataset returns x, y so the y is not relabeled, but why are they the same {_y}, {y=}'
        # idk what this is but could be useful? mnist had this.
        # img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            x = self.transform(x)
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

def check_xs_align_mnist():
    root = Path('~/data/').expanduser()
    import torchvision
    # - test 1, imgs (not the recommended use)
    train = torchvision.datasets.MNIST(root=root, train=True, download=True)
    test = torchvision.datasets.MNIST(root=root, train=False, download=True)
    concat = ConcatDatasetMutuallyExclusiveLabels([train, test], compare_imgs_directly=True)
    print(f'{len(concat)=}')
    print(f'{len(concat.labels)=}')
    # - test 2, tensor imgs
    train = torchvision.datasets.MNIST(root=root, train=True, download=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       target_transform=lambda data: torch.tensor(data, dtype=torch.int))
    test = torchvision.datasets.MNIST(root=root, train=False, download=True,
                                      transform=torchvision.transforms.ToTensor(),
                                      target_transform=lambda data: torch.tensor(data, dtype=torch.int))
    concat = ConcatDatasetMutuallyExclusiveLabels([train, test])
    print(f'{len(concat)=}')
    print(f'{len(concat.labels)=}')
    assert len(concat) == 10 * 7000, f'Err, unexpected number of datapoints {len(concat)=} expected {100 * 700}'
    assert len(
        concat.labels) == 20, f'Note it should be 20 (since it is not a true union), but got {len(concat.labels)=}'

    # - test dataloader
    loader = DataLoader(concat)
    for batch in loader:
        x, y = batch
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)


def check_xs_align_cifar100():
    from pathlib import Path
    root = Path('~/data/').expanduser()
    import torchvision
    # - test 1, imgs (not the recommended use)
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    concat = ConcatDatasetMutuallyExclusiveLabels([train, test], compare_imgs_directly=True)
    print(f'{len(concat)=}')
    print(f'{len(concat.labels)=}')
    # - test 2, tensor imgs
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          target_transform=lambda data: torch.tensor(data, dtype=torch.int))
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=True,
                                         transform=torchvision.transforms.ToTensor(),
                                         target_transform=lambda data: torch.tensor(data, dtype=torch.int))
    concat = ConcatDatasetMutuallyExclusiveLabels([train, test])
    print(f'{len(concat)=}')
    print(f'{len(concat.labels)=}')
    assert len(concat) == 100 * 600, f'Err, unexpected number of datapoints {len(concat)=} expected {100 * 600}'
    assert len(
        concat.labels) == 200, f'Note it should be 200 (since it is not a true union), but got {len(concat.labels)=}'
    # details on cifar100: https://www.cs.toronto.edu/~kriz/cifar.html

    # - test dataloader
    loader = DataLoader(concat)
    for batch in loader:
        x, y = batch
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)


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
    union = ConcatDatasetMutuallyExclusiveLabels([train_dataset, validation_dataset, test_dataset])
    assert_dataset_is_pytorch_dataset([union])
    assert len(union) == 100 * 600, f'got {len(union)=}'
    assert len(union.labels) == 100, f'got {len(union.labels)=}'

    # - create dataloader
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    union_loader, _ = get_serial_or_distributed_dataloaders(train_dataset=union, val_dataset=union)
    for batch in union_loader:
        x, y = batch
        assert x is not None
        assert y is not None


if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    check_xs_align_mnist()
    check_xs_align_cifar100()
    concat_data_set_mi()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
