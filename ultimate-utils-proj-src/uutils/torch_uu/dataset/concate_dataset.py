"""
"""
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Callable, Optional, Union

import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from uutils import report_times, get_duplicates, expanduser, pprint_dict


# int2tensor: Callable = lambda data: torch.tensor(data, dtype=torch.int)
# int2tensor: Callable = lambda data: torch.tensor(data, dtype=torch.long)
def int2tensor(data: torch.Tensor) -> torch.long:
    return torch.tensor(data, dtype=torch.long)


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

    def __init__(self,
                 datasets: list[Dataset],
                 root: str,  # suggested value: Path('~/data/l2l_data/').expanduser(),
                 relabel_filename: str,  # e.g. f'hdb4_micod_{split}_relabel_usl.pt'
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 compare_imgs_directly: bool = False,
                 verify_xs_align: bool = False,
                 pickle_usl_relabling: bool = True,
                 ):
        """
        Concatenates different data sets assuming the labels are mutually exclusive in the data sets.
        Thus, even if you have labels of cats on both they are treated differently.

        compare_imgs_directly: adds the additional test that imgs compare at the PIL imgage level.
        """
        self.root = root
        self.pickle_usl_relabling = pickle_usl_relabling
        # more essental inits
        self.datasets = datasets
        self.transform = transform
        self.target_transform = target_transform
        # I think concat is better than passing data to a self.data = x obj since concat likely using the getitem method of the passed dataset and thus if the passed dataset doesnt put all the data in memory concat won't either
        self.concat_datasets = torch.utils.data.ConcatDataset(datasets)
        # maps a class label to a list of sample indices with that label.
        self.labels_to_indices = defaultdict(list)
        # maps a sample index to its corresponding class label.
        self.indices_to_labels = defaultdict(None)
        # other
        self.compare_imgs_directly = compare_imgs_directly
        self.verify_xs_align = verify_xs_align
        self.img2tensor: Callable = torchvision.transforms.ToTensor()
        self.labels = None  # re-labaling not done yet
        # - do the relabeling
        if self.need_to_relabel(root, relabel_filename):
            # load pickle-load from the location of where the data is the array/maps global_idx <-> global_new_labels, since relabling is expensive. I feel my relabling is dumb, but it works and not going back to improve it.
            self._re_label_all_dataset(datasets, compare_imgs_directly, verify_xs_align, pickle_usl_relabling, root,
                                       relabel_filename)
        else:
            self.get_relabeling_data_from_file_(root, relabel_filename)
            # relabling file already exists so no need to save it again
        print(f'Set up of ConcatDatasetMutuallyExclusiveLabels done. {self.__repr__()}, {self.__len__()}, {self}')

    def __len__(self):
        return len(self.concat_datasets)

    def _re_label_all_dataset(self, datasets: list[Dataset],
                              compare_imgs_directly: bool = False,
                              verify_xs_align: bool = False,
                              pickle_usl_relabling: bool = False,
                              root: Optional[Path] = Path('~/data/l2l_data/'),
                              relabel_filename: Optional[str] = 'relabeling_data.pt',
                              ):
        """
        Relabels by treating all labels as unique/different. Thus, even if you have labels of cats on two data sets
        this procedure will treat them as different.

        algorithm outline:
        for each data set
        - go through all data points its data points img & label
        - collect all the original labels & img idx
        - get a global idx for each img idx
        - re-label the original labels (with corresponding img) to 0 to number of original labels to get a local label
        - re-label the local label wrt a global label and associate it to it's global idx
        - then to the reverse mapping (global index to global label)
        done, now when indexing use the self.indices_to_labels to map a global idx to a global label.
        Also, get the .labels field to keep track of the number of labels in the concatenation.
        """
        import time
        from uutils import report_times
        start = time.time()
        print(f'-> preprocessing data to relabel, might take a while... {self._re_label_all_dataset=}')
        self.img2tensor: Callable = torchvision.transforms.ToTensor()
        total_num_labels_so_far: int = 0
        global_idx: int = 0  # new_idx
        assert len(self.indices_to_labels.keys()) == 0
        assert len(self.labels_to_indices.keys()) == 0
        for dataset_idx, dataset in enumerate(datasets):
            # if hasattr(dataset, 'labels'):
            #     print(f'{len(dataset.labels)=}')
            assert len(dataset) == len(self.concat_datasets.datasets[dataset_idx])
            assert dataset == self.concat_datasets.datasets[dataset_idx]
            original_label2global_idx: defaultdict = defaultdict(list)
            for original_data_idx, (x, original_y) in enumerate(dataset):
                original_y = int(original_y)
                # - get data point from concataned data set (to compare with the data point from the data set list)
                _x, _y = self.concat_datasets[global_idx]
                _y = int(_y)
                # - sanity check concatanted data set aligns with the list of datasets
                assert original_y == _y, f'{original_y=}, {_y=}'
                if compare_imgs_directly:
                    # from PIL import ImageChops
                    # diff = ImageChops.difference(x, _x)  # https://stackoverflow.com/questions/35176639/compare-images-python-pil
                    # assert diff.getbbox(), f'comparison of imgs failed: {diff.getbbox()=}' # doesn't work :/
                    assert list(x.getdata()) == list(_x.getdata()), f'\n{list(x.getdata())=}, \n{list(_x.getdata())=}'
                # - tensor comparison of raw images
                if not isinstance(x, Tensor):
                    x, _x = self.img2tensor(x), self.img2tensor(_x)
                # if isinstance(original_y, int):
                #     original_y, _y = int2tensor(original_y), int2tensor(_y)
                if verify_xs_align:  # checks the data points after doing get item make them match.
                    # this might fails if there are random ops in the getitem
                    assert torch.equal(x,
                                       _x), f'Error for some reason, got: {dataset_idx=},' \
                                            f' {global_idx=}, {original_data_idx=}, ' \
                                            f'{x.norm()=}, {_x.norm()=}, ' \
                                            f'{x=}, {_x=}'
                # - collect original labels in dictionary keys
                original_label2global_idx[int(original_y)].append(global_idx)
                global_idx += 1
            local_num_dps: int = sum(len(global_indices) for global_indices in original_label2global_idx.values())
            assert len(dataset) == local_num_dps, f'Error: \n{local_num_dps=} \n{len(dataset)=}'
            # - do relabeling - original labeling to new global labels
            assert total_num_labels_so_far != len(dataset), f'Err:\n{total_num_labels_so_far=}\n{len(dataset)=}'
            new_local_label2global_indices: dict = {}
            global_label2global_indices: dict = {}
            # make sure to sort to avoid random looping of unordered data structures e.g. keys in a dict
            for new_local_label, original_label in enumerate(sorted(original_label2global_idx.keys())):
                global_indices: list[int] = original_label2global_idx[original_label]
                new_local_label2global_indices[int(new_local_label)] = global_indices
                new_global_label: int = total_num_labels_so_far + new_local_label
                global_label2global_indices[int(new_global_label)] = global_indices
            local_num_dps: int = sum(len(global_indices) for global_indices in original_label2global_idx.values())
            assert len(dataset) == local_num_dps, f'Error: \n{local_num_dps=} \n{len(dataset)=}'
            local_num_dps: int = sum(len(global_indices) for global_indices in new_local_label2global_indices.values())
            assert len(dataset) == local_num_dps, f'Error: \n{local_num_dps=} \n{len(dataset)=}'
            local_num_dps: int = sum(len(global_indices) for global_indices in global_label2global_indices.values())
            assert len(dataset) == local_num_dps, f'Error: \n{local_num_dps=} \n{len(dataset)=}'
            # - this assumes the integers in each data set is different, if there were unions you'd likely need semantic information about the label e.g. the string cat instead of absolute integers, or know the integers are shared between the two data sets
            # this is the step where classes are concatenated. Note due to the previous loops assuming each label is uning this should never have intersecting keys.
            dup: list = get_duplicates(list(self.labels_to_indices.keys()) + list(global_label2global_indices.keys()))
            assert len(
                dup) == 0, f'Error:\n{self.labels_to_indices.keys()=}\n{global_label2global_indices.keys()=}\n{dup=}'
            for global_label, global_indices in global_label2global_indices.items():
                # note g_idx might different to global_idx!
                global_indices: list[int]
                for g_idx in global_indices:
                    self.labels_to_indices[int(global_label)] = g_idx
                    self.indices_to_labels[g_idx] = int(global_label)
            # - update number of labels seen so far
            num_labels_for_current_dataset: int = len(original_label2global_idx.keys())
            total_num_labels_so_far += num_labels_for_current_dataset
            assert total_num_labels_so_far == len(self.labels_to_indices.keys()), f'Err:\n{total_num_labels_so_far=}' \
                                                                                  f'\n{len(self.labels_to_indices.keys())=}'
            assert global_idx == len(
                self.indices_to_labels.keys()), f'Err:\n{global_idx=}\n{len(self.indices_to_labels.keys())=}'
            if hasattr(dataset, 'labels'):
                assert len(dataset.labels) == num_labels_for_current_dataset, f'Err:\n{len(dataset.labels)=}' \
                                                                              f'\n{num_labels_for_current_dataset=}'
        # - re-labling done
        assert len(self.indices_to_labels.keys()) == len(self.concat_datasets), f'Err: ' \
                                                                                f'\n{len(self.indices_to_labels.keys())=}' \
                                                                                f'\n {len(self.concat_datasets)=}'
        if all(hasattr(dataset, 'labels') for dataset in datasets):
            assert sum(len(dataset.labels) for dataset in datasets) == total_num_labels_so_far
        # - set & validate new labels
        self.labels = range(total_num_labels_so_far)
        labels: list[int] = list(sorted(list(self.labels_to_indices.keys())))
        assert labels == list(labels), f'labels should match and be consecutive, but got: \n{labels=}, \n{self.labels=}'
        print(f'-> done data preprocessing for relabel... {self._re_label_all_dataset=}, took: {report_times(start)}')
        # - save all the data to file
        if pickle_usl_relabling:
            relabeled_data: dict = dict()
            relabeled_data['labels'] = labels
            relabeled_data['labels_to_indices'] = self.labels_to_indices
            relabeled_data['indices_to_labels'] = self.indices_to_labels
            self.save_relabeling_data_to_file(relabeled_data, root, relabel_filename)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Get's the data point and it's new label according to a mutually exclusive concatenation.

        For later?
        to do the relabling on the fly we'd need to figure out which data set idx corresponds to and to compute the
        total_num_labels_so_far. Something like this:
            current_data_set_idx = bisect_left(idx)
            total_num_labels_so_far = sum(max(_, y in dataset)+1 for dataset_idx, dataset in enumerate(self.datasets) if dataset_idx <= current_data_set_idx)
            new_y = total_num_labels_so_far + y
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

    def get_relabeling_data_from_file_(self, root: Path, relabel_filename: str = 'relabeling_data.pt') -> dict:
        # - load re-labling from file using torch.load
        root: Path = Path(root).expanduser()
        relabeled_data: dict = torch.load(root / relabel_filename)
        # - mutation assignments from ._re_label_all_dataset for consistency in case they are needed later for the correct function of this class
        self.img2tensor: Callable = torchvision.transforms.ToTensor()
        # - load relabeled data
        self.labels = relabeled_data['labels']
        self.labels_to_indices = relabeled_data['labels_to_indices']
        self.indices_to_labels = relabeled_data['indices_to_labels']
        # - asserts
        assert len(self.indices_to_labels.keys()) == len(self.concat_datasets), f'Err: ' \
                                                                                f'\n{len(self.indices_to_labels.keys())=}' \
                                                                                f'\n {len(self.concat_datasets)=}'
        labels: list[int] = list(sorted(list(self.labels_to_indices.keys())))
        assert labels == list(labels), f'labels should match and be consecutive, but got: \n{labels=}, \n{self.labels=}'
        # - print number of classes
        num_classes_results: dict = relabeled_data['num_classes_results']
        pprint_dict(num_classes_results)
        # for split, num_classes in num_classes_results.items():
        #     print(f'-> {split} num classes: {num_classes=}')
        # - done
        print(f'-> Loading relabeling data from file {root / relabel_filename} Success!')
        return relabeled_data

    def save_relabeling_data_to_file(self,
                                     relabeled_data: dict,
                                     root: Union[Path, str],
                                     relabel_filename: str,
                                     verbose_n_cls_datasets: bool = True,
                                     ) -> None:
        # - also save num classes per split
        from uutils.torch_uu.dataloaders.common import get_num_classes_l2l_list_meta_dataset
        num_classes_results: dict = get_num_classes_l2l_list_meta_dataset(self.datasets, verbose=verbose_n_cls_datasets)
        relabeled_data['num_classes_results'] = num_classes_results
        # - save relabeling data to pt file specified by relabel_filename and root using torch.save
        root: Path = expanduser(root)
        torch.save(relabeled_data, root / relabel_filename)
        # - asserts
        assert len(self.indices_to_labels.keys()) == len(self.concat_datasets), f'Err: ' \
                                                                                f'\n{len(self.indices_to_labels.keys())=}' \
                                                                                f'\n {len(self.concat_datasets)=}'
        labels: list[int] = list(sorted(list(self.labels_to_indices.keys())))
        assert labels == list(labels), f'labels should match and be consecutive, but got: \n{labels=}, \n{self.labels=}'
        # - done
        print(f'-> Saving relabeling data to file {root / relabel_filename} Success!')

    def need_to_relabel(self, root: Path, relabel_filename: str = 'relabeling_data.pt') -> bool:
        # - load re-labling from file using torch.load
        try:
            root: Path = Path(root).expanduser()
            # if the torch.load fails then def relabel and overwrite file
            relabeled_data: dict = torch.load(root / relabel_filename)
            # - mutation assignments from ._re_label_all_dataset for consistency in case they are needed later for the correct function of this class
            self.img2tensor: Callable = torchvision.transforms.ToTensor()
            # - load relabeled data
            self.labels = relabeled_data['labels']
            self.labels_to_indices = relabeled_data['labels_to_indices']
            self.indices_to_labels = relabeled_data['indices_to_labels']
            if len(self.labels) == 0 or self.labels is None:
                return True
            if len(self.labels_to_indices.keys()) == 0 or self.labels_to_indices is None:
                return True
            if len(self.indices_to_labels.keys()) == 0 or self.indices_to_labels is None:
                return True
            return False
        except Exception as e:
            print(f'need_to_relabel says we need to relabel and happened to throw this exception to detect that '
                  f'decision: {e=}')
            return True


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
                                       target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    test = torchvision.datasets.MNIST(root=root, train=False, download=True,
                                      transform=torchvision.transforms.ToTensor(),
                                      target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    concat = ConcatDatasetMutuallyExclusiveLabels([train, test], verify_xs_align=True)
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
                                          target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=True,
                                         transform=torchvision.transforms.ToTensor(),
                                         target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    concat = ConcatDatasetMutuallyExclusiveLabels([train, test], verify_xs_align=True)
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
    concat = ConcatDatasetMutuallyExclusiveLabels([validation_dataset, test_dataset])
    assert len(concat.labels) == 16 + 20, f'got {len(concat.labels)=}'
    print('---> test1 passed!')
    concat = ConcatDatasetMutuallyExclusiveLabels([train_dataset, validation_dataset, test_dataset])
    assert_dataset_is_pytorch_dataset([concat])
    print('---> test2 passed!')
    print(f'{len(concat)=}')
    print(f'{len(concat.labels)=}')
    assert len(concat) == 100 * 600, f'Err, got {len(concat)=}'
    assert len(concat.labels) == 64 + 16 + 20, f'Err got {len(concat.labels)=}'

    # - create dataloader
    loader = DataLoader(concat)
    print(f'{loader=}')
    for batch in loader:
        print(f'{batch=}')
        x, y = batch
        assert x is not None
        assert y is not None
        break
    # - loader with the code that will run it for real experiments
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders
    union_loader, _ = get_serial_or_distributed_dataloaders(train_dataset=concat, val_dataset=concat)
    print(f'{union_loader=}')
    for batch in union_loader:
        print(f'{batch=}')
        x, y = batch
        assert x is not None
        assert y is not None
        break
    print('-- done with concat mi test! --')


def hdb1_mio_check_dataloader():
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_and_omniglot_list_data_set_splits
    from uutils.torch_uu.dataloaders.common import get_serial_or_distributed_dataloaders

    root: str = '~/data/l2l_data/'
    data_augmentation: str = 'hdb1'

    # - test if data sets can be created into pytorch dataloader
    dataset_list_train, dataset_list_val, dataset_list_test = get_mi_and_omniglot_list_data_set_splits(root,
                                                                                                       data_augmentation)
    mi, omni = dataset_list_test
    loader = DataLoader(omni, num_workers=1)
    next(iter(loader))
    print()
    loader = DataLoader(mi, num_workers=1)
    next(iter(loader))
    print()

    # - hdb1/mio test
    train_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_train)
    assert len(train_dataset) == sum(len(dataset) for dataset in dataset_list_train), f'Err, got {len(train_dataset)=}'
    assert len(train_dataset.labels) == 64 + 1100, f'Err:\n{len(train_dataset.labels)=}'
    loader = DataLoader(train_dataset, num_workers=1)
    next(iter(loader))

    valid_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_val)
    assert len(valid_dataset) == sum(len(dataset) for dataset in dataset_list_val), f'Err, got {len(valid_dataset)=}'
    assert len(valid_dataset.labels) == 16 + 100, f'Err:\n{len(valid_dataset.labels)=}'
    loader = DataLoader(train_dataset, num_workers=1)
    next(iter(loader))
    loader = get_serial_or_distributed_dataloaders(valid_dataset, valid_dataset)
    next(iter(loader))

    test_dataset = ConcatDatasetMutuallyExclusiveLabels(dataset_list_test)
    assert len(test_dataset) == sum(len(dataset) for dataset in dataset_list_test), f'Err, got {len(test_dataset)=}'
    assert len(test_dataset.labels) == 20 + 423, f'Err:\n{len(test_dataset.labels)=}'
    loader = DataLoader(train_dataset, num_workers=1)
    next(iter(loader))
    loader = get_serial_or_distributed_dataloaders(test_dataset, test_dataset)
    next(iter(loader))

    # - my dataloader get function test
    train_loader, val_loader = get_serial_or_distributed_dataloaders(train_dataset, valid_dataset)
    test_loader, _ = get_serial_or_distributed_dataloaders(test_dataset, test_dataset)
    next(iter(train_loader))
    next(iter(val_loader))
    next(iter(test_loader))

    # - done
    print('-- done hdb1 tests')


def hdb4_micod_relabling_test_through_loaders():
    # - for determinism
    import random
    random.seed(0)
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)

    # - args
    from argparse import Namespace
    args = Namespace(batch_size=8, batch_size_eval=2, rank=-1, world_size=1)

    # - get data loaders
    # dataloaders: dict = hdb1_mi_omniglot_usl_all_splits_dataloaders(args)
    from uutils.torch_uu.dataloaders.usl.usl_dataloaders import hdb4_micod_usl_all_splits_dataloaders
    dataloaders: dict = hdb4_micod_usl_all_splits_dataloaders(args)

    print(dataloaders['train'].dataset.labels)
    print(dataloaders['val'].dataset.labels)
    print(dataloaders['test'].dataset.labels)
    n_train_cls: int = len(dataloaders['train'].dataset.labels)
    print('-- got the usl hdb data loaders --')

    # - loop through tasks
    import torch
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # model = get_model('resnet18', pretrained=False, num_classes=n_train_cls).to(device)
    # model = get_model('resnet18', pretrained=True, num_classes=n_train_cls).to(device)
    # from uutils.torch_uu.models.resnet_rfs import get_resnet_rfs_model_mi
    # model, _ = get_resnet_rfs_model_mi('resnet12_rfs', num_classes=n_train_cls)
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
    args.model, args.model_hps = get_default_learner_and_hps_dict()
    # - get model
    model = args.model
    model.to(device)
    from torch import nn
    criterion = nn.CrossEntropyLoss()
    for split, dataloader in dataloaders.items():
        print(f'-- {split=}')
        # next(iter(dataloaders[split]))
        for it, batch in enumerate(dataloaders[split]):
            print(f'{it=}')

            X, y = batch
            print(f'{X.size()=}')
            print(f'{y.size()=}')
            print(f'{y=}')

            y_pred = model(X)
            print(f'{y_pred.size()=}')
            # loss = criterion(y_pred, y)
            print()
            break
    print('-- end of test --')


if __name__ == '__main__':
    import time

    start = time.time()
    # - run experiment
    # check_xs_align_mnist()
    # check_xs_align_cifar100()
    # concat_data_set_mi()
    # hdb1_mio_check_dataloader()
    hdb4_micod_relabling_test_through_loaders()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
