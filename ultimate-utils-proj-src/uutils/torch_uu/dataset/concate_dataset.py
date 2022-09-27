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
"""
from pathlib import Path

import torchvision
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
        # maps a class label to a list of sample indices with that label.
        self.labels_to_indices = {}
        # maps a sample index to its corresponding class label.
        self.indices_to_labels = {}
        # - do the relabeling
        new_label = 0
        dataset: Dataset
        for dataset in datasets:
            for x, y in dataset:
                print(f'{(x, y)=}')
                print()
        print()

    # def __len__(self):
    #     return len(self.landmarks_frame)
    #
    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #
    #     img_name = os.path.join(self.root_dir,
    #                             self.landmarks_frame.iloc[idx, 0])
    #     image = io.imread(img_name)
    #     landmarks = self.landmarks_frame.iloc[idx, 1:]
    #     landmarks = np.array([landmarks])
    #     landmarks = landmarks.astype('float').reshape(-1, 2)
    #     sample = {'image': image, 'landmarks': landmarks}
    #
    #     if self.transform:
    #         sample = self.transform(sample)
    #
    #     return sample

def assert_dataset_is_pytorch_dataset(datasets: list):
    """ to do 1 data set wrap it in a list"""
    for dataset in datasets:
        print(f'{type(dataset)=}')
        print(f'{type(dataset.dataset)=}')
        assert isinstance(dataset, Dataset), f'Expect dataset to be of type Dataset but got {type(dataset)=}.'

def assert_relabling_is_correct(dataset: Dataset) -> dict:
    counts: dict = {}
    for x, y in dataset:
        if y not in counts:
            counts[y] = 0
        else:
            counts[y] += 1
    return counts

def check_counts(counts: dict, labels: int = 100, counts_per_label: int = 600):
    """
    default values are for MI.

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
    assert label == labels

# - tests

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
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_mi_datasets
    train_dataset, validation_dataset, test_dataset = get_mi_datasets()
    assert_dataset_is_pytorch_dataset([train_dataset, validation_dataset, test_dataset])
    from learn2learn.data import UnionMetaDataset
    union = UnionMetaDataset([train_dataset, validation_dataset, test_dataset])
    assert len(union) == 100


if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    # check_cifar100_is_100_in_usl()
    check_mi_omniglot_in_usl()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
