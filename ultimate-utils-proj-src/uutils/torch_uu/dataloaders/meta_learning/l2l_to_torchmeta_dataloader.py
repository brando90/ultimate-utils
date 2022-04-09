# pytorch data set (e.g. l2l, normal pytorch) using the torchmeta format
"""
key idea: sample l2l task_data

"""
from pathlib import Path

import torch

from argparse import Namespace

from learn2learn.data import TaskDataset, MetaDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from uutils.torch_uu import tensorify
from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets

from learn2learn.vision.benchmarks import BenchmarkTasksets


def get_standard_pytorch_dataset_from_l2l_taskdatasets(tasksets: BenchmarkTasksets, split: str) -> Dataset:
    """
    Trying to do:
        type(args.tasksets.train.dataset.dataset)
        <class 'learn2learn.vision.datasets.cifarfs.CIFARFS'>

    Example call:
        dataset: Dataset = get_standard_pytorch_dataset_from_l2l_taskdatasets(args.tasksets)

    :param tasksets:
    :param split:
    :return:
    """
    # trying to do something like: args.tasksets.train
    taskset: TaskDataset = getattr(tasksets, split)
    # trying to do: type(args.tasksets.train.dataset.dataset)
    dataset: MetaDataset = taskset.dataset
    dataset: Dataset = dataset.dataset
    # assert isinstance(args.tasksets.train.dataset.dataset, Dataset)
    # asser isinstance(tasksets.train.dataset.dataset, Dataset)
    assert isinstance(dataset, Dataset), f'Expect dataset to be of type Dataset but got {type(dataset)=}.'
    return dataset


def get_l2l_torchmeta_dataloaders(args: Namespace) -> dict:
    args.tasksets: BenchmarkTasksets = get_l2l_tasksets(args)

    meta_trainloader = TorchMetaDLforL2L(args, split='train')
    meta_valloader = TorchMetaDLforL2L(args, split='val')
    meta_testloader = TorchMetaDLforL2L(args, split='test')

    args.dataloaders: dict = {'train': meta_trainloader, 'val': meta_valloader, 'test': meta_testloader}
    return args.dataloaders


class TorchMetaDLforL2L:
    """
    Not intended or tested to work with ddp. For that extension see this: https://github.com/learnables/learn2learn/issues/263
    """

    def __init__(self, args, split: str):
        self.args = args
        assert split in ['train', 'val', 'test']
        self.split = split
        self.batch_size = None

    def __iter__(self):
        # initialization code for iterator usually goes here, I don't think we need any
        if self.split == 'train':
            self.batch_size = self.args.batch_size
        else:
            self.split = 'validation' if self.split == 'val' else self.split
            self.batch_size = self.args.batch_size_eval
        return self

    def __next__(self):
        """
        Returns a batch of tasks, the way that torchmeta would.

        :return: {'train': (spt_x, spt_y), 'test': (qry_x, qry_y)} with spt_x of size [B, k*n, D]
            and spt y_of size [B, k*n, D]. qry_x of size [B, k*n, D] and spt_y of size
        depending on your task.

        note:
            - task_dataset: TaskDataset = getattr(args.tasksets, split)
            - recall a task is a "mini (spt) classification data set" e.g. with n classes and k shots (and it's qry set too)
            - torchmeta example: https://tristandeleu.github.io/pytorch-meta/
        """
        import learn2learn
        from learn2learn.data import TaskDataset

        shots = self.args.k_shots
        ways = self.args.n_classes
        # meta_batch_size: int = max(self.args.batch_size // self.args.world_size, 1)
        meta_batch_size: int = self.batch_size

        task_dataset: TaskDataset = getattr(self.args.tasksets, self.split)

        spt_x, spt_y, qry_x, qry_y = [], [], [], []
        for task in range(meta_batch_size):
            # - Sample all data data for spt & qry sets for current task: thus size [n*(k+k_eval), C, H, W] (or [n(k+k_eval), D])
            task_data: list = task_dataset.sample()  # data, labels
            data, labels = task_data
            # data, labels = data.to(device), labels.to(device)  # note here do it process_meta_batch, only do when using original l2l training method

            # Separate data into adaptation/evalutation sets
            # [n*(k+k_eval), C, H, W] -> [n*k, C, H, W] and [n*k_eval, C, H, W]
            (support_data, support_labels), (query_data, query_labels) = learn2learn.data.partition_task(
                data=data,
                labels=labels,
                shots=shots,  # shots to separate to two data sets of size shots and k_eval
            )
            # checks coordinate 0 of size() [n*(k + k_eval), C, H, W]
            assert support_data.size(0) == shots * ways, f' Expected {shots * ways} but got {support_data.size(0)}'
            # checks [n*k] since these are the labels
            assert support_labels.size() == torch.Size([shots * ways])

            # append task to lists of tasks to ultimately create: [B, n*k, D], [B, n*k], [B, n*k_eval, D], [B, n*k_eval],
            spt_x.append(support_data)
            spt_y.append(support_labels)
            qry_x.append(query_data)
            qry_y.append(query_labels)
        #
        spt_x, spt_y, qry_x, qry_y = tensorify(spt_x), tensorify(spt_y), tensorify(qry_x), tensorify(qry_y)
        assert spt_x.size(0) == meta_batch_size, f'Error, expected {spt_x.size(0)=} got {meta_batch_size=}.'
        assert qry_x.size(0) == meta_batch_size, f'Error, expected {spt_x.size(0)=} got {meta_batch_size=}.'
        # should be sizes [B, n*k, C, H, W] or [B,]
        # - Return meta-batch of tasks
        batch = {'train': (spt_x, spt_y), 'test': (qry_x, qry_y)}
        return batch


# - tests

# def args_5cnn_cifarfs(args: Namespace) -> Namespace:
#     """
#     """
#     from uutils.torch_uu.models.resnet_rfs import get_recommended_batch_size_cifarfs_resnet12rfs_body, \
#         get_feature_extractor_conv_layers
#     # - model
#     args.model_option = '4CNN_l2l_cifarfs'
#
#     # - data
#     args.data_option = 'cifarfs_rfs'  # no name assumes l2l, make sure you're calling get_l2l_tasksets
#     args.data_path = Path('~/data/l2l_data/').expanduser()
#     args.data_augmentation = 'rfs2020'
#     args.augment_train = True
#
#     args.batch_size = 5
#     args.batch_size_eval = args.batch_size
#
#     # - fix for backwards compatibility
#     args = fix_for_backwards_compatibility(args)
#     # - setup paths to ckpts for data analysis
#     args = setup_args_path_for_ckpt_data_analysis(args, 'ckpt.pt')
#     # - fill in the missing things and make sure things make sense for run
#     args = uutils.setup_args_for_experiment(args)
#     return args

def l2l_example(meta_batch_size: int = 4, num_iterations: int = 5):
    from uutils.torch_uu import process_meta_batch
    args: Namespace = Namespace(k_shots=5, n_classes=5, k_eval=15, data_option='cifarfs_rfs',
                                data_path=Path('~/data/l2l_data/'), batch_size=4, batch_size_eval=6,
                                data_augmentation='rfs2020', device='cpu')

    dataloaders: dict = get_l2l_torchmeta_dataloaders(args)
    for split, dataloader in dataloaders.items():
        print(f'{split=}')
        assert dataloader.split == split
        for batch in dataloader:
            train_inputs, train_targets = batch["train"]
            spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
            print('Train inputs shape: {0}'.format(train_inputs.shape))  # (16, 25, 1, 28, 28)
            print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

            test_inputs, test_targets = batch["test"]
            print('Test inputs shape: {0}'.format(test_inputs.shape))  # (16, 75, 1, 28, 28)
            print('Test targets shape: {0}'.format(test_targets.shape))  # (16, 75)
            break


if __name__ == '__main__':
    l2l_example()
    print('Done!\a')
