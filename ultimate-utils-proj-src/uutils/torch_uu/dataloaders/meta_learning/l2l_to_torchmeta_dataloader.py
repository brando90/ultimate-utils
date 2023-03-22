# pytorch data set (e.g. l2l, normal pytorch) using the torchmeta format
"""
key idea: sample l2l task_data

"""
from pathlib import Path
from typing import Optional, Union, Any

import torch

from argparse import Namespace

from learn2learn.data import TaskDataset, MetaDataset
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from uutils.torch_uu import tensorify
from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets

from learn2learn.vision.benchmarks import BenchmarkTasksets
import learn2learn
from learn2learn.data import TaskDataset

from pdb import set_trace as st


def get_standard_pytorch_dataset_from_l2l_taskdatasets(tasksets: BenchmarkTasksets, split: str) -> Dataset:
    """
    Trying to do:
        type(args.dataloaders.train.dataset.dataset)
        <class 'learn2learn.vision.datasets.cifarfs.CIFARFS'>

    Example call:
        dataset: Dataset = get_standard_pytorch_dataset_from_l2l_taskdatasets(args.dataloaders)

    :param tasksets:
    :param split:
    :return:
    """
    # todo, I don't actually know if this works for indexable data sets
    # trying to do something like: args.dataloaders.train
    taskset: TaskDataset = getattr(tasksets, split)
    # trying to do: type(args.dataloaders.train.dataset.dataset)
    dataset: MetaDataset = taskset.dataset
    dataset: Dataset = dataset.dataset
    # assert isinstance(args.dataloaders.train.dataset.dataset, Dataset)
    # asser isinstance(tasksets.train.dataset.dataset, Dataset)
    assert isinstance(dataset, Dataset), f'Expect dataset to be of type Dataset but got {type(dataset)=}.'
    return dataset


def get_l2l_torchmeta_dataloaders(args: Namespace) -> dict:
    """
    Returns a batch of tasks, the way that torchmeta would.
    """
    dataloaders: BenchmarkTasksets = get_l2l_tasksets(args)

    meta_trainloader = TorchMetaDLforL2L(args, 'train', dataloaders)
    meta_valloader = TorchMetaDLforL2L(args, 'val', dataloaders)
    meta_testloader = TorchMetaDLforL2L(args, 'test', dataloaders)

    args.dataloaders: dict = {'train': meta_trainloader, 'val': meta_valloader, 'test': meta_testloader}
    return args.dataloaders


class TorchMetaDLforL2L:
    """
    Returns a batch of tasks, the way that torchmeta would.

    Not intended or tested to work with ddp. For that extension see this: https://github.com/learnables/learn2learn/issues/263
    """

    def __init__(self, args, split: str, dataloaders: BenchmarkTasksets):
        self.args = args
        assert split in ['train', 'val', 'test']
        self.split = split
        self.batch_size = None
        self.dataloaders = dataloaders

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
            - task_dataset: TaskDataset = getattr(args.dataloaders, split)
            - recall a task is a "mini (spt) classification data set" e.g. with n classes and k shots (and it's qry set too)
            - torchmeta example: https://tristandeleu.github.io/pytorch-meta/
        """

        shots = self.args.k_shots
        ways = self.args.n_classes
        # meta_batch_size: int = max(self.args.batch_size // self.args.world_size, 1)
        meta_batch_size: int = self.batch_size

        task_dataset: TaskDataset = getattr(self.dataloaders, self.split)

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


class EpisodicBatchAsTaskDataset(TaskDataset):

    def __init__(self, batch: Tensor, verbose: bool = False):
        self.batch = batch
        self.idx = 0
        self.verbose = verbose
        self.num_tasks = self.batch[0].size(0)
        assert len(batch) == 4, f'Error: Expected 4 tensors in batch because we have 4 [spt_x, spt_y, qry_x, qry_y] ' \
                                f'but got {len(batch)}.'
        if self.verbose:
            print(f'Running {self.__init__=}')
            self.debug_print()

    def sample(self, idx: Optional[int] = None,
               ) -> list[Tensor, Tensor]:
        """
        Gets a single task from the batch of tasks.

        the l2l forward pass is as dollows:
            meta_losses, meta_accs = [], []
            for task in range(meta_batch_size):
                # print(f'{task=}')
                # - Sample all data data for spt & qry sets for current task: thus size [n*(k+k_eval), C, H, W] (or [n(k+k_eval), D])
                task_data: list = task_dataset.sample()  # data, labels

                # -- Inner Loop Adaptation
                learner = meta_learner.maml.clone()
                loss, acc = fast_adapt(
                    args=args,
                    task_data=task_data,
                    learner=learner,
                    loss=args.loss,
                    adaptation_steps=meta_learner.nb_inner_train_steps,
                    shots=args.k_shots,
                    ways=args.n_classes,
                    device=args.device,
                )
        therefore, we need to concatenate the tasks in the right dimension and return it. The foward pass then splits it
        according to the shots and ways on its own.
        """
        # - checks
        if self.verbose:
            print(f'{self.sample=}')
            print(f'{self.idx=}')
            print(f'{idx=}')
            print(f'{self.num_tasks=}')
            print(f'{len(self)=}')
            self.debug_print()
        assert len(self) == self.num_tasks, f'Error, expected {len(self)=} got {self.num_tasks=}.'
        # - get idx
        if idx is None:
            idx = self.idx
        else:
            self.idx = idx

        # - want x, y to be of shape: [n*(k+k_eval), C, H, W] (or [n(k+k_eval), D])
        spt_x, spt_y, qry_x, qry_y = self.batch
        spt_x, spt_y, qry_x, qry_y = spt_x[idx], spt_y[idx], qry_x[idx], qry_y[idx]

        # - concatenate spt_x, qry_x & spt_y, qry_y
        x = torch.cat([spt_x, qry_x], dim=0)
        y = torch.cat([spt_y, qry_y], dim=0)
        task_data: list = [x, y]

        # - return
        if self.verbose:
            print(f'{self.idx=}')
            print(f'{x.size()=}')
            print(f'{y.size()=}')
        self.idx += 1
        assert len(self) == self.num_tasks, f'Error, expected {len(self)=} got {self.num_tasks=}.'
        return task_data

    def __len__(self) -> int:
        """ Returns numbers of tasks. Should be (meta) batch size. """
        num_tasks: int = self.batch[0].size(0)
        assert num_tasks == self.num_tasks, f'Error, expected {num_tasks=}, got {self.num_tasks=}.'
        return num_tasks

    def __repr__(self) -> str:
        self_str = f'{self.__class__.__name__}({self.idx=}, {self.num_tasks=}, {len(self)=}, {self.batch[0].size(0)=})'
        return self_str

    def debug_print(self):
        if self.verbose:
            print(f'{self=}')
            print(f'{len(self.batch)=} (should be 4 e.g. [spt_x, spt_y, qry_x, qry_y])')
            print(f'{self.batch[0].size()=}')  # e.g. for vision it should be [B, n*k, C, H, W]
            print(f'{self.batch[1].size()=}')  # e.g. for vision it should be [B, n*k]
            print(f'{self.batch[2].size()=}')  # e.g. for vision it should be [B, n*k, C, H, W]
            print(f'{self.batch[3].size()=}')  # e.g. for vision it should be [B, n*k]
            print(f'{self.idx=}')
            print(f'{self.num_tasks=}')
            print(f'{len(self)=}')


def episodic_batch_2_task_dataset(batch: Tensor,
                                  loader: DataLoader,
                                  meta_learner: Optional = None,  # MAMLMetaLearnerL2L
                                  ) -> Union[TaskDataset, Any]:
    """ Convert episodic batch -> task dataset, else return batch as is (i.e. do identity). """
    if hasattr(loader, 'episodic_batch_2_task_dataset'):
        # - checks
        assert loader.episodic_batch_2_task_dataset, f'Error: {loader.episodic_batch_2_task_dataset=},' \
                                                     f' but it should be True.'
        if meta_learner is not None:
            from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearnerL2L
            assert isinstance(meta_learner, MAMLMetaLearnerL2L), f'Error: meta_learner is: {type(meta_learner)=}'
        # - convert episodic batch -> task dataset
        batch: TaskDataset = EpisodicBatchAsTaskDataset(batch)
    return batch


# - tests

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


def maml_l2l_test_():
    # - try forward pass with random data first
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.argparse_uu.meta_learning import get_args_vit_mdl_maml_l2l_agent_default
    args: Namespace = get_args_vit_mdl_maml_l2l_agent_default()
    from uutils.torch_uu.distributed import set_devices
    set_devices(args)
    get_and_create_model_opt_scheduler_for_run(args)
    print(f'{type(args.model)=}')
    from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearnerL2L
    args.agent = MAMLMetaLearnerL2L(args, args.model)
    args.meta_learner = args.agent
    print(f'{type(args.agent)=}, {type(args.meta_learner)=}')
    # create a list of four tensors, idx 0,2 of size [25, 3, 84, 84], [75, 3, 84, 84] using torch.randn and te other two idx 1,3 of size [25], [75] of type long
    # print('--> random data test as episodic batch')
    # import torch
    # spt_x = torch.randn(3, 25, 3, 84, 84)
    # spt_y = torch.randint(low=0, high=5, size=(3, 25,), dtype=torch.long)
    # qry_x = torch.randn(3, 75, 3, 84, 84)
    # qry_y = torch.randint(low=0, high=5, size=(3, 75,), dtype=torch.long)
    # batch: list = [spt_x, spt_y, qry_x, qry_y]
    # task_data: TaskDataset = EpisodicBatchAsTaskDataset(batch)
    # train_loss, train_acc = args.meta_learner(task_data, training=True)
    # print(f'{train_loss, train_acc=}')

    # - torchmeta, maybe we could test with torchmeta mi dataloaders
    print('--> mini-imagenet test with episodic batch (from torch meta)')
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
    args.dataloaders = get_meta_learning_dataloaders(args)
    batch: Any = next(iter(args.dataloaders['train']))
    spt_x, spt_y, qry_x, qry_y = batch['train'][0], batch['train'][1], batch['test'][0], batch['test'][1]
    batch: list = [spt_x, spt_y, qry_x, qry_y]
    task_data: TaskDataset = EpisodicBatchAsTaskDataset(batch)
    train_loss, train_acc = args.meta_learner(task_data, training=True)
    print(f'{train_loss, train_acc=}')
    # train_loss, train_acc = meta_train_fixed_iterations(args, args.agent, args.dataloaders, args.opt, args.scheduler)
    # print(f'{train_loss, train_acc=}')

def forward_pass_with_pretrain_convergence_ffl_meta_learner():
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.argparse_uu.meta_learning import get_args_vit_mdl_maml_l2l_agent_default
    args: Namespace = get_args_vit_mdl_maml_l2l_agent_default()
    from uutils.torch_uu.distributed import set_devices
    set_devices(args)
    get_and_create_model_opt_scheduler_for_run(args)
    print(f'{type(args.model)=}')
    from uutils.torch_uu.meta_learners.pretrain_convergence import FitFinalLayer
    args.agent = FitFinalLayer(args, args.model)
    args.meta_learner = args.agent
    print(f'{type(args.agent)=}, {type(args.meta_learner)=}')

    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
    args.dataloaders = get_meta_learning_dataloaders(args)
    batch: Any = next(iter(args.dataloaders['train']))
    spt_x, spt_y, qry_x, qry_y = batch['train'][0], batch['train'][1], batch['test'][0], batch['test'][1]
    batch: list = [spt_x, spt_y, qry_x, qry_y]
    train_loss, train_loss_ci, train_acc, train_acc_ci = args.meta_learner(batch, training=True)
    print(f'{train_loss, train_loss_ci, train_acc, train_acc_ci=}')


"""
python ~/ultimate-utils/ultimate-utils-proj-src/uutils/torch_uu/dataloaders/meta_learning/l2l_to_torchmeta_dataloader.py
"""

if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    # maml_l2l_test_()
    forward_pass_with_pretrain_convergence_ffl_meta_learner()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
