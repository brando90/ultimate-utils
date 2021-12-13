import torch

import numpy as np

from torchmeta.utils.data import Task, MetaDataset

from pathlib import Path

from pdb import set_trace as st

# example interface for MiniImagenet
# metaset_miniimagenet = torchmeta.datasets.MiniImagenet(data_path, num_classes_per_task=5, meta_train=True, download=True)
class RandFNN(MetaDataset):

    def __init__(
            self,
            data_path,
            meta_split='train',
            noise_std=None,
            transform=None,
            target_transform=None,
            dataset_transform=None):
        super().__init__(meta_split=meta_split,
                         target_transform=target_transform,
                         dataset_transform=dataset_transform)
        #
        self.data_path = (data_path / meta_split).expanduser()
        self.tasks_folders = sorted([f for f in self.data_path.iterdir() if f.is_dir()])
        assert ('f_avg' not in self.tasks_folders)
        assert ('.' not in self.tasks_folders and '..' not in self.tasks_folders)
        self.num_tasks = len(self.tasks_folders)
        self.transform = transform
        self.noise_std = noise_std
        #
        self.np_random = None  # put here to match Sinusoid, but it's hardcoded to None

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        # get task/function with such index
        task_foldername = self.tasks_folders[index]
        # print(f'task_foldername = {task_foldername}')

        # get task/function
        task = RandFNNTask(index,
                           task_path=self.data_path / task_foldername,
                           noise_std=self.noise_std,
                           np_random=self.np_random)

        # note: needed or classplitter transform won't work for getting train/spt & qry/test splits
        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task


class RandFNNTask(Task):
    def __init__(self, index, task_path, noise_std, transform=None, target_transform=None, np_random=None):
        try:
            super().__init__(index=index, num_classes=None)  # Regression task
        except:
            super().__init__(num_classes=None)  # Regression task
        self.noise_std = noise_std
        self.task_path = task_path  # path to all data for function/task f_i

        self.transform = transform
        self.target_transform = target_transform

        # get data from disk
        db = torch.load(str(self.task_path / 'fi_db.pt'))
        self.inputs = db['x']  # as numpy
        self.targets = db['y']  # as numpy
        self.f = db['f']  # true target function

        # get task length (# of samples)
        self.num_samples = len(self.inputs)

        # process targets
        if np_random is None:
            np_random = np.random.RandomState(None)
        if (noise_std is not None) and (noise_std > 0.):
            self.targets += self.targets + (noise_std * np_random.randn(self.num_samples, 1))
        assert(len(self.inputs) == len(self.targets))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        input, target = self.inputs[index], self.targets[index]

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

def test_rand_nn():
    # loop through meta-batches of this data set, print the size, make sure it's the size you exepct
    from torchmeta.utils.data import BatchMetaDataLoader
    from torchmeta.transforms import ClassSplitter
    from torchmeta.toy import Sinusoid

    from tqdm import tqdm

    # paths for test
    meta_split = 'train'
    data_path = Path('~/data/debug_dataset_LS_fully_connected_NN_with_BN/meta_set_fully_connected_NN_with_BN_std1_4.0_std2_1.0_noise_std0.1/').expanduser()

    # get data set
    dataset = RandFNN(data_path, meta_split)
    shots, test_shots = 5, 15
    # get metaset
    metaset = ClassSplitter(
        dataset,
        num_train_per_class=shots,
        num_test_per_class=test_shots,
        shuffle=True)
    # get meta-dataloader
    batch_size = 16
    num_workers = 0
    meta_dataloader = BatchMetaDataLoader(metaset, batch_size=batch_size, num_workers=num_workers)
    epochs = 2

    print(f'batch_size = {batch_size}')
    print(f'len(metaset) = {len(metaset)}')
    print(f'len(meta_dataloader) = {len(meta_dataloader)}\n')
    with tqdm(range(epochs)) as tepochs:
        for epoch in tepochs:
            print(f'\n[epoch={epoch}]')
            for batch_idx, batch in enumerate(meta_dataloader):
                print(f'batch_idx = {batch_idx}')
                train_inputs, train_targets = batch['train']
                test_inputs, test_targets = batch['test']
                print(f'train_inputs.shape = {train_inputs.shape}')
                print(f'train_targets.shape = {train_targets.shape}')
                print(f'test_inputs.shape = {test_inputs.shape}')
                print(f'test_targets.shape = {test_targets.shape}')
                print()


def test_sinusoid():
    # loop through meta-batches of this data set, print the size, make sure it's the size you exepct
    from torchmeta.utils.data import BatchMetaDataLoader
    from torchmeta.transforms import ClassSplitter
    from torchmeta.toy import Sinusoid

    from tqdm import tqdm

    dataset = Sinusoid(num_samples_per_task=100, num_tasks=20)
    shots, test_shots = 5, 15
    # get metaset
    metaset = ClassSplitter(
        dataset,
        num_train_per_class=shots,
        num_test_per_class=test_shots,
        shuffle=True)
    # get meta-dataloader
    batch_size = 16
    num_workers = 0
    meta_dataloader = BatchMetaDataLoader(metaset, batch_size=batch_size, num_workers=num_workers)
    epochs = 2

    print(f'batch_size = {batch_size}')
    print(f'len(metaset) = {len(metaset)}')
    print(f'len(meta_dataloader) = {len(meta_dataloader)}\n')
    with tqdm(range(epochs)) as tepochs:
        for epoch in tepochs:
            print(f'\n[epoch={epoch}]')
            for batch_idx, batch in enumerate(meta_dataloader):
                print(f'batch_idx = {batch_idx}')
                train_inputs, train_targets = batch['train']
                test_inputs, test_targets = batch['test']
                print(f'train_inputs.shape = {train_inputs.shape}')
                print(f'train_targets.shape = {train_targets.shape}')
                print(f'test_inputs.shape = {test_inputs.shape}')
                print(f'test_targets.shape = {test_targets.shape}')
                print()

if __name__ == '__main__':
    print('\nstarting test...')
    test_rand_nn()
    # test_sinusoid()
    print('Done with test! \a')

