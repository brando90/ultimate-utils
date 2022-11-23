"""
Outline of how to create a l2l taskloader:
- for now see https://github.com/learnables/learn2learn/issues/375

note:
    - to create a task loader from a new data set: https://github.com/learnables/learn2learn/issues/375
    - remember we have a l2l to torchmeta loader converter. So for now all code is written in l2l's api.
"""
import os

import learn2learn
from learn2learn.vision.benchmarks import BenchmarkTasksets
from learn2learn.data import MetaDataset
from torch.utils.data import Dataset

from uutils.torch_uu import make_code_deterministic
from uutils.torch_uu.dataset.delaunay_uu import get_l2l_bm_split_paths, get_delauny_dataset_splits

from learn2learn.data.transforms import TaskTransform


def get_remaining_delauny_l2l_task_transforms(dataset: MetaDataset, ways: int, samples: int) -> list[TaskTransform]:
    """
    Get delauny task transforms to create fsl tasks. Same as mi.

    Same as MI to create fsl tasks: NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels.
    """
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import get_remaining_transforms_mi
    remaining_task_transforms: TaskTransform = get_remaining_transforms_mi(dataset, ways, samples)
    return remaining_task_transforms


def get_delauny_l2l_datasets_and_task_transforms(train_ways=5,
                                                 train_samples=10,
                                                 test_ways=5,
                                                 test_samples=10,
                                                 root: str = '~/data/delauny_l2l_bm_splits',
                                                 data_augmentation: str = 'delauny_uu',
                                                 device=None,
                                                 **kwargs,
                                                 ):
    """ """
    # - get data set with its data transfroms
    path2train, path2val, path2test = get_l2l_bm_split_paths(root)
    train_dataset, valid_dataset, test_dataset = get_delauny_dataset_splits(path2train, path2val, path2test,
                                                                            data_augmentation)
    if device is None:
        pass  # in the original l2l it hardcodes a side effect to change the data transforms for some reason, we already put the data transforms so do nothing
    else:
        train_dataset = learn2learn.OnDeviceDataset(
            dataset=train_dataset,
            transform=train_dataset.transform,
            device=device,
        )
        valid_dataset = learn2learn.OnDeviceDataset(
            dataset=valid_dataset,
            transform=valid_dataset.transform,
            device=device,
        )
        test_dataset = learn2learn.OnDeviceDataset(
            dataset=test_dataset,
            transform=test_dataset.transform,
            device=device,
        )
    train_dataset: MetaDataset = MetaDataset(train_dataset)
    valid_dataset: MetaDataset = MetaDataset(valid_dataset)
    test_dataset: MetaDataset = MetaDataset(test_dataset)

    # - decided same as MI transforms because I don't acppreciate the omniglot one + delauny seems more similar to mi? (since it has colours...?)
    train_transforms = get_remaining_delauny_l2l_task_transforms(train_dataset, train_ways, train_samples)
    valid_transforms = get_remaining_delauny_l2l_task_transforms(valid_dataset, test_ways, test_samples)
    test_transforms = get_remaining_delauny_l2l_task_transforms(test_dataset, test_ways, test_samples)

    # - add names to be able to get the right task transform for the indexable dataset
    train_dataset.name = 'train_delauny_uu'
    valid_dataset.name = 'val_delauny_uu'
    test_dataset.name = 'test_delauny_uu'

    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms


def setup_dot_labels_field():
    """ see delauny_uu

    just call the other one, assert it worked.
    """
    raise NotImplementedError


def get_delauny_tasksets(
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        num_tasks=-1,  # let it be -1 for continual tasks https://github.com/learnables/learn2learn/issues/315
        root='~/data/delauny_l2l_bm_splits',
        data_augmentation: str = '',
        device=None,
        **kwargs,
) -> BenchmarkTasksets:
    """ """
    print(f'-> {data_augmentation=}')
    root = os.path.expanduser(root)
    # Load task-specific data and transforms
    datasets, transforms = get_delauny_l2l_datasets_and_task_transforms(train_ways=train_ways,
                                                                        train_samples=train_samples,
                                                                        test_ways=test_ways,
                                                                        test_samples=test_samples,
                                                                        root=root,
                                                                        data_augmentation=data_augmentation,
                                                                        device=device,
                                                                        **kwargs)
    train_dataset, validation_dataset, test_dataset = datasets
    train_transforms, validation_transforms, test_transforms = transforms

    # Instantiate the tasksets
    train_tasks = learn2learn.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = learn2learn.data.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks,
    )
    test_tasks = learn2learn.data.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks,
    )
    return BenchmarkTasksets(train_tasks, validation_tasks, test_tasks)


# -- Run experiment

def loop_through_delaunay():
    print(f'test: {loop_through_delaunay=}')
    from uutils.plot.image_visualization import visualize_pytorch_tensor_img
    # - for determinism
    seed = 0
    make_code_deterministic(seed)

    # - get benchmark
    batch_size = 5
    kwargs: dict = dict(train_ways=2, train_samples=2, test_ways=2, test_samples=2, root='~/data/delauny_l2l_bm_splits')
    kwargs['data_augmentation'] = 'delauny_pad_random_resized_crop'
    print(f"{kwargs['data_augmentation']=}")

    print(f'total number of plots: {batch_size=}')
    print(f"total number of image classes: {kwargs['train_ways']=}")
    print(f"total number of images per classes: {kwargs['train_samples']=}")
    splits = ['train', 'validation', 'test']

    # - get model
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # from models import get_model
    # model = get_model('resnet18', pretrained=False, num_classes=5).to(device)
    # model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V2")
    # model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
    model, _ = get_default_learner_and_hps_dict()  # 5cnn
    model.to(device)
    from torch import nn
    criterion = nn.CrossEntropyLoss()
    # - loop through tasks
    benchmark: BenchmarkTasksets = get_delauny_tasksets(**kwargs)
    tasksets = [(split, getattr(benchmark, split)) for split in splits]
    for i, (split, taskset) in enumerate(tasksets):
        for task_num in range(batch_size):
            print(f'{task_num=}')
            print(f'-- {split=}')
            print(f'{taskset.dataset.dataset.transform=}')

            X, y = taskset.sample()
            print(f'{X.size()=}')
            print(f'{y.size()=}')
            print(f'{y=}')
            # for img_idx in range(X.size(0)):
            #     visualize_pytorch_tensor_img(X[img_idx], show_img_now=True)
            #     if img_idx >= 5:  # print 5 images only
            #         break

            assert X.size(2) == 84
            y_pred = model(X)
            loss = criterion(y_pred, y)
            print(f'{loss=}')
            print()
            break
        # break
    print(f'done test: {loop_through_delaunay=}')
    # - print some mi examples too
    # plot_some_mi_images_using_l2l_hdb1_data_augmentation()
    return


def plot_some_mi_images_using_l2l_hdb1_data_augmentation():
    """
    So prints some MI & hdb1 images.

    https://stackoverflow.com/questions/74482017/why-isnt-randomcrop-inserting-the-padding-in-pytorch
    """
    # - for determinism
    import random
    random.seed(0)
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)

    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import hdb1_mi_omniglot_tasksets
    from uutils.plot.image_visualization import visualize_pytorch_tensor_img
    from uutils.torch_uu import make_code_deterministic

    make_code_deterministic(0)
    # -
    batch_size = 5
    # kwargs: dict = dict(name='mini-imagenet', train_ways=2, train_samples=2, test_ways=2, test_samples=2)
    kwargs: dict = dict(train_ways=2, train_samples=2, test_ways=2, test_samples=2)
    print(f'total number of plots: {batch_size=}')
    print(f"total number of image classes: {kwargs['train_ways']=}")
    print(f"total number of images per classes: {kwargs['train_samples']=}")
    splits = ['train', 'validation', 'test']

    # - print size & plot a few images using HDB1 data augmentation, does the data augmenation look similar to omniglot & delauny?
    # benchmark: learn2learn.BenchmarkTasksets = learn2learn.vision.benchmarks.get_tasksets(**kwargs)
    benchmark: BenchmarkTasksets = hdb1_mi_omniglot_tasksets(**kwargs)
    tasksets = [(split, getattr(benchmark, split)) for split in splits]
    for i, (split, taskset) in enumerate(tasksets):
        print(f'-- {splits[i]=}')
        print(f'{taskset=}')
        print(f'{taskset.dataset.dataset.datasets[0].dataset.transform=}')
        print(f'{taskset.dataset.dataset.datasets[1].dataset.dataset.transform=}')
        for task_num in range(batch_size):
            print(f'-- {splits[i]=}')
            print(f'{task_num=}')
            print(f'{taskset.dataset.dataset.datasets[0].dataset.transform=}')
            print(f'{taskset.dataset.dataset.datasets[1].dataset.dataset.transform=}')
            X, y = taskset.sample()
            print(f'{X.size()=}')
            assert X.size(2) == 84
            print(f'{y.size()=}')
            print(f'{y=}')
            for img_idx in range(X.size(0)):
                visualize_pytorch_tensor_img(X[img_idx], show_img_now=True)
                if img_idx >= 5:  # print 5 images only
                    break
            print()
            if task_num >= 4:  # so to get a MI image finally (note omniglot does not have padding at train...oops!)
                break
        # break


def torchmeta_plot_images_is_the_padding_there():
    """ Padding should be there look at torchmeta_ml_dataloaders.py """
    from uutils.torch_uu import process_meta_batch
    from uutils.torch_uu.dataloaders.meta_learning.torchmeta_ml_dataloaders import \
        get_minimum_args_for_torchmeta_mini_imagenet_dataloader
    from uutils.torch_uu.dataloaders.meta_learning.torchmeta_ml_dataloaders import \
        get_miniimagenet_dataloaders_torchmeta
    from diversity_src.experiment_mains.main_experiment_analysis_sl_vs_maml_performance_comp_distance import \
        resnet12rfs_mi

    from uutils.plot.image_visualization import visualize_pytorch_tensor_img
    # - for determinism
    import random
    random.seed(0)
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)

    # -
    args = get_minimum_args_for_torchmeta_mini_imagenet_dataloader()
    args = resnet12rfs_mi(args)
    print(f'{args=}')
    dataloaders = get_miniimagenet_dataloaders_torchmeta(args)

    print(f'{len(dataloaders)}')
    for split, datalaoder in dataloaders.items():
        for batch_idx, batch in enumerate(datalaoder):
            spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
            print(f'Train inputs shape: {spt_x.size()}')  # e.g. (2, 25, 3, 28, 28)
            print(f'Train targets shape: {spt_y.size()}'.format(spt_y.shape))  # e.g. (2, 25)
            batch_size = spt_x.size(0)
            for task_num in range(batch_size):
                print(f'-- {split=}')
                print(f'{task_num=}')
                print(f'{datalaoder.dataset.dataset.transform}')
                X: torch.Tensor = spt_x[task_num]
                print(f'{X.size()=}')
                assert X.size(2) == 84
                for img_idx in range(X.size(0)):
                    visualize_pytorch_tensor_img(X[img_idx], show_img_now=True)
                    if img_idx >= 5:  # print 5 images only
                        break
                print()
                if task_num >= 4:  # so to get a MI image finally (note omniglot does not have padding at train...oops!)
                    break


def check_padding_random_crop_cifar_pure_torch():
    # -
    import sys
    print(f'python version: {sys.version=}')
    import torch
    print(f'{torch.__version__=}')
    # -
    from uutils.plot.image_visualization import visualize_pytorch_tensor_img
    from torchvision.transforms import RandomCrop

    # - for determinism
    import random
    random.seed(0)
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)

    # -
    from pathlib import Path
    root = Path('~/data/').expanduser()
    import torch
    import torchvision
    # - test tensor imgs
    from torchvision.transforms import Resize
    from torchvision.transforms import Pad
    from torchvision.transforms import ToTensor
    from torchvision.transforms import Compose

    # -- see if pad doubles length
    print(f'--- test padding doubles length with Pad(...)')
    transform = Compose([Resize((32, 32)), Pad(padding=4), ToTensor()])
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True,
                                          transform=transform,
                                          target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    transform = Compose([Resize((32, 32)), Pad(padding=8), ToTensor()])
    test = torchvision.datasets.CIFAR100(root=root, train=True, download=True,
                                         transform=transform,
                                         target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    # - test padding doubles length
    from torch.utils.data import DataLoader
    loader = DataLoader(train)
    x, y = next(iter(loader))
    print(f'{x[0].size()=}')
    assert x[0].size(2) == 32 + 4 * 2
    assert x[0].size(2) == 32 + 8
    visualize_pytorch_tensor_img(x[0], show_img_now=True)
    #
    loader = DataLoader(test)
    x, y = next(iter(loader))
    print(f'{x[0].size()=}')
    assert x.size(2) == 32 + 8 * 2
    assert x.size(2) == 32 + 16
    visualize_pytorch_tensor_img(x[0], show_img_now=True)

    # -- see if RandomCrop also puts the pad
    print(f'--- test RandomCrop indeed puts padding')
    transform = Compose([Resize((32, 32)), RandomCrop(28, padding=8), ToTensor()])
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True,
                                          transform=transform,
                                          target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    transform = Compose([Resize((32, 32)), RandomCrop(28), ToTensor()])
    test = torchvision.datasets.CIFAR100(root=root, train=True, download=True,
                                         transform=transform,
                                         target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    # - test that the padding is there visually
    from torch.utils.data import DataLoader
    loader = DataLoader(train)
    x, y = next(iter(loader))
    print(f'{x[0].size()=}')
    assert x[0].size(2) == 28
    visualize_pytorch_tensor_img(x[0], show_img_now=True)
    #
    loader = DataLoader(test)
    x, y = next(iter(loader))
    print(f'{x[0].size()=}')
    assert x.size(2) == 28
    visualize_pytorch_tensor_img(x[0], show_img_now=True)


if __name__ == "__main__":
    import time
    from uutils import report_times

    import sys

    print(f'python version: {sys.version=}')
    import torch

    print(f'{torch.__version__=}')

    start = time.time()
    # - run experiment
    loop_through_delaunay()
    # plot_some_mi_images_using_l2l_hdb1_data_augmentation()
    # torchmeta_plot_images_is_the_padding_there()
    # check_padding_random_crop_cifar_pure_torch()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
