"""

Goal:
- find good values for data transforms for Delauny s.t. the data augmentation is as similar to MI. This is because we
want the diversity of the images to comre form the nature of the images themselves the most and not from the data
transforms.
At the same time we want it to be representative to what is done in practice and thus plan to use standard data augmentation
strategies as much as possible.
In addition, when this data set is concatenated with Mini-Imagenet, we want fls tasks difference between the two to come
from the data itself and not from the data augmentation. Thus, we will try to make the images look as similar to the
ones from Mini-Imagenet.

options:
-
"""
import learn2learn
from learn2learn.vision.benchmarks import BenchmarkTasksets

from uutils.plot.histograms_uu import histograms_heigh_width_of_imgs_in_dataset, print_compute_useful_size_stats
from uutils.plot.image_visualization import visualize_pytorch_tensor_img, visualize_pytorch_batch_of_imgs
from uutils.torch_uu import make_code_deterministic

splits = ['train', 'validation', 'test']


# -- Run experiment

def plot_some_delauny_images_data_augmentation_visualization_experiments():
    from uutils.torch_uu.dataset.l2l_uu.delaunay_l2l import get_delauny_tasksets
    print(f'--- {plot_some_delauny_images_data_augmentation_visualization_experiments=}')

    make_code_deterministic(0)
    # -
    batch_size = 5
    # kwargs: dict = dict(name='mini-imagenet', train_ways=2, train_samples=2, test_ways=2, test_samples=2)
    kwargs: dict = dict(train_ways=2, train_samples=2, test_ways=2, test_samples=2, root='~/data/delauny_l2l_bm_splits')
    # kwargs['data_augmentation'] = '_original_delauny_only_resize_to_84'
    # kwargs['data_augmentation'] = '_original_delauny_only_resize_256'
    kwargs['data_augmentation'] = 'delauny_pad_random_resized_crop'
    print(f"{kwargs['data_augmentation']=}")

    print(f'total number of plots: {batch_size=}')
    print(f"total number of image classes: {kwargs['train_ways']=}")
    print(f"total number of images per classes: {kwargs['train_samples']=}")
    splits = ['train', 'validation', 'test']

    # - print size & plot a few images using HDB1 data augmentation, does the data augmenation look similar to omniglot & delauny?
    benchmark: BenchmarkTasksets = get_delauny_tasksets(**kwargs)
    tasksets = [(split, getattr(benchmark, split)) for split in splits]
    for i, (split, taskset) in enumerate(tasksets):
        print(f'{taskset=}')
        print(f'{taskset.dataset.dataset.transform=}')
        # print(f'{taskset.dataset.dataset.datasets[1].dataset.transform=}')
        for task_num in range(batch_size):
            X, y = taskset.sample()
            print(f'{X.size()=}')
            for img_idx in range(X.size(0)):
                visualize_pytorch_tensor_img(X[img_idx], show_img_now=True)
                if img_idx >= 5:  # print 5 images only
                    break
            visualize_pytorch_batch_of_imgs(X, show_img_now=True)
            print()
            break
        # break


def plot_some_mi_images_using_l2l_hdb1_data_augmentation2():
    """
    So prints some MI & hdb1 images.
    """
    from uutils.torch_uu.dataset.l2l_uu.delaunay_l2l import plot_some_mi_images_using_l2l_hdb1_data_augmentation
    plot_some_mi_images_using_l2l_hdb1_data_augmentation()


def check_size_of_mini_imagenet_original_img():
    import random
    import numpy as np
    import torch
    import os
    seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    import learn2learn
    batch_size = 5
    kwargs: dict = dict(name='mini-imagenet', train_ways=2, train_samples=2, test_ways=2, test_samples=2)
    kwargs['data_augmentation'] = 'lee2019'
    benchmark: learn2learn.BenchmarkTasksets = learn2learn.vision.benchmarks.get_tasksets(**kwargs)
    tasksets = [(split, getattr(benchmark, split)) for split in splits]
    for i, (split, taskset) in enumerate(tasksets):
        print(f'{taskset=}')
        print(f'{taskset.dataset.dataset.transform=}')
        for task_num in range(batch_size):
            X, y = taskset.sample()
            print(f'{X.size()=}')
            assert X.size(2) == 84
            print(f'{y.size()=}')
            print(f'{y=}')
            for img_idx in range(X.size(0)):
                visualize_pytorch_tensor_img(X[img_idx], show_img_now=True)
                if img_idx >= 5:  # print 5 images only
                    break
            # visualize_pytorch_batch_of_imgs(X, show_img_now=True)
            print()
            if task_num >= 4:  # co to get a MI image finally (note omniglot does not have padding at train...oops!)
                break
            break
        break


def check_that_padding_is_added_on_both_sides_so_in_one_dim_it_doubles_the_size():
    """
    So height -> heigh + padding * 2
    """
    from pathlib import Path
    root = Path('~/data/').expanduser()
    import torch
    import torchvision
    # - test 1, imgs (not the recommended use)
    # train = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    # test = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    # - test 2, tensor imgs
    from torchvision.transforms import Resize
    from torchvision.transforms import Pad
    from torchvision.transforms import ToTensor
    from torchvision.transforms import Compose
    transform = Compose([Resize((32, 32)), Pad(padding=4), ToTensor()])
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True,
                                          transform=transform,
                                          target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    transform = Compose([Resize((32, 32)), Pad(padding=8), ToTensor()])
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=True,
                                         transform=transform,
                                         target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    # - test padding doubles length
    #
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


def histograms_heigh_width_of_imgs_in_delauny():
    from uutils.torch_uu.dataset.delaunay_uu import get_all_delauny_dataset
    dataset = get_all_delauny_dataset(path_to_all_data='~/data/delauny_original_data/DELAUNAY')
    # histograms_heigh_width_of_imgs_in_dataset(dataset, show_hist_now=True)
    print_compute_useful_size_stats(dataset)


if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    # plot_some_mi_images_using_l2l_hdb1_data_augmentation2()
    # plot_some_delauny_images_data_augmentation_visualization_experiments()
    # check_size_of_mini_imagenet_original_img()
    # check_that_padding_is_added_on_both_sides_so_in_one_dim_it_doubles_the_size()
    # histograms_heigh_width_of_imgs_in_delauny()
    # - Done
    from uutils import report_times

    print(f"\nSuccess Done!: {report_times(start)}\a")
