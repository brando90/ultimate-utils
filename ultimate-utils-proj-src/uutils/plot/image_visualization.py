import learn2learn.vision.benchmarks
import numpy as np
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from uutils.torch_uu import make_code_deterministic


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    """
    ref: https://pytorch.org/vision/main/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
    """
    plt.rcParams["savefig.bbox"] = 'tight'
    from PIL.Image import Image
    orig_img = Image.open(Path('assets') / 'astronaut.jpg')
    # if you change the seed, make sure that the randomly-applied transforms
    # properly show that the image can be both transformed and *not* transformed!
    torch.manual_seed(0)

    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def plot_images(images, labels, preds=None):
    """
    ref:
        - https://gist.github.com/MattKleinsmith/5226a94bad5dd12ed0b871aed98cb123
    """

    assert len(images) == len(labels) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0, :, :], interpolation='spline16', cmap='gray')

        label = str(labels[i])
        if preds is None:
            xlabel = label
        else:
            pred = str(preds[i])
            xlabel = "True: {0}\nPred: {1}".format(label, pred)

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def visualize_some_samples(dataset: Dataset, shuffle=False, num_workers=4, pin_memory=False):
    sample_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=9,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory)
    """
    """
    from uutils.plot.image_visualization import plot_images
    data_iter = iter(sample_loader)
    images, labels = data_iter.next()
    X = images.numpy()
    plot_images(X, labels)


def visualize_some_images(images: list, labels):
    X = images.numpy()
    plot_images(X, labels)


def visualize_pytorch_tensor_img(tensor_image: torch.Tensor, show_img_now: bool = False):
    """
    Due to channel orders not agreeing in pt and matplot lib.
    Given a Tensor representing the image, use .permute() to put the channels as the last dimension:

    ref: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
    """
    assert len(tensor_image.size()) == 3, f'Err your tensor is the wrong shape {tensor_image.size()=}' \
                                          f'likely it should have been a single tensor with 3 channels' \
                                          f'i.e. CHW.'
    if tensor_image.resize(0) == 3:  # three chanels
        plt.imshow(tensor_image.permute(1, 2, 0))
    else:
        plt.imshow(tensor_image)
    if show_img_now:
        plt.tight_layout()
        plt.show()


def visualize_pytorch_batch_of_imgs(imgs: torch.Tensor, show_img_now: bool = False):
    assert len(imgs.size()) == 4, f'Err your tensor is the wrong shape {imgs.size()=}' \
                                  f'likely it should have been a single tensor with 3 channels.'
    assert imgs.size(1) == 3 or imgs.size(1) == 1, f'Err, first dim should be channel so it should be ' \
                                                   f'3 or 1 but got {imgs.size(0)=}'
    # - make the vertical subplots
    batch_size = imgs.size(0)
    nrows, ncols = 1, batch_size
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    assert isinstance(axes, np.ndarray)
    axes: np.ndarray = axes.reshape((nrows, ncols))
    assert axes.shape == (nrows, ncols)

    # - plot them
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            img = imgs[col_idx]
            img = img.permute(1, 2, 0)
            # print(f'{type(img)=}')
            # print(f'{img=}')
            assert len(img.size()) == 3
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            # img = np.asarray(img)
            # ax.imshow(np.asarray(img))
            # ax.imshow(np.asarray(img))
            # ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            print()

    # - show images
    if show_img_now:
        plt.tight_layout()
        plt.show()


# - test

def visualize_pytorch_tensor_img_playground():
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import hdb1_mi_omniglot_tasksets

    # -
    batch_size = 5
    kwargs: dict = dict(train_ways=4, train_samples=4, test_ways=1, test_samples=1)
    splits = ['train', 'validation', 'test']

    # - print size & plot a few images using HDB1 data augmentation, does the data augmenation look similar to omniglot & delauny?
    benchmark: learn2learn.BenchmarkTasksets = hdb1_mi_omniglot_tasksets(**kwargs)
    tasksets = [getattr(benchmark, split) for split in splits]
    for i, taskset in enumerate(tasksets):
        for task_num in range(batch_size):
            X, y = taskset.sample()
            visualize_pytorch_tensor_img(X[0], show_img_now=True)
            break


def visualize_pytorch_tensor_batch_imgs_playground():
    """
    _TASKSETS = {
        'omniglot': omniglot_tasksets,
        'mini-imagenet': mini_imagenet_tasksets,
        'tiered-imagenet': tiered_imagenet_tasksets,
        'fc100': fc100_tasksets,
        'cifarfs': cifarfs_tasksets,
    }
    """
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import hdb1_mi_omniglot_tasksets

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
    benchmark: learn2learn.BenchmarkTasksets = hdb1_mi_omniglot_tasksets(**kwargs)
    tasksets = [getattr(benchmark, split) for split in splits]
    for i, taskset in enumerate(tasksets):
        print(f'{taskset=}')
        print(f'{taskset.dataset.dataset.datasets[0].dataset.transform=}')
        # print(f'{taskset.dataset.dataset.datasets[1].dataset.transform=}')
        for task_num in range(batch_size):
            X, y = taskset.sample()
            # print(f'{X.size()=}')
            visualize_pytorch_batch_of_imgs(X, show_img_now=True)
            print()
        break


if __name__ == '__main__':
    import time

    start = time.time()
    print('starting')
    # - run experiment
    # visualize_pytorch_tensor_img_playground()
    visualize_pytorch_tensor_batch_imgs_playground()
    # - Done
    from uutils import report_times

    print(f"\nSuccess Done!: {report_times(start)}\a")
    pass
