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

from uutils.plot.image_visualization import visualize_pytorch_tensor_img, visualize_pytorch_batch_of_imgs
from uutils.torch_uu import make_code_deterministic

splits = ['train', 'validation', 'test']


# -- Run experiment

def plot_some_delauny_images_data_augmentation_visualization_experiments():
    from uutils.torch_uu.dataset.l2l_uu.delaunay_l2l import get_delauny_tasksets

    make_code_deterministic(0)
    # -
    batch_size = 5
    # kwargs: dict = dict(name='mini-imagenet', train_ways=2, train_samples=2, test_ways=2, test_samples=2)
    kwargs: dict = dict(train_ways=2, train_samples=2, test_ways=2, test_samples=2, root='~/data/delauny_l2l_bm_splits')
    kwargs['data_augmentation'] = 'original_delauny'
    kwargs['data_augmentation'] = 'original_delauny_84'
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
            visualize_pytorch_batch_of_imgs(X, show_img_now=True)
            print()
            break
        break


def plot_some_mi_images_using_l2l_hdb1_data_augmentation():
    """
    So prints some MI & hdb1 images.
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
    benchmark: BenchmarkTasksets = hdb1_mi_omniglot_tasksets(**kwargs)
    tasksets = [(split, getattr(benchmark, split)) for split in splits]
    for i, (split, taskset) in enumerate(tasksets):
        print(f'{taskset=}')
        print(f'{taskset.dataset.dataset.datasets[0].dataset.transform=}')
        # print(f'{taskset.dataset.dataset.datasets[1].dataset.transform=}')
        for task_num in range(batch_size):
            X, y = taskset.sample()
            # print(f'{X.size()=}')
            visualize_pytorch_batch_of_imgs(X, show_img_now=True)
            print()
        break


def check_size_of_mini_imagenet_original_img():
    # - not using .jpg because torchmeta & l2l rfs use pickle files
    # orig_img = Image.open(Path('assets') / 'astronaut.jpg')
    # -
    batch_size = 5
    kwargs: dict = dict(name='mini-imagenet', train_ways=2, train_samples=2, test_ways=2, test_samples=2)
    benchmark: learn2learn.BenchmarkTasksets = learn2learn.vision.benchmarks.get_tasksets(**kwargs)
    tasksets = [(split, getattr(benchmark, split)) for split in splits]
    for i, (split, taskset) in enumerate(tasksets):
        print(f'{taskset=}')
        print(f'{taskset.dataset.dataset.transform=}')
        for task_num in range(batch_size):
            X, y = taskset.sample()
            print(f'{X.size()=}')
            visualize_pytorch_batch_of_imgs(X, show_img_now=True)
            break
        break


if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    # plot_some_mi_images_using_l2l_hdb1_data_augmentation()
    # plot_some_delauny_images_data_augmentation_visualization_experiments()
    check_size_of_mini_imagenet_original_img()
    # - Done
    from uutils import report_times

    print(f"\nSuccess Done!: {report_times(start)}\a")
