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
from learn2learn.vision.benchmarks import BenchmarkTasksets

from uutils.plot.image_visualization import visualize_pytorch_tensor_img, visualize_pytorch_batch_of_imgs

splits = ['train', 'validation', 'test']


# -- Run experiment

def plot_some_delauny_images_using_RandomCrop():
    # benchmark: BenchmarkTasksets = get_delauny_tasksets()
    pass


def plot_some_mi_images_using_l2l_hdb1_data_augmentation():
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import hdb1_mi_omniglot_tasksets

    # -
    batch_size = 5
    kwargs: dict = dict(train_ways=4, train_samples=4, test_ways=1, test_samples=1)

    # - print size & plot a few images using HDB1 data augmentation, does the data augmenation look similar to omniglot & delauny?
    benchmark: BenchmarkTasksets = hdb1_mi_omniglot_tasksets(**kwargs)
    tasksets = [getattr(benchmark, split) for split in splits]
    for i, taskset in enumerate(tasksets):
        print(f'-- {splits[i]=}')
        for task_num in range(batch_size):
            print(f'{task_num=}')

            X, y = taskset.sample()
            print(f'{X.size()=}')
            print(f'{y.size()=}')
            print(f'{y=}')
            # visualize_pytorch_tensor_img(X[0], show_img_now=True)
            visualize_pytorch_batch_of_imgs(X)
            print()
    pass


def plot_some_omniglot_images_using_hdb1_data_augmentation():
    pass


if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    plot_some_mi_images_using_l2l_hdb1_data_augmentation()
    # - Done
    from uutils import report_times

    print(f"\nSuccess Done!: {report_times(start)}\a")
