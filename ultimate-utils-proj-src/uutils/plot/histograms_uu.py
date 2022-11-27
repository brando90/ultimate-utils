from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from uutils.plot import save_to_desktop
from uutils.torch_uu import make_code_deterministic


def get_num_bins(n: int, option: Optional[str] = None) -> Union[int, str]:
    """

    refs:
        - https://www.quora.com/How-do-you-determine-the-number-of-bins-in-a-histogram
        - https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
    """
    if option is None:
        return len(n) // 5
    elif option == 'log':
        return int(np.log(n))
    elif option == 'square_root':
        return int(n ** 0.5)
    elif option == 'auto':
        return 'auto'  # from seaborn https://seaborn.pydata.org/generated/seaborn.histplot.html
    else:
        raise ValueError(f'Number of bins: {option=} not implemented')


def get_histogram(array: np.ndarray,
                  xlabel: str,
                  ylabel: str,
                  title: str,

                  dpi=200,  # dots per inch,
                  facecolor: str = 'white',
                  bins: int = None,
                  show: bool = False,
                  tight_layout=False,
                  linestyle: Optional[str] = '--',
                  alpha: float = 0.75,
                  edgecolor: str = "black",
                  stat: Optional = 'count',
                  color: Optional[str] = None,
                  ):
    """ """
    # - check it's of size (N,)
    if isinstance(array, list):
        array: np.ndarray = np.array(array)
    assert array.shape == (array.shape[0],)
    assert len(array.shape) == 1
    assert isinstance(array.shape[0], int)
    # -
    # n: int = array.shape[0]
    # if bins is None:
    #     bins: int = get_num_bins(n, option='square_root')
    #     # bins: int = get_num_bins(n, option='square_root')
    # print(f'using this number of {bins=} and data size is {n=}')
    # -
    fig = plt.figure(dpi=dpi)
    fig.patch.set_facecolor(facecolor)

    import seaborn as sns
    p = sns.histplot(array, stat=stat, color=color)
    # n, bins, patches = plt.hist(array, bins=bins, facecolor='b', alpha=alpha, edgecolor=edgecolor, density=True)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    plt.grid(linestyle=linestyle) if linestyle else None
    plt.tight_layout() if tight_layout else None
    plt.show() if show else None


def histograms_heigh_width_of_imgs_in_dataset(dataset: Dataset,
                                              bins_height: int = 100,
                                              bins_width: int = 100,
                                              seed: int = 0,
                                              show_hist_now: bool = True,
                                              save_plot: bool = False,
                                              plot_name: str = 'height_width_stats_dataset',
                                              ):
    """ Plot two histograms of all the sizes of the images.

    fyi: ...[..., H, W] shape...
    """
    make_code_deterministic(seed)
    dl: DataLoader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    sizes_height: list[int] = []
    sizes_width: list[int] = []
    sizes_channel: list[int] = []
    for i, batch in enumerate(dl):
        if len(batch) == 1:
            xs = batch[0]
            print(f'{xs.size()=}')
        else:
            xs, ys = batch
            print(f'{xs.size()=} {ys=}')
            assert len(xs.size()) == 4, 'Needs to be [B, C, H , W].'
        sizes_channel.append(xs.size(1))
        sizes_height.append(xs.size(2))
        sizes_width.append(xs.size(3))
    # -
    fig, axes = plt.subplots(1, 3)

    ax = axes[0]
    ax.hist(sizes_channel, bins=bins_height)
    ax.set_xlabel("sizes (channel)")
    ax.set_ylabel("frequnecy")
    ax.legend('C')
    ax.set_title('Distribution of Sizes (channel)')

    # todo - copy the above
    axes[1].hist(sizes_height, bins=bins_height)
    plt.xlabel("sizes (height)")
    plt.ylabel("frequnecy")
    # plt.legend('H')
    plt.title('Distribution of Sizes (height)')

    axes[2].hist(sizes_width, bins=bins_width)
    plt.xlabel("sizes (width)")
    plt.ylabel("frequnecy")
    # plt.legend('W')
    plt.title('Distribution of Sizes (width)')

    # - todo - compute inter quarter range plot in wisker plot
    pass

    # -
    if save_plot:
        save_to_desktop(plot_name)
    if show_hist_now:
        plt.show()
    return


def print_compute_useful_size_stats(dataset: Dataset,
                                    size: int = 84,
                                    seed: int = 0,
                                    ):
    """ compute useful size stats."""
    make_code_deterministic(seed)
    # -
    dl: DataLoader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    sizes_height: list[int] = []
    sizes_width: list[int] = []
    sizes_channel: list[int] = []
    for i, batch in enumerate(dl):
        if len(batch) == 1:
            xs = batch[0]
            print(f'{xs.size()=}')
        else:
            xs, ys = batch
            print(f'{xs.size()=} {ys=}')
            assert len(xs.size()) == 4, 'Needs to be [B, C, H , W].'
        sizes_channel.append(xs.size(1))
        sizes_height.append(xs.size(2))
        sizes_width.append(xs.size(3))
    # -
    lst = sizes_height
    df: DataFrame = pd.DataFrame(lst)
    stats = df.describe()
    print(f'\nstats of height {stats}')
    print(f'bellow_size {bellow_size(size, lst)=} {size=}')

    lst = sizes_width
    df: DataFrame = pd.DataFrame(lst)
    stats = df.describe()
    print(f'\nstats of width {stats}')
    print(f'bellow_size {bellow_size(size, lst)=} {size=}')

    lst = sizes_channel
    df: DataFrame = pd.DataFrame(lst)
    stats = df.describe()
    print(f'\nstats of channel {stats}')
    print(f'bellow_size {bellow_size(size, lst)=} {size=}')
    # -
    return


def bellow_size(size: float, lst: list[float]) -> int:
    """Number of numbers bellow a size."""
    count: int = 0
    for l in lst:
        if l < size:
            count += 1
    return count


# - tests

def hist_test():
    from torch.utils.data import TensorDataset
    ds = TensorDataset(torch.randn(500, 3, 84, 86))
    histograms_heigh_width_of_imgs_in_dataset(ds, show_hist_now=True)


def useful_stats_test():
    from torch.utils.data import TensorDataset
    ds = TensorDataset(torch.randn(500, 3, 84, 86))
    print_compute_useful_size_stats(ds)


def dummy_task2vec_test():
    from uutils.numpy_uu.common import get_diagonal
    distance_matrix = np.array([[0., 0.24079067, 0.23218697, 0.1620301, 0.16845202],
                                [0.24079067, 0., 0.12441093, 0.26010787, 0.27561545],
                                [0.23218697, 0.12441093, 0., 0.2537393, 0.26971972],
                                [0.1620301, 0.26010787, 0.2537393, 0., 0.174303],
                                [0.16845202, 0.27561545, 0.26971972, 0.174303, 0.]])
    print(f'{distance_matrix.shape}')
    distances_as_flat_array, _, _ = get_diagonal(distance_matrix, check_if_symmetric=True)
    distances_as_flat_array: np.ndarray = np.random.randn(500)
    print(f'{distances_as_flat_array.shape=}')
    #
    title: str = 'Distribution of Task2Vec Distances'
    xlabel: str = 'Cosine Distance between Task Pairs'
    ylabel = 'Frequency Density (pmf)'
    # ylabel = 'Normalized Frequency'
    # ylabel = 'Percent'
    # - Frequency Density (pmf)
    # get_histogram(distances_as_flat_array, xlabel, ylabel, title, stat='probability')
    get_histogram(distances_as_flat_array, xlabel, ylabel, title, stat='probability', linestyle=None, color='b')
    plt.show()
    # - counts/frequency
    ylabel = 'Frequency'
    # get_histogram(distances_as_flat_array, xlabel, ylabel, title)
    get_histogram(distances_as_flat_array, xlabel, ylabel, title, linestyle=None, color='b')
    plt.show()


if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    # hist_test()
    # useful_stats_test()
    dummy_task2vec_test()
    # - Done
    from uutils import report_times

    print(f"\nSuccess Done!: {report_times(start)}\a")
