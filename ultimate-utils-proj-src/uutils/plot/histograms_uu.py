import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from uutils.plot import save_to_desktop
from uutils.torch_uu import make_code_deterministic


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


if __name__ == "__main__":
    import time

    start = time.time()
    # - run experiment
    # hist_test()
    useful_stats_test()
    # - Done
    from uutils import report_times

    print(f"\nSuccess Done!: {report_times(start)}\a")
