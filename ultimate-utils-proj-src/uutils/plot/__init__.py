from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

from typing import Union

Array = Union[list, np.ndarray]

def plot_simple(y: Array, xlabel: str, ylabel: str, linewidth: float = 2.0, show: bool = False, save_plot: bool = False,
                plot_name: str = 'plot'):
    """
    Plots y against it's indices
    """
    plt.plot(y, lw=linewidth)  # lw is linewidth
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    # note: needs to be done in this order or it will clear the plot.
    if save_plot:
        save_to_desktop(plot_name)
    if show:
        plt.show()

def plot(x: Array, y: Array, xlabel: str, ylabel: str, linewidth: float = 2.0, show: bool = False,
         save_plot: bool = False, plot_name: str = 'plot'):
    """
    Plots y against x.
    """
    plt.plot(x, y, lw=linewidth)  # lw is linewidth
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    # note: needs to be done in this order or it will clear the plot.
    if save_plot:
        save_to_desktop(plot_name)
    if show:
        plt.show()

def save_to_desktop(plot_name: str = 'plot'):
    """
    Assuming you have not called show, saves it to local users desktop as a png, svg & pdf.
    """
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / f'{plot_name}.png')
    plt.savefig(root / f'{plot_name}.svg')
    plt.savefig(root / f'{plot_name}.pdf')

def save_to_home(plot_name: str = 'plot'):
    """
    Assuming you have not called show, saves it to local users desktop as a png, svg & pdf.
    """
    root = Path('~/').expanduser()
    plt.savefig(root / f'{plot_name}.png')
    plt.savefig(root / f'{plot_name}.svg')
    plt.savefig(root / f'{plot_name}.pdf')

def save_to(root: Path, plot_name: str = 'plot'):
    """
    Assuming there is a plot in display, saves it to local users desktop users desktop as a png, svg & pdf.
    """
    root: Path = root.expanduser()
    plt.savefig(root / f'{plot_name}.png')
    plt.savefig(root / f'{plot_name}.svg')
    plt.savefig(root / f'{plot_name}.pdf')

# - tests

def save_plot_test():
    """
    Should show the plot and also save three none-empty plot.
    """
    x = [1, 2, 3]
    y = [2, 3, 4]
    plot(x, y, 'x', 'y', show=True, save_plot=True)


if __name__ == '__main__':
    save_plot_test()
    print('Done, success! \a')

