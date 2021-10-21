from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

from typing import Union, Optional

Array = Union[list, np.ndarray]

# ref: https://stackoverflow.com/questions/47074423/how-to-get-default-blue-colour-of-matplotlib-pyplot-scatter/47074742
MY_DEFAULT_BLUE: str = '#1f77b4'  # I like this blue but it might change whats default,
MDB: str = MY_DEFAULT_BLUE

def plot_quick(y: Array, xlabel: str, ylabel: str, linewidth: float = 2.0, show: bool = False, save_plot: bool = False,
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

def _plot(x: Array, y: Array, xlabel: str, ylabel: str,
         linewidth: float = 2.0, show: bool = False,
         save_plot: bool = False, plot_name: str = 'plot', title: Optional[str] = None):
    """
    Plots y against x.
    """
    plt.plot(x, y, lw=linewidth)  # lw is linewidth
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    # - optionals
    plt.title(title)
    # note: needs to be done in this order or it will clear the plot.
    if save_plot:
        save_to_desktop(plot_name)
    if show:
        plt.show()

def plot(x: Array, y: Array, xlabel: str, ylabel: str,
         linewidth: float = 2.0, show: bool = False,
         save_plot: bool = False, plot_name: str = 'plot', title: Optional[str] = None, label: Optional[str] = None,
         y_hline: Optional[float] = None, y_hline_label: Optional[str] = None,
         x_hline: Optional[float] = None, x_hline_label: Optional[str] = None):
    """
    Nice easy plot function to quickly plot x vs y and labeling the x and y.

    Nice optional args, like plotting straight (horizontal or vertical lines), saving plots, showing the plot, adding
    optional legends etc.
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, tight_layout=True)
    axs.plot(x, y, marker='x', label=label, lw=linewidth, color=MDB)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)
    axs.grid()  # adds nice grids instead of plot being white
    plt.tight_layout()  # automatically adjusts subplot params so that the subplot(s) fits in to the figure area.
    # - optionals
    if x_hline:
        axs.axvline(x=x_hline, color='g', linestyle='--', label=x_hline_label)
    if y_hline:
        axs.axhline(y=y_hline, color='r', linestyle='--', label=y_hline_label)
    if label or y_hline_label or x_hline_label:
        axs.legend()  # LABEL = LEGEND. A legend is an area describing the elements of the graph.
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

