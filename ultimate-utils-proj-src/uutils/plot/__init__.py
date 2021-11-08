from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

from typing import Union, Optional, OrderedDict

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
         save_plot: bool = False, plot_filename: str = 'plot', title: Optional[str] = None):
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
        save_to_desktop(plot_filename)
    if show:
        plt.show()

def plot(x: Array, y: Array, xlabel: str, ylabel: str,
         linewidth: float = 2.0, show: bool = False,
         save_plot: bool = False, plot_filename: str = 'plot', title: Optional[str] = None, label: Optional[str] = None,
         y_hline: Optional[float] = None, y_hline_label: Optional[str] = None,
         x_hline: Optional[float] = None, x_hline_label: Optional[str] = None):
    """
    Nice easy plot function to quickly plot x vs y and labeling the x and y.

    Nice optional args, like plotting straight (horizontal or vertical lines), saving plots, showing the plot, adding
    optional legends etc.
    Saves png, svg, pdf, to Desktop automatically if save_plot=True.

    Easiest use: plot(x, y, xlabel, ylabel)
    Easy recommended use: plost(x, y, xlabel, ylabel, save_plot=True, title=title)
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, tight_layout=True)
    axs.plot(x, y, marker='x', label=label, lw=linewidth, color=MDB)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)
    axs.grid()  # adds nice grids instead of plot being white
    plt.tight_layout()  # automatically adjusts subplot params so that the subplot(s) fits in to the figure area.
    # - optionals
    if x_hline:  # horizontal sets a constant x value
        axs.axvline(x=x_hline, color='g', linestyle='--', label=x_hline_label)
    if y_hline:  # vertical sets a constant y value
        axs.axhline(y=y_hline, color='r', linestyle='--', label=y_hline_label)
    if label or y_hline_label or x_hline_label:
        axs.legend()  # LABEL = LEGEND. A legend is an area describing the elements of the graph.
    if save_plot:
        save_to_desktop(plot_filename)
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

# - seaborn

LayerIdentifier = str

def _list_order_dict2data_frame(data: list[OrderedDict[LayerIdentifier, float]], metric: str):
    """
    Converts [B, L] to pandas dataframe (just a table) with the row as:
        {'layer_name': f'Layer{col}', 'metric': metric, 'sample_val': data[row, col]}
    so it makes sure it has the metric repeated for all sample values basically in one single table.
    """
    from pandas import DataFrame
    B: int = len(list)  # e.g. data.shape[0] number of columns
    L: int = len(list)  # e.g. data.shape[1] number of rows
    for row in range(B):  # b
        for col in range(L):  # l
            df_row = {'layer_name': f'Layer{col}', 'metric': metric, 'sample_val': data[row, col]}
            # - make sure you do it like this so that the previous data frame is added to the new one and the assignment
            # is there to not forget the growing data_frame (really a table)
            data_df: DataFrame = data_df.append(df_row, ignore_index=True)
    return data_df


def _list_metrics_full_data2data_frame(dist_metric2data: OrderedDict[str, list[OrderedDict[LayerIdentifier, float]]]):
    """
    lst_data should be a list of experiment results for each metric. e.g.
        dist_metric2data[metric_name] -> [B, L] matrix for the individual sample values for each layer (total B*L sample values).
    usually you have 4 num_metrics [svcca, pwcca, lincka, opd].
    Then the data frame effective has size [4, B, L] values but all flattened into a table where each row has a row value
    as [row_arbitrary_name, metric, layer_name, val]
    """
    import pandas as pd
    from pandas import DataFrame
    column_names = ["layer_name", "metric", "sample_val"]
    data_df: DataFrame = pd.DataFrame(columns=column_names)
    for metric, data in dist_metric2data.items():
        # assert data is size [B, L]
        # get [B, L] -> have the metric in the table
        new_data_df: DataFrame = _list_order_dict2data_frame(data, metric)
        # - append current data to growing data frame
        data_df = data_df.join(new_data_df)
    return data_df

def plot_seaborn_table_with_metric(dist_metric2data: OrderedDict[str, list[OrderedDict[LayerIdentifier, float]]]):
    """
    The main idea of this function is that we have a collection of values in [B, L] e.g. B sim/dist values for each
    layer L organized as a list[OrderedDict[str, float]].
    But we have one for each metric type so we have another OrderDict[metric] -> [B, L] data.
    But we need to append it all into one table and have one column for the metric and each [B, L] has that value
    appended to it. Each metric essentially will be it's one curve (with error bands).
    Then the hue parameter can create one curve (with error bans) for each metric. Since there are [B, L] of them
    it aggregates along B by specifying what the y sample is and makes sure the x-axis are plotted in the horizontal
    line by specifying layer_name to be the x-axis.

    Notes:
        - hue parameter: this specifies which rows in the df to bunch as a [Samples, x-axis-values] so to consider
        that as the data matrix for a specific curve.
    """
    import seaborn as sns
    from pandas import DataFrame
    # sns.set(color_codes=True)
    # - join all data for each metric
    data_df: DataFrame = _list_metrics_full_data2data_frame(dist_metric2data)
    # - plot, crucial
    sns.lineplot(x='layer_name', y='sample_val', hue='metric', data=data_df, err_style='band')
    sns.lineplot(x='layer_name', y='sample_val', hue='metric', data=data_df, err_style='band')

# - tests

def save_plot_test():
    """
    Should show the plot and also save three none-empty plot.
    """
    x = [1, 2, 3]
    y = [2, 3, 4]
    plot(x, y, 'x', 'y', show=True, save_plot=True)


def my_example_seaborn_error_bands():
    """
    An example of how to plot my [B, L] for each metric.


    https://seaborn.pydata.org/tutorial/relational.html#relational-tutorial

    https://seaborn.pydata.org/examples/errorband_lineplots.html
    https://www.youtube.com/watch?v=G3F0EZcW9Ew
    https://github.com/knathanieltucker/seaborn-weird-parts/commit/3e571fd8e211ea04b6c9577fd548e7e532507acf
    https://github.com/knathanieltucker/seaborn-weird-parts/blob/3e571fd8e211ea04b6c9577fd548e7e532507acf/tsplot.ipynb

    https://seaborn.pydata.org/examples/errorband_lineplots.html
    """
    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt
    import pandas as pd

    print(sns)

    np.random.seed(22)
    sns.set(color_codes=True)

    # the number of x values to consider in a given range e.g. [0,1] will sample 10 raw features x sampled at in [0,1] interval
    num_x: int = 4
    # the repetitions for each x feature value e.g. multiple measurements for sample x=0.0 up to x=1.0 at the end
    rep_per_x: int = 5  # B
    total_size_data_set: int = num_x * rep_per_x
    print(f'{total_size_data_set=}')
    # - create fake data set
    # only consider 10 features from 0 to 1
    x = np.arange(start=1, stop=5, step=num_x)
    # to introduce fake variation add uniform noise to each feature and pretend each one is a new observation for that feature
    noise_uniform: np.ndarray = np.random.rand(rep_per_x, num_x)
    # same as above but have the noise be the same for each x (thats what the 1 means)
    noise_normal: np.ndarray = np.random.randn(rep_per_x, 1)
    # signal function
    sin_signal: np.ndarray = np.sin(x)
    cos_signal: np.ndarray = np.cos(x)
    # [rep_per_x, num_x]
    data1: np.ndarray = sin_signal + noise_uniform + noise_normal
    data2: np.ndarray = cos_signal + noise_uniform + noise_normal

    column_names = ["layer_name", "metric", "sample_val"]
    data_df = pd.DataFrame(columns=column_names)

    data = data1
    metric = 'sin'
    for row in range(data.shape[0]):  # b
        for col in range(data.shape[1]):  # l
            df_row = {'layer_name': f'Layer{col}', 'metric': metric, 'sample_val': data[row, col]}
            data_df = data_df.append(df_row, ignore_index=True)

    data = data2
    metric = 'cos'
    for row in range(data.shape[0]):  # b
        for col in range(data.shape[1]):  # l
            df_row = {'layer_name': f'Layer{col}', 'metric': metric, 'sample_val': data[row, col]}
            data_df = data_df.append(df_row, ignore_index=True)

    print(data_df)
    sns.lineplot(x='layer_name', y='sample_val', hue='metric', data=data_df, err_style='band')

    plt.show()

def default_seabron_example():
    """
    https://seaborn.pydata.org/examples/errorband_lineplots.html
    """
    import seaborn as sns
    from matplotlib import pyplot as plt
    from pandas import DataFrame

    fmri: DataFrame = sns.load_dataset("fmri")

    # important print to know the format of the data frame
    print(fmri)

    # plot
    sns.lineplot(x="timepoint", y="signal",  hue="region", style="event", data=fmri)
    plt.show()


if __name__ == '__main__':
    save_plot_test()
    # my_example_seaborn_error_bands()
    # default_seabron_example()
    print('Done, success! \a')

