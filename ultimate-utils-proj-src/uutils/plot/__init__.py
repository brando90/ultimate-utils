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
    plt.grid(True)
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
    plt.grid(True)
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
         save_plot: bool = False, plot_filename: str = 'plot', title: Optional[str] = None,
         label: Optional[str] = None,
         y_hline: Optional[float] = None, y_hline_label: Optional[str] = None,
         x_hline: Optional[float] = None, x_hline_label: Optional[str] = None,
         new_plot: bool = False, marker: Optional = None, color: Optional = None,
         tight_layout: bool = False
         ):
    """
    Nice easy plot function to quickly plot x vs y and labeling the x and y.

    Nice optional args, like plotting straight (horizontal or vertical lines), saving MI_plots_sl_vs_maml_1st_attempt, showing the plot, adding
    optional legends etc.
    Saves png, svg, pdf, to Desktop automatically if save_plot=True.

    Easiest use: plot(x, y, xlabel, ylabel)
    Easy recommended use: plost(x, y, xlabel, ylabel, save_plot=True, title=title)
    """
    if new_plot:
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, tight_layout=True)
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        axs.set_title(title)
    else:
        axs = plt
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        axs.title(title)
    axs.plot(x, y, marker=marker, label=label, lw=linewidth, color=color)
    axs.grid(True)  # adds nice grids instead of plot being white
    if tight_layout:
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


def plot_with_error_bands(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                          xlabel: str, ylabel: str,
                          title: str,
                          curve_label: Optional[str] = None,
                          error_band_label: Optional[str] = None,
                          x_vals_as_symbols: Optional[list[str]] = None,
                          color: Optional[str] = None, ecolor: Optional[str] = None,
                          linewidth: float = 1.0,
                          style: Optional[str] = 'default',
                          capsize: float = 3.0,
                          alpha: float = 0.2,
                          ylim: Optional[tuple[float, float]] = None,
                          xlim: Optional[tuple[float, float]] = None,
                          show: bool = False
                          ):
    """
    Plot custom error bands given x and y.

    note:
        - example values for color and ecolor:
            color='tab:blue', ecolor='tab:blue'
        - capsize is the length of the horizontal line for the error bar. Larger number makes it longer horizontally.
        - alpha value create than 0.2 make the error bands color for filling it too dark. Really consider not changing.
        - sample values for curves and error_band labels:
            curve_label: str = 'mean with error bars',
            error_band_label: str = 'error band',
        - use x_vals_as_symbols to have strings in the x-axis for each individual points. Warning, it might clutter the
        x-axis so use just a few.
    refs:
        - for making the seaborn and matplot lib look the same see: https://stackoverflow.com/questions/54522709/my-seaborn-and-matplotlib-plots-look-the-same
    """
    if style == 'default':
        # use the standard matplotlib
        plt.style.use("default")
    elif style == 'seaborn' or style == 'sns':
        # looks idential to seaborn
        import seaborn as sns
        sns.set()
    elif style == 'seaborn-darkgrid':
        # uses the default colours of matplot but with blue background of seaborn
        plt.style.use("seaborn-darkgrid")
    elif style == 'ggplot':
        # other alternative to something that looks like seaborn
        plt.style.use('ggplot')

    # ax = plt.gca()
    # fig = plt.gcf(
    # fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, tight_layout=True)
    # - if symbols in x axis instead of raw x value
    if x_vals_as_symbols is not None:
        # plt.xticks(x, [f'val{v}' for v in x]) to test
        plt.xticks(x, x_vals_as_symbols)
    # - plot bands
    plt.errorbar(x=x, y=y, yerr=yerr, color=color, ecolor=ecolor,
                 capsize=capsize, linewidth=linewidth, label=curve_label)
    plt.fill_between(x=x, y1=y - yerr, y2=y + yerr, alpha=alpha, label=error_band_label)
    plt.grid(True)
    if curve_label or error_band_label:
        plt.legend()
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

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


def draw_veritcal_line(at_x_value: float, start: float, stop: float, num: int = 100, linestyle: str = "--"):
    """
    Draws a vertical line at a specific x value for a given range [start, stop].

    Note:
        - num tells us how dense to draw this line.
    """
    plt.plot([at_x_value] * num, np.linspace(start, stop, num), linestyle=linestyle)


# - seaborn

LayerIdentifier = str


def _list_order_dict2data_frame(data: list[OrderedDict[LayerIdentifier, float]], metric: str):
    """
    Converts [B, L] to pandas dataframe (just a table) with the row as:
        {'layer_name': f'Layer{col}', 'metric': metric, 'sample_val': data[row, col]}
    so it makes sure it has the metric repeated for all sample values basically in one single table.
    """
    from pandas import DataFrame
    # - for each data value in [B, L] create a table augmented with the name for that data (which associates all the
    # values for one curve together in the table)
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

    x, yvectors or keys in data
    Variables that specify positions on the x and y axes.

    huevector or key in data
    Grouping variable that will produce lines with different colors. Can be either categorical or numeric, although color mapping will behave differently in latter case
    """
    import seaborn as sns
    from pandas import DataFrame
    # sns.set(color_codes=True)
    # - join all data for each metric
    data_df: DataFrame = _list_metrics_full_data2data_frame(dist_metric2data)
    # - plot, crucial
    sns.lineplot(x='layer_name', y='sample_val', hue='metric', data=data_df, err_style='band')
    sns.lineplot(x='layer_name', y='sample_val', hue='metric', data=data_df, err_style='band')


def plot_seaborn_curve_with_x_values_y_values(x: np.ndarray, y: np.ndarray,
                                              xlabel: str, ylabel: str,
                                              title: str,
                                              curve_label: Optional[str] = None,
                                              err_style: str = 'band',
                                              marker: Optional[str] = 'x',
                                              dashes: bool = False,
                                              show: bool = False
                                              ):
    """
    Given a list of x values in a range with num_x_values number of x values in that range and the corresponding samples
    for each specific x value (so [samples_per_x] for each value of x giving in total a matrix of size
    [samples_per_x, num_x_values]), plot aggregates of them with error bands.
    Note that the main assumption is that each x value has a number of y values corresponding to it (likely due to noise
    for example).


    Note:
        - if you want string in the specific x axis point do
        sns.lineplot(x=np.tile([f'Layer{i}' for i in range(1, num_x+1)], rep_per_x),...) assuming the x values are the
        layers. https://stackoverflow.com/questions/69888181/how-to-show-error-bands-for-pure-matrices-samples-x-range-with-seaborn-error/69889619?noredirect=1#comment123544763_69889619
        - note you can all this function multiple times to insert different curves to your plot.
        - note its recommended call show only for if you have one or at the final curve you want to add.
        - if you want bands and bars it might work if you call this function twice but using the bar and band argument
        for each call.

    ref:
        - https://stackoverflow.com/questions/69888181/how-to-show-error-bands-for-pure-matrices-samples-x-range-with-seaborn-error/69889619?noredirect=1#comment123544763_69889619

    :param x: [num_x_values]
    :param y: [samples_per_x, num_x_values]
    :param xlabel:
    :param ylabel:
    :param title:
    :param curve_label:
    :param err_style:
    :param marker:
    :param dashes:
    :param show:
    :return:
    """
    import seaborn as sns
    samples_per_x: int = y.shape[0]
    num_x_values: int = x.shape[0]
    assert (num_x_values == y.shape[1]), f'We are plotting aggreagates for one specific value of x multple values of y,' \
                                         f'thus we need to have the same number of x values match in the x axis.'

    # - since seaborn expects a an x value paired with it's y value, let's flatten the y's and make sure the corresponding
    # x value is aligned with it's y value [num_x_values * samples_per_x]
    x: np.ndarray = np.tile(x,
                            samples_per_x)  # np.tile = Construct an array by repeating A the number of times given by reps.
    assert (x.shape == (num_x_values * samples_per_x,))
    y: np.ndarray = np.ravel(y)  # flatten the y's to match the x values to have the x to it's corresponding y
    assert (y.shape == (num_x_values * samples_per_x,))
    assert (x.shape == y.shape)

    # - plot
    ax = sns.lineplot(x=x, y=y, err_style=err_style, label=curve_label, marker=marker, dashes=dashes)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()


# - tables

def put_pm_to_pandas_data(data: dict) -> dict:
    """
    Change the +- to \pm for latex display.

    Note: to have the pandas frame display the table string correctly use the escapte=False as in:
        latex_table: str = df.to_latex(index=False, escape=False, caption='caption goes here', label='label_goes_here')

    ref:
        - https://stackoverflow.com/questions/70008992/how-to-print-a-literal-backslash-to-get-pm-in-a-pandas-data-frame-to-generate-a
    """
    for column_name, data_values in data.items():
        data[column_name] = [data_value.replace('+-', ' $\pm$ ') for data_value in data_values]
    return data


def get_latex_table_as_text_nice_default(data_frame, column_format: Optional = None) -> str:
    """
    Transform a pandas data frame to latex table string with nice default format similar to this paper:
    https://arxiv.org/pdf/1909.09157.pdf

    ref:
        - https://stackoverflow.com/questions/70010421/how-does-one-put-the-caption-and-label-at-the-end-of-a-latex-string-outputted-by
        - https://stackoverflow.com/questions/70008992/how-to-print-a-literal-backslash-to-get-pm-in-a-pandas-data-frame-to-generate-a
        - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html

    :param data_frame:
    :param column_format: None or default left or centered for centering
    :return:
    """
    import pandas as pd
    data_frame: pd.DataFrame = data_frame
    # - get data, where initial keys are the colums see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    data: dict = data_frame.to_dict()
    if column_format is None:
        pass
    elif column_format == 'centered':
        # - create string c...c according to number of columns
        column_format = ''.join(['c' for table_column in data.keys()])
    else:
        raise ValueError(f'Invalid option, got: {column_format=}')
    # - see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html
    latex_table: str = data_frame.to_latex(index=False,
                                           escape=False,
                                           column_format=column_format,
                                           caption='caption goes here, what is your table saying?',
                                           label='label_for_using_as_ref_goes_here',
                                           position='!h')
    return latex_table


# def _data_frame_2_table(data: dict):
#     import pandas as pd
#
#     # - to pandas table
#     df = pd.DataFrame(data)
#     print(df)
#
#     # https://stackoverflow.com/questions/70009242/how-does-one-generate-latex-table-images-with-proper-equations-from-python-panda
#
#     # - to latex
#     data = put_pm_to_pandas_data(data)
#     df = pd.DataFrame(data)
#
#     print(df.to_latex(index=False, escape=False))
#
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     from pandas.plotting import table
#
#     # ax = plt.subplot(111, frame_on=False) # no visible frame
#     ax = plt.gca()
#     ax.xaxis.set_visible(False)  # hide the x axis
#     ax.yaxis.set_visible(False)  # hide the y axis
#
#     # table(ax, df)  # where df is your data frame
#     table(ax, df)  # where df is your data frame
#
#     plt.show()


def data_frame_to_latex_with_backslashes(data_frame) -> str:
    return data_frame.to_latex(index=False, escape=False)


# - bar graph with error bars

def bar_graph_with_error_using_pandas(group_row_names: list,  # input to index. Usual good name is "groups"
                                      columns: list,
                                      rows: list[list],
                                      val_names: list[str],
                                      error_bar_names: list[str],
                                      title: str,
                                      xlabel: str,
                                      ylabel: str,

                                      linestyle='--',
                                      tight_layout=False,
                                      rotation_rows_group_names=0,
                                      show=False,
                                      kind='bar',
                                      alpha=0.7, capsize=2.5, width=0.15,
                                      loc='best',
                                      ):
    """
    ref:
        - https://stackoverflow.com/questions/23144784/plotting-error-bars-on-grouped-bars-in-pandas/71179448#71179448
    """
    import pandas as pd

    assert len(group_row_names) == len(rows), f'One name for each row, so they must have same size. But got: ' \
                                              f'{len(group_row_names)=},{len(rows)=}'
    # create data matrix, size [# rows, # columns]
    for row in rows:
        assert len(row) == len(columns), f'each value for a row corresponds to a column. So they need to match in size' \
                                         f'But got {len(row)=},{len(columns)=}.'
    data = rows

    #
    df = pd.DataFrame(data, columns=columns, index=group_row_names)

    #
    yerr = df[error_bar_names].to_numpy().T  # e.g. yerr = df[['MAML5 ci', 'MAML10 ci', 'USL ci']].to_numpy().T

    df[val_names].plot(kind=kind, yerr=yerr, alpha=alpha, capsize=capsize, width=width)
    plt.grid(linestyle=linestyle)
    plt.tight_layout() if tight_layout else None
    plt.xticks(rotation=rotation_rows_group_names)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim([0, 100.0])
    # plt.legend(loc='best')
    # plt.legend(loc="upper left", bbox_to_anchor=(0.5, 1.15), ncol=2)
    # plt.legend(loc="lower left", bbox_to_anchor=(0.5, -0.15), ncol=2)
    # plot.legend(loc=2, prop={'size': 10000})
    # plt.legend(fontsize="x-small")
    plt.legend(loc=loc)

    plt.show() if show else None


# - examples

def bar_graph_with_eroror_bars_full_example():
    groups = ['MI, 5CNN', 'MI, 5CNN', 'MI, 5CNN', 'MI 5CNN']  # the rows of a df
    adapted_models = ['MAML5', 'MAML10', 'USL', 'MAML5 ci', 'MAML10 ci', 'USL ci']  # columns of a df
    meta_test_acc = [62.4, 62.3, 60.1]
    meta_test_ci = [1.64, 1.5, 1.37]
    row1 = meta_test_acc + meta_test_ci
    row2 = meta_test_acc + meta_test_ci
    row3 = meta_test_acc + meta_test_ci
    row4 = meta_test_acc + meta_test_ci
    data = [row1, row2, row3, row4]

    bar_graph_with_error_using_pandas(group_row_names=groups,
                                      columns=adapted_models,
                                      rows=data,
                                      val_names=adapted_models[0:3],
                                      error_bar_names=adapted_models[3:],
                                      title='Performance Comparsion MAML vs TL',
                                      xlabel='Dataset, Architecture',
                                      ylabel='Meta-Test Accuracy'
                                      )
    plt.show()


def bar_graph_with_error_bars():
    import pandas as pd
    import matplotlib.pyplot as plt
    from numpy import array

    df = pd.DataFrame([[4, 6, 1, 3], [5, 7, 5, 2]], columns=['mean1', 'mean2', 'std1', 'std2'], index=['A', 'B'])

    # convert the std columns to an array
    yerr = df[['std1', 'std2']].to_numpy().T

    # print(yerr)
    array([[1, 5],
           [3, 2]], dtype=int)

    df[['mean1', 'mean2']].plot(kind='bar', yerr=yerr, alpha=0.5, error_kw=dict(ecolor='k'), capsize=5.0)
    plt.show()


def bar_graph_with_error_bars1():
    import pandas as pd
    import matplotlib.pyplot as plt
    from numpy import array

    groups = ['MI 5CNN']
    adapted_models = ['MAML5', 'MAML10', 'USL', 'MAML5 ci', 'MAML10 ci', 'USL ci']
    meta_test_acc = [62.4, 62.3, 60.1]
    meta_test_ci = [1.64, 1.5, 1.37]
    row1 = array([meta_test_acc + meta_test_ci])
    data = row1

    df = pd.DataFrame(data, columns=adapted_models, index=groups)
    print(df)

    # convert the std columns to an array
    yerr = df[['MAML5 ci', 'MAML10 ci', 'USL ci']].to_numpy().T
    print(yerr)

    # df[['MAML5', 'MAML10', 'USL']].plot(kind='bar', yerr=yerr, alpha=0.5, error_kw=dict(ecolor='k'), capsize=5.0)
    df[['MAML5', 'MAML10', 'USL']].plot(kind='bar', yerr=yerr, alpha=0.7, capsize=5.0, width=0.08)
    # plt.grid(True)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.xticks(rotation=0)
    plt.show()


def bar_graph_with_error_bars2():
    # %%

    import pandas as pd
    import matplotlib.pyplot as plt

    groups = ['MI 5CNN', 'MI 5CNN 2']  # the rows of a df
    adapted_models = ['MAML5', 'MAML10', 'USL', 'MAML5 ci', 'MAML10 ci', 'USL ci']  # columns of a df
    meta_test_acc = [62.4, 62.3, 60.1]
    meta_test_ci = [1.64, 1.5, 1.37]
    row1 = meta_test_acc + meta_test_ci
    row2 = meta_test_acc + meta_test_ci
    data = [row1, row2]
    print(data)

    df = pd.DataFrame(data, columns=adapted_models, index=groups)
    print(df)

    # convert the std columns to an array
    yerr = df[['MAML5 ci', 'MAML10 ci', 'USL ci']].to_numpy().T
    print(yerr)

    # df[['MAML5', 'MAML10', 'USL']].plot(kind='bar', yerr=yerr, alpha=0.5, error_kw=dict(ecolor='k'), capsize=5.0)
    df[['MAML5', 'MAML10', 'USL']].plot(kind='bar', yerr=yerr, alpha=0.7, capsize=2.5, width=0.15)
    # plt.grid(True)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.xticks(rotation=0)
    plt.show()


def my_example_seaborn_error_bands_example():
    """
    An example of how to plot my [B, L] for each metric.

    https://stackoverflow.com/questions/69888181/how-to-show-error-bands-for-pure-matrices-samples-x-range-with-seaborn-error
    https://seaborn.pydata.org/generated/seaborn.lineplot.html

    https://seaborn.pydata.org/tutorial/relational.html#relational-tutorial

    https://seaborn.pydata.org/examples/errorband_lineplots.html
    https://www.youtube.com/watch?v=G3F0EZcW9Ew
    https://github.com/knathanieltucker/seaborn-weird-parts/commit/3e571fd8e211ea04b6c9577fd548e7e532507acf
    https://github.com/knathanieltucker/seaborn-weird-parts/blob/3e571fd8e211ea04b6c9577fd548e7e532507acf/tsplot.ipynb

    x, yvectors or keys in data
    Variables that specify positions on the x and y axes.

    huevector or key in data
    Grouping variable that will produce lines with different colors. Can be either categorical or numeric, although color mapping will behave differently in latter case
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
    sns.lineplot(x="timepoint", y="signal", hue="region", style="event", data=fmri)
    plt.show()


def custom_errorbands_without_seaborn_example():
    """
    pure matplot lib error bands.

    https://stackoverflow.com/questions/55368485/draw-error-shading-bands-on-line-plot-python?noredirect=1&lq=1
    """

    import numpy as np  # v 1.19.2
    import matplotlib.pyplot as plt  # v 3.3.2

    rng = np.random.default_rng(seed=1)

    x = np.linspace(0, 5 * np.pi, 50)
    y = np.sin(x)
    # error = np.random.normal(0.1, 0.02, size=x.shape) # I leave this out
    nb_yfuncs = 25
    ynoise = rng.normal(1, 0.1, size=(nb_yfuncs, y.size))
    yfuncs = nb_yfuncs * [y] + ynoise

    # fig, ax = plt.subplots(figsize=(10,4))
    # for yfunc in yfuncs:
    #     plt.plot(x, yfunc, 'k-')
    #
    # plt.show()

    ymean = yfuncs.mean(axis=0)
    ymin = yfuncs.min(axis=0)
    ymax = yfuncs.max(axis=0)
    yerror = np.stack((ymean - ymin, ymax - ymean))

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.fill_between(x, ymin, ymax, alpha=0.2, label='error band')
    plt.errorbar(x, ymean, yerror, color='tab:blue', ecolor='tab:blue',
                 capsize=3, linewidth=1, label='mean with error bars')
    plt.legend()

    plt.show()


def seaborn_multiple_curves_with_only_matrices_example():
    """
    Plot multple curves with error bands only having data matrices of the samples [num_samples, x_values_num_in_range] and
    x values in a given range with x_values_num_in_range as the number of x-values in that range.

    Ref:
        - https://stackoverflow.com/questions/69888181/how-to-show-error-bands-for-pure-matrices-samples-x-range-with-seaborn-error/69889619#69889619
    """
    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt

    print(sns)

    np.random.seed(22)
    sns.set(color_codes=True)

    # the number of x values to consider in a given range e.g. [0,1] will sample 10 raw features x sampled at in [0,1] interval
    num_x: int = 10
    # the repetitions for each x feature value e.g. multiple measurements for sample x=0.0 up to x=1.0 at the end
    rep_per_x: int = 5
    total_size_data_set: int = num_x * rep_per_x
    print(f'{total_size_data_set=}')
    # - create fake data set
    # only consider 10 features from 0 to 1
    x = np.linspace(start=0.0, stop=1.0, num=num_x)

    # to introduce fake variation add uniform noise to each feature and pretend each one is a new observation for that feature
    noise_uniform: np.ndarray = np.random.rand(rep_per_x, num_x)
    # same as above but have the noise be the same for each x (thats what the 1 means)
    noise_normal: np.ndarray = np.random.randn(rep_per_x, 1)
    # signal function
    sin_signal: np.ndarray = np.sin(x)
    cos_signal: np.ndarray = np.cos(x)
    # [rep_per_x, num_x]
    y1: np.ndarray = sin_signal + noise_uniform + noise_normal
    y2: np.ndarray = cos_signal + noise_uniform + noise_normal

    # - since seaborn expects a an x value paired with it's y value, let's flatten the y's and make sure the corresponding
    # x value is alined with it's y value.
    x: np.ndarray = np.tile(x,
                            rep_per_x)  # np.tile = Construct an array by repeating A the number of times given by reps.
    y1: np.ndarray = np.ravel(y1)  # flatten the y's to match the x values to have the x to it's corresponding y
    y2: np.ndarray = np.ravel(y2)  # flatten the y's to match the x values to have the x to it's corresponding y

    # - plot
    err_style = 'band'
    # err_style = 'bars'
    ax = sns.lineplot(x=x, y=y1, err_style=err_style, label='sin', marker='x', dashes=False)
    ax = sns.lineplot(x=x, y=y2, err_style=err_style, label='cos', marker='x', dashes=False)
    plt.title('Sin vs Cos')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def xticks():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.array([0, 1, 2, 3])
    y = np.array([20, 21, 22, 23])
    my_xticks = ['John', 'Arnold', 'Mavis', 'Matt']
    plt.xticks(x, my_xticks)
    plt.plot(x, y)
    plt.show()


def table_example():
    import pandas as pd

    data = {
        'Initialization': ['Random',
                           'Random2',
                           ],

        'Test Accuracy': ['0.200+-0.029',
                          '0.200+-0.0',
                          ],
    }

    # - to pandas table
    df = pd.DataFrame(data)
    print(df)

    # https://stackoverflow.com/questions/70009242/how-does-one-generate-latex-table-images-with-proper-equations-from-python-panda

    # - to latex,
    # idea is to have it initially print a table and then custumize it manually
    # https://www.overleaf.com/learn/latex/Tables#Creating_a_simple_table_in_LaTeX
    data = put_pm_to_pandas_data(data)
    df = pd.DataFrame(data)

    print()
    # column_format = ''.join(['c' for k in data.keys()])
    # print(df.to_latex(index=False, escape=False, column_format=column_format))
    # print(df.to_latex(index=False, escape=False, caption='caption goes here', label='label_goes_here'))
    print(get_latex_table_as_text_nice_default(df))


# - tests

def save_plot_test():
    """
    Should show the plot and also save three none-empty plot.
    """
    x = [1, 2, 3]
    y = [2, 3, 4]
    plot(x, y, 'x', 'y', show=True, save_plot=True)


def plot_seaborn_curve_with_x_values_y_values_test():
    # the number of x values to consider in a given range e.g. [0,1] will sample 10 raw features x sampled at in [0,1] interval
    num_x: int = 10
    # the repetitions for each x feature value e.g. multiple measurements for sample x=0.0 up to x=1.0 at the end
    rep_per_x: int = 5
    total_size_data_set: int = num_x * rep_per_x
    print(f'{total_size_data_set=}')
    # - create fake data set
    # only consider 10 features from 0 to 1
    x = np.linspace(start=0.0, stop=1.0, num=num_x)

    # to introduce fake variation add uniform noise to each feature and pretend each one is a new observation for that feature
    noise_uniform: np.ndarray = np.random.rand(rep_per_x, num_x)
    # same as above but have the noise be the same for each x (thats what the 1 means)
    noise_normal: np.ndarray = np.random.randn(rep_per_x, 1)
    # signal function
    sin_signal: np.ndarray = np.sin(x)
    cos_signal: np.ndarray = np.cos(x)
    # [rep_per_x, num_x]
    y1: np.ndarray = sin_signal + noise_uniform + noise_normal
    y2: np.ndarray = cos_signal + noise_uniform + noise_normal

    plot_seaborn_curve_with_x_values_y_values(x=x, y=y1, xlabel='x', ylabel='y', title='Sin vs Cos')
    plot_seaborn_curve_with_x_values_y_values(x=x, y=y2, xlabel='x', ylabel='y', title='Sin vs Cos')
    plt.show()


def plot_with_error_bands_test():
    import numpy as np  # v 1.19.2
    import matplotlib.pyplot as plt  # v 3.3.2

    # the number of x values to consider in a given range e.g. [0,1] will sample 10 raw features x sampled at in [0,1] interval
    num_x: int = 30
    # the repetitions for each x feature value e.g. multiple measurements for sample x=0.0 up to x=1.0 at the end
    rep_per_x: int = 5
    total_size_data_set: int = num_x * rep_per_x
    print(f'{total_size_data_set=}')
    # - create fake data set
    # only consider 10 features from 0 to 1
    x = np.linspace(start=0.0, stop=2 * np.pi, num=num_x)

    # to introduce fake variation add uniform noise to each feature and pretend each one is a new observation for that feature
    noise_uniform: np.ndarray = np.random.rand(rep_per_x, num_x)
    # same as above but have the noise be the same for each x (thats what the 1 means)
    noise_normal: np.ndarray = np.random.randn(rep_per_x, 1)
    # signal function
    sin_signal: np.ndarray = np.sin(x)
    cos_signal: np.ndarray = np.cos(x)
    # [rep_per_x, num_x]
    y1: np.ndarray = sin_signal + noise_uniform + noise_normal
    y2: np.ndarray = cos_signal + noise_uniform + noise_normal

    y1mean = y1.mean(axis=0)
    y1err = y1.std(axis=0)
    y2mean = y2.mean(axis=0)
    y2err = y2.std(axis=0)

    plot_with_error_bands(x=x, y=y1mean, yerr=y1err, xlabel='x', ylabel='y', title='Custom Seaborn')
    plot_with_error_bands(x=x, y=y2mean, yerr=y2err, xlabel='x', ylabel='y', title='Custom Seaborn')
    plt.show()


def plot_with_error_bands_xticks_test():
    import numpy as np  # v 1.19.2
    import matplotlib.pyplot as plt  # v 3.3.2

    # the number of x values to consider in a given range e.g. [0,1] will sample 10 raw features x sampled at in [0,1] interval
    num_x: int = 5
    # the repetitions for each x feature value e.g. multiple measurements for sample x=0.0 up to x=1.0 at the end
    rep_per_x: int = 5
    total_size_data_set: int = num_x * rep_per_x
    print(f'{total_size_data_set=}')
    # - create fake data set
    # only consider 10 features from 0 to 1
    x = np.linspace(start=0.0, stop=2 * np.pi, num=num_x)

    # to introduce fake variation add uniform noise to each feature and pretend each one is a new observation for that feature
    noise_uniform: np.ndarray = np.random.rand(rep_per_x, num_x)
    # same as above but have the noise be the same for each x (thats what the 1 means)
    noise_normal: np.ndarray = np.random.randn(rep_per_x, 1)
    # signal function
    sin_signal: np.ndarray = np.sin(x)
    cos_signal: np.ndarray = np.cos(x)
    # [rep_per_x, num_x]
    y1: np.ndarray = sin_signal + noise_uniform + noise_normal
    y2: np.ndarray = cos_signal + noise_uniform + noise_normal

    y1mean = y1.mean(axis=0)
    y1err = y1.std(axis=0)
    y2mean = y2.mean(axis=0)
    y2err = y2.std(axis=0)

    x_vals_as_symbols: list[str] = [f'Val{v:0.2f}' for v in x]
    plot_with_error_bands(x=x, y=y1mean, yerr=y1err, xlabel='x', ylabel='y', title='Custom Seaborn',
                          x_vals_as_symbols=x_vals_as_symbols)
    plot_with_error_bands(x=x, y=y2mean, yerr=y2err, xlabel='x', ylabel='y', title='Custom Seaborn',
                          x_vals_as_symbols=x_vals_as_symbols)
    plt.show()


if __name__ == '__main__':
    # save_plot_test()
    # default_seabron_example()
    # plot_seaborn_curve_with_x_values_y_values_test()
    # plot_with_error_bands_test()
    # plot_with_error_bands_xticks_test()
    bar_graph_with_eroror_bars_full_example()
    print('Done, success! \a')
