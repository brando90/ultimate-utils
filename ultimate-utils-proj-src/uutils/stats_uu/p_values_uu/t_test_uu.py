import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt


def decision_procedure_based_on_statistically_significance(p_value: float, alpha: float = 0.05):
    """
    Statistically significant means we can reject the null hypothesis (& accept our own e.g. means are different)
    so its statistically significant if the observation or more extreme under H0 is very unlikely i.e. p < 0.05.

    :arg p_value: p-value, probability that you observe the current your sample mean or more extreme under H0.
    :arg alpha: significance level.
    """
    print(f'Decision: (Statistically significant?)')
    if p_value <= alpha:
        print(f'H1 (Reject H0, means are statistically different) {p_value=}, {alpha=}, {p_value < alpha=}')
    else:
        print(f'H0 (Can\'t reject H0, means are statistically the same) {p_value=}, {alpha=}, {p_value < alpha=}')


# -

def show_n_does_or_doesnt_affect_p_value_for_t_test_test_():
    """
    plot p-value vs n. Does n affect p-value?
    """
    print('P-values is the probability that you observe the current your sample mean or more extreme under H0.')
    mu_x, mu_y = 0, 10  # should reject null hypothesis i.e. statistically significant
    std_x, std_y = 10, 12
    # ns: list[int] = [2, 10, 30, 100, 500, 1000, 5000, 10_000, 50_000, 100_000]
    ns: list[int] = [2, 10, 15, 30, 50, 75, 100, 150, 300]
    p_values: list[float] = []
    for n in ns:
        x = np.random.normal(mu_x, std_x, n)
        y = np.random.normal(mu_y, std_y, n)
        t, p = stats.ttest_ind(x, y, equal_var=False)
        print(f'{p=} (Probability that this sample mean or more extreme is observerd, also type I error)')
        p_values.append(p)
        from uutils.stats_uu.p_values_uu.common import print_statisticall_significant
        print_statisticall_significant(p)
    print(p_values)
    # plot p-values vs n, y axis is p value, x axis is n (sample size), smooth curve
    from uutils.plot import plot
    plot(ns, p_values, title='p-values vs n', xlabel='n (sample size)', ylabel='p-values')
    # plot horizonal line at alpha=0.05
    plt.axhline(y=0.05, color='r', linestyle='-')
    plt.show()

    # import pandas as pd
    # df = pd.DataFrame({'Num Filters': list(d.keys()), 'Num Params': list(d.values())})
    # print(df)
    # # - plot number of filters vs number of params title nums params vs num filters x labl num filters y label num params, using uutils
    # import matplotlib.pyplot as plt
    # from uutils.plot import plot
    # plot(num_filters, num_params, title='Number of Parameters vs Number of Filters', xlabel='Number of Filters',
    #      ylabel='Number of Parameters', marker='o', color='b')
    # plt.axhline(y=1.4e6, color='r', linestyle='-', label='ResNet12RFS (num params)')

def comparbale_metrics_for_different_means():
    import numpy as np
    mu_x, mu_y = 0.1, 0.23
    std_x, std_y = 0.018, 0.02
    # mu_x, mu_y = 0.5, 170_000_000_000
    # std_x, std_y = 5, 10_000_000_000
    # ns: list[int] = [2, 10, 30, 100, 500, 1000, 5000, 10_000, 50_000, 100_000]
    # ns: list[int] = [2, 10, 15, 30, 50, 75, 100, 150, 300]
    # p_values: list[float] = []
    # for n in ns:
    n = 30
    x = np.random.normal(mu_x, std_x, n)
    # y = np.random.normal(mu_y, std_y, n)
    # y = x
    # print(f'{x=}')
    # print(f'{y=}')
    # print(f'{x.mean()=}')
    # print(f'{y.mean()=}')
    # print(f'{x.std()=}')
    # print(f'{y.std()=}')
    # t, p = stats.ttest_ind(x, y, equal_var=False)
    # print(f'{t=}')

    import numpy as np
    from scipy.stats import zscore

    # data = np.array([1, 2, 3, 4, 5])
    data = x
    z = zscore(data)
    print(f'{z=}')
    print(f'{z.mean()=}')
    print(f'{z.std()=}')


if __name__ == '__main__':
    import time

    start = time.time()
    # - run experiment
    # main()
    comparbale_metrics_for_different_means()
    # - Done
    print(f'Done in {time.time() - start:.2f} seconds')
