import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt


# -

def show_n_does_or_doesnt_affect_p_value():
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
        t, p = stats.ttest_ind(x, y, equal_var=True)
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


if __name__ == '__main__':
    print('Starting...')
    show_n_does_or_doesnt_affect_p_value()
    print(f'Done!\a\n')
