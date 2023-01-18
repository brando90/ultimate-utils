"""
Goal: sanity check. As N->30, 100, 10_000, the p-value should be the same.
"""
# %%
import numpy as np
from scipy import stats


def _print_statisticall_significant(p, alpha=0.05):
    """
    Statistically significant means we can reject the null hypothesis (& accept our own e.g. means are different)
    so its statistically significant if the observation or more extreme under H0 is very unlikely i.e. p < 0.05
    """
    print(f'Statistically significant? (i.e. can we reject H0, mean isn\'t zero): {p < alpha=}.')
    if p < alpha:
        # print(f'p < alpha so we can Reject H0, means are statistically different.')
        print('H1')
    else:
        # print(f'p > alpha so we can\'t reject H0, means are statistically the same.')
        print('H0')


# Sample data
mu_x, mu_y = 0, 10  # should reject null hypothesis i.e. statistically significant
# mu_x, mu_y = 5, 5  # should accept null hypothesis i.e. not statistically significant
N = 30
x = np.random.normal(mu_x, 10, N)
y = np.random.normal(mu_y, 12, N)

# perform t-test with different variances
t, p = stats.ttest_ind(x, y, equal_var=False)
print(f'{p=} (Probability that this sample mean or more extreme is observerd, also type I error)')
# print(f'{t=} (t-statistic, the difference btw the sample means in units of the sample standard deviation)')
_print_statisticall_significant(p)

N = 100
x = np.random.normal(mu_x, 10, N)
y = np.random.normal(mu_y, 12, N)

# perform t-test with different variances
t, p = stats.ttest_ind(x, y, equal_var=False)
print(f'{p=} (Probability that this sample mean or more extreme is observerd, also type I error)')
# print(f'{t=} (t-statistic, the difference btw the sample means in units of the sample standard deviation)')
_print_statisticall_significant(p)

N = 10_000
x = np.random.normal(mu_x, 10, N)
y = np.random.normal(mu_y, 12, N)

# perform t-test with different variances
t, p = stats.ttest_ind(x, y, equal_var=False)
print(f'{p=} (Probability that this sample mean or more extreme is observerd, also type I error)')
# print(f'{t=} (t-statistic, the difference btw the sample means in units of the sample standard deviation)')
_print_statisticall_significant(p)

"""
sample output:

p=3.58351141779882e-05 (Probability that this sample mean or more extreme is observerd, also type I error)
Statistically significant? (i.e. can we reject H0, mean isn't zero): p < alpha=True.
H1
p=8.333610732575984e-12 (Probability that this sample mean or more extreme is observerd, also type I error)
Statistically significant? (i.e. can we reject H0, mean isn't zero): p < alpha=True.
H1
p=0.0 (Probability that this sample mean or more extreme is observerd, also type I error)
Statistically significant? (i.e. can we reject H0, mean isn't zero): p < alpha=True.
H1

it was already tiny e-5 and it only decreased to -12 then 0.0 exactly. So the p-value is unaffected by N up to errors.
"""
