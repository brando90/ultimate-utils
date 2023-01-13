"""
Goal: sanity check. As N->30, 100, 10_000, the p-value should be the same.
"""
import numpy as np
from scipy import stats

# Sample data
mu_x, mu_y = 0, 10  # should reject null hypothesis i.e. statistically significant
mu_x, mu_y = 5, 5  # should accept null hypothesis i.e. not statistically significant
x = np.random.normal(mu_x, 10, 1000)
y = np.random.normal(mu_y, 10, 1000)

# Perform t-test
t, p = stats.ttest_ind(x, y)
print(f'{p=} (Probability that this sample mean or more extreme is observerd)')
print(f'{t=} (t-statistic, the difference btw the sample means in units of the sample standard deviation)')
# Statistically significant means we can reject the null hypothesis (& accept our own e.g. means are different)
# so its statistically significant if the observation or more extreme under H0 is very unlikely i.e. p < 0.05
print(f'Statistically significant? (i.e. can we reject H0, mean isn\'t zero): {p < 0.05=}.')
if p < 0.05:
    print(f'p < 0.05 so we can reject H0, means are statistically different.')
else:
    print(f'p > 0.05 so we can\'t reject H0, means are statistically the same.')