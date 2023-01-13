#%%
"""
A very small p-value means that such an extreme observed outcome would be very unlikely under the null hypothesis.
- One sided (consider y is sample mean):
    p = Pr[Y >= y]

Decision Rule: https://online.stat.psu.edu/statprogram/reviews/statistical-concepts/hypothesis-testing/p-value-approach
    - If the P-value is less than (or equal to) alpha, then the null hypothesis is rejected in favor of the alternative
    hypothesis. And, if the P-value is greater than , then the null hypothesis is not rejected.
    - Set the significance level, alpha, the probability of making a Type I error to be small — 0.01, 0.05, or 0.10.
    Compare the P-value to alpha. If the P-value is less than (or equal to) alpha, reject the null hypothesis in
    favor of the alternative hypothesis. If the P-value is greater than , do not reject the null hypothesis.

ref:
    - https://en.wikipedia.org/wiki/P-value
    - https://online.stat.psu.edu/statprogram/reviews/statistical-concepts/hypothesis-testing/p-value-approach
"""

import numpy as np
from scipy import stats

# Sample data
# sample 100 values using numpy from a normal with mean 5 and std 10
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

#%%
# mu_x, mu_y = 5, 5
# std_x, std_y = 10, 100
# x = np.random.normal(mu_x, std_x, 1000)
# y = np.random.normal(mu_y, std_y, 1000)
#
# # should/might make a mistake since the stds are very different but it assume they are equal (or at least the p value is different from the one bellow)
# t, p = stats.ttest_ind(x, y, equal_var=True)
# print(f'{p=} {t=}')
# print(f'Statistically significant? {p < 0.05=}')
#
# # should NOT mistake since the stds are very different but it assume they are equal (or at least the p value is different from the one bellow)
# t, p = stats.ttest_ind(x, y, equal_var=False)
# print(p)
# print(f'Statistically significant? {p < 0.05=}')
