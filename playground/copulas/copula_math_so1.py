# %%
"""
https://chat.stackexchange.com/rooms/133701/discussion-between-charlie-parker-and-snoop
"""

import numpy
import scipy.stats
import matplotlib.pyplot as mp

I = 10000  # number of sample pairs
rho = 0.9  # correlation
cov = numpy.array([[1, rho],
                   [rho, 1]])  # covariance matrix
mu = numpy.array([2, -5])  # means
X = scipy.stats.multivariate_normal.rvs(size=I, mean=mu, cov=cov)  # observed samples
excdf1 = scipy.stats.norm.cdf(X[:, 0], loc=mu[0])  # transformations using marginals
excdf2 = scipy.stats.norm.cdf(X[:, 1], loc=mu[1])  #

mp.plot(excdf1, excdf2, '.')  # copula
mp.show()
