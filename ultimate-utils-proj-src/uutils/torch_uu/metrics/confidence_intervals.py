"""
ref:
    - https://stackoverflow.com/questions/70356922/what-is-the-proper-way-to-compute-95-confidence-intervals-with-pytorch-for-clas
    - https://discuss.pytorch.org/t/what-is-the-proper-way-to-compute-95-confidence-intervals-with-pytorch-for-classification-and-regression/139398
"""
import numpy as np
import scipy
import torch
from torch import Tensor

P_CI = {0.90: 1.64,
        0.95: 1.96,
        0.98: 2.33,
        0.99: 2.58,
        }


def mean_confidence_interval_rfs(data, confidence=0.95):
    """
    https://stackoverflow.com/a/15034143/1601580
    """
    import scipy.stats

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats

    a = 1.0 * data
    n = len(a)
    m, se = a.mean(), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def torch_mean_confidence_interval(data: Tensor,
                                   confidence: float = 0.95,
                                   by_pass_30_data_points: bool = False
                                   ) -> Tensor:
    """

    todo: when is it fine to bypass 30? I mean the other code above doesn't check for that explcitly, so perhaps I won't
    either.
    """
    B: int = data.size(0)
    assert (data >= 0.0).all(), f'Data has to be positive for this CI to work but you have some negative value.'
    assert B >= 30 or by_pass_30_data_points, f' Not enough data for CI calc to be valid and approximate a' \
                                              f'normal, you have: {B=} but needed 30.'
    a = 1.0 * data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def ci(a, p=0.95):
    import numpy as np, scipy.stats as st
    st.t.interval(p, len(a) - 1, loc=np.mean(a), scale=st.sem(a))


# def ci(a, p=0.95):
#     import statsmodels.stats.api as sms
#
#     sms.DescrStatsW(a).tconfint_mean()

def compute_confidence_interval_classification(data: Tensor,
                                               by_pass_30_data_points: bool = False,
                                               p_confidence: float = 0.95
                                               ) -> Tensor:
    """
    Computes CI interval
        [B] -> [1]
    According to [1] CI the confidence interval for classification error can be calculated as follows:
        error +/- const * sqrt( (error * (1 - error)) / n)

    The values for const are provided from statistics, and common values used are:
        1.64 (90%)
        1.96 (95%)
        2.33 (98%)
        2.58 (99%)
    Assumptions:
    Use of these confidence intervals makes some assumptions that you need to ensure you can meet. They are:

    Observations in the validation data set were drawn from the domain independently (e.g. they are independent and
    identically distributed).
    At least 30 observations were used to evaluate the model.
    This is based on some statistics of sampling theory that takes calculating the error of a classifier as a binomial
    distribution, that we have sufficient observations to approximate a normal distribution for the binomial
    distribution, and that via the central limit theorem that the more observations we classify, the closer we will get
    to the true, but unknown, model skill.

    Ref:
        - computed according to: https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/

    todo:
        - how does it change for other types of losses
    """
    B: int = data.size(0)
    assert (data >= 0.0).all(), f'Data has to be positive for this CI to work but you have some negative value.'
    assert B >= 30 or by_pass_30_data_points, f' Not enough data for CI calc to be valid and approximate a' \
                                              f'normal, you have: {B=} but needed 30.'
    const: float = P_CI[p_confidence]
    error: Tensor = data.mean()
    val = torch.sqrt((error * (1 - error)) / B)
    print(val)
    ci_interval: float = const * val
    return ci_interval


def compute_confidence_interval_regression(data: Tensor,
                                           by_pass_30_data_points: bool = False,
                                           p_confidence: float = 0.95
                                           ) -> Tensor:
    """
    todo

    Note: this should be ok in practice since we only report errors for acc and compare with other work with ACC
    (or other metrics that are 0-1 e.g. nn similarities). So return var is fine most likely.
    :return:
    """
    # raise NotImplementedError
    B: int = data.size(0)
    assert (data >= 0.0).all(), f'Data has to be positive for this CI to work but you have some negative value.'
    assert B >= 30 or by_pass_30_data_points, f' Not enough data for CI calc to be valid and approximate a' \
                                              f'normal, you have: {B=} but needed 30.'
    ci_inverval: Tensor = data.var()
    return ci_inverval


# - tests

def ci_test():
    n: int = 25
    by_pass_30_data_points: bool = True

    x: Tensor = abs(torch.randn(n))
    ci_pytorch = compute_confidence_interval_classification(x, by_pass_30_data_points)
    ci_rfs = mean_confidence_interval(x)
    print(f'{x.var()=}')
    print(f'{ci_pytorch=}')
    print(f'{ci_rfs=}')

    x: Tensor = abs(torch.randn(n, requires_grad=True))
    ci_pytorch = compute_confidence_interval_classification(x, by_pass_30_data_points)
    # ci_rfs = mean_confidence_interval(x)
    print(f'{x.var()=}')
    print(f'{ci_pytorch=}')
    # print(f'{ci_rfs=}')

    x: Tensor = torch.randn(n) - 10
    ci_pytorch = compute_confidence_interval_classification(x, by_pass_30_data_points)
    ci_rfs = mean_confidence_interval(x)
    print(f'{x.var()=}')
    print(f'{ci_pytorch=}')
    print(f'{ci_rfs=}')

    print()


if __name__ == '__main__':
    ci_test()
    print('Done, success! \a')
