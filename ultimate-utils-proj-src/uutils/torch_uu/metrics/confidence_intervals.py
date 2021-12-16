"""
Review for confidence intervals. Confidence intervals say that the true mean is inside the estimated confidence interval
(the r.v. the user generates). In particular it says:
    Pr[mu^* \in [mu_n +- t.val(p) * std_n / sqrt(n) ] ] >= p
e.g. p = 0.95
This does not say that for a specific CI you compute the true mean is in that interval with prob 0.95. Instead it means
that if you surveyed/sampled 100 data sets D_n = {x_i}^n_{i=1} of size n (where n is ideally >=30) then for 95 of those
you'd expect to have the truee mean inside the CI compute for that current data set. Note you can never check for which
ones mu^* is in the CI since mu^* is unknown. If you knew mu^* you wouldn't need to estimate it. This analysis assumes
that the the estimator/value your estimating is the true mean using the sample mean (estimator). Since it usually uses
the t.val or z.val (second for the standardozed r.v. of a normal) then it means the approximation that mu_n ~ gaussian
must hold. This is most likely true if n >= 0. Note this is similar to statistical learning theory where we use
the MLE/ERM estimator to choose a function with delta, gamma etc reasoning. Note that if you do algebra you can also
say that the sample mean is in that interval but wrt mu^* but that is borning, no one cares since you do not know mu^*
so it's not helpful.

An example use could be for computing the CI of the loss (e.g. 0-1, CE loss, etc). The mu^* you want is the expected
risk. So x_i = loss(f(x_i), y_i) and you are computing the CI for what is the true expected risk for that specific loss
function you choose. So mu_n = emperical mean of the loss and std_n = (unbiased) estimate of the std and then you can
simply plug in the values.

ref:
    - https://stats.stackexchange.com/questions/554332/confidence-interval-given-the-population-mean-and-standard-deviation?noredirect=1&lq=1
    - https://stackoverflow.com/questions/70356922/what-is-the-proper-way-to-compute-95-confidence-intervals-with-pytorch-for-clas
    - https://www.youtube.com/watch?v=MzvRQFYUEFU&list=PLUl4u3cNGP60hI9ATjSFgLZpbNJ7myAg6&index=205

wontfix:
    - make it differentiable wrt confidence (though not really needed just for fun)
"""
import numpy as np
import scipy
import torch
from torch import Tensor

# P_CI = {0.90: 1.64,
#         0.95: 1.96,
#         0.98: 2.33,
#         0.99: 2.58,
#         }


def mean_confidence_interval(data, confidence: float = 0.95):
    """
    Computes the confidence interval for a given survey of a data set.

    ref:
        - https://stackoverflow.com/a/15034143/1601580
        - https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/eval/meta_eval.py#L19
    """
    import scipy.stats

    a = 1.0 * data
    n = len(a)
    m, se = a.mean(), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    # return m, m - h, m + h
    return m, h


def torch_compute_confidence_interval_classification(data: Tensor,
                                                     confidence: float = 0.95
                                                     ) -> tuple[Tensor, Tensor]:
    """
    Computes the confidence interval for a given survey of a data set.

    Ref:
        - https://stats.stackexchange.com/a/556302/28986
    """
    B: int = data.size(0)
    # z_p: float = P_CI[confidence]
    t_p: float = scipy.stats.t.ppf((1 + confidence) / 2., B - 1)
    error: Tensor = data.mean()
    margin_of_error = torch.sqrt((error * (1 - error)) / B)
    # ci_interval: Tensor = z_p * margin_of_error
    ci_interval: Tensor = t_p * margin_of_error
    return error, ci_interval


def torch_compute_confidence_interval(data: Tensor,
                                           confidence: float = 0.95
                                           ) -> Tensor:
    """
    Computes the confidence interval for a given survey of a data set.
    """
    n = len(data)
    mean: Tensor = data.mean()
    # se: Tensor = scipy.stats.sem(data)  # compute standard error
    # se, mean: Tensor = torch.std_mean(data, unbiased=True)  # compute standard error
    se: Tensor = data.std(unbiased=True) / (n**0.5)
    t_p: float = float(scipy.stats.t.ppf((1 + confidence) / 2., n - 1))
    ci = t_p * se
    return mean, ci

# - tests

def ci_test():
    from uutils.torch_uu import approx_equal
    n: int = 30
    # n: int = 1000  # uncomment to see how it gets more accurate
    x_bernoulli: Tensor = torch.torch.randint(0, 2, [n]).float()
    mean, ci_95 = mean_confidence_interval(x_bernoulli, confidence=0.95)
    mean, ci_95_cls = torch_compute_confidence_interval_classification(x_bernoulli, confidence=0.95)
    mean, ci_95_anything = torch_compute_confidence_interval(x_bernoulli, confidence=0.95)
    print(f'{x_bernoulli.std()=}')
    print(f'{ci_95=}')
    print(f'{ci_95_cls=}')
    print(f'{ci_95_anything=}')
    assert approx_equal(ci_95, ci_95_cls, tolerance=1e-2)
    assert approx_equal(ci_95, ci_95_anything, tolerance=1e-2)

    x_bernoulli: Tensor = torch.torch.randint(0, 2, [n]).float()
    x_bernoulli.requires_grad = True
    mean, ci_95_torch = torch_compute_confidence_interval_classification(x_bernoulli, confidence=0.95)
    print(f'{x_bernoulli.std()=}')
    print(f'{ci_95_torch=}')


def ci_test_regression():
    from uutils.torch_uu import approx_equal
    n: int = 30
    x: Tensor = torch.randn(n) - 10
    mean, ci_95 = mean_confidence_interval(x, confidence=0.95)
    mean, ci_95_torch = torch_compute_confidence_interval(x, confidence=0.95)
    print(f'{x.std()=}')
    print(f'{ci_95=}')
    print(f'{ci_95_torch=}')
    assert approx_equal(ci_95, ci_95_torch, tolerance=1e-2)

    x: Tensor = torch.randn(n, requires_grad=True) - 10
    mean, ci_95_torch = torch_compute_confidence_interval(x, confidence=0.95)
    print(f'{x.std()=}')
    print(f'{ci_95_torch=}')


if __name__ == '__main__':
    ci_test()
    ci_test_regression()
    print('Done, success! \a')
