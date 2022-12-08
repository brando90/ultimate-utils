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

Assumptions for p-CI:
    - we are making a statement that mu^* is in mu+-pCI = mu+-t_p * sig_n / sqrt n, sig_n ~ Var[x] is inside the CI
    p% of the time.
    - we are estimating mu^, a mean
    - since the quantity of interest is mu^, then the z_p value (or p-value, depending which one is the unknown), is
    computed using the normal distribution.
    - p(mu) ~ N(mu; mu_n, sig_n/ sqrt n), vial CTL which holds for sample means. Ideally n >= 30.
    - x ~ p^*(x) are iid.

Std_n vs t_p*std_n/ sqrt(n)
    - std_n = var(x) is more pessimistic but holds always. Never shrinks as n->infity
    - but if n is small then pCI might be too small and your "lying to yourself". So if you have very small data
    perhaps doing std_n for the CI is better. That holds with prob 99.9%. Hopefuly std is not too large for your
    experiments to be invalidated.

ref:
    - https://stats.stackexchange.com/questions/554332/confidence-interval-given-the-population-mean-and-standard-deviation?noredirect=1&lq=1
    - https://stackoverflow.com/questions/70356922/what-is-the-proper-way-to-compute-95-confidence-intervals-with-pytorch-for-clas
    - https://www.youtube.com/watch?v=MzvRQFYUEFU&list=PLUl4u3cNGP60hI9ATjSFgLZpbNJ7myAg6&index=205

todo:
    - how to do CI for meta-learning, for now std is fine, since it's usually a very worst case estimate.
    Previous work does CI so should we do CI too? No do std, since that is a worser case analysis (and holds with 99.9
    of the time). Drawback, interval doesn't shrink as we have more data. But this analysis makes the entire net
    analysis harder, perhaps try CI only on entire net analysis to see what happens.
"""
from typing import Union

import numpy as np
import scipy
import torch
from torch import Tensor
import scipy.stats

# P_CI = {0.90: 1.64,
#         0.95: 1.96,
#         0.98: 2.33,
#         0.99: 2.58,
#         }
# t_p=1.6991270265334972 at n=30
# t_p=2.045229642132703 at n=30
# t_p=2.4620213601503833 at n=30
# t_p=2.756385903670335 at n=30
#
# t_p=1.6603911559963895 at n=100
# t_p=1.9842169515086827 at n=100
# t_p=2.3646058614359737 at n=100
# t_p=2.6264054563851857 at n=100
"""
Conclusion: yes, using t(p) raises the multiplier of the ci in ci=t(p)*std/sqrt(n). 
"""


def compute_ci(std: float, confidence: float, n: int) -> np.ndarray:
    """
    Computes confidence interval:
        ci = z(p) * std/ sqrt(n)
    """
    tp: float = scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    se: float = std / (n ** 0.5)
    ci: float = tp * se
    return ci


def nth_central_moment_and_its_confidence_interval(data: np.ndarray,
                                                   moment_idx: int,
                                                   confidence: float = 0.95,
                                                   centered: bool = True,
                                                   ) -> tuple[float, np.ndarray]:
    """
    Compute the nth central moment and it's confidence interval.

    ci formula: z(p) * s / sqrt(n)
    Central moments are computed using mom_n = E[X^n]. To compute their CI we need their variance which is the next next moment
    i.e. var_mom_n = E[X^2n] so std_mom_n = sqrt(E[X^2n]) = sqrt(var_mom_n) which can be then plugged in to the
    ci formula as normal.
    """
    import numpy as np
    from scipy.stats import moment

    a: np.ndarray = 1.0 * np.array(data)
    n: int = len(a)

    if centered:
        mom: float = moment(a=data, moment=moment_idx)
        var_mom: float = moment(a=data, moment=2 * moment_idx)
    else:
        from uutils.numpy_uu.common import compute_uncentered_moments
        mom: float = compute_uncentered_moments(a=data, moment_idx=moment_idx)
        var_mom: float = compute_uncentered_moments(a=data, moment_idx=2 * moment_idx)

    std: float = var_mom ** 0.5
    ci: np.ndarray = compute_ci(std, confidence, n)
    return mom, ci


def mean_confidence_interval(data, confidence: float = 0.95) -> tuple[float, np.ndarray]:
    """
    Returns (tuple of) the mean and confidence interval for given data.
    Data is a np.arrayable iterable.
    e.g.
        - list of floats [1.0, 1.3, ...]
        - tensor of size [B]

    note:
        - I think it can be used for any data because of the following: I believe it is fine since the mean and
        std are calculated for general numeric data and the z_p/t_p value only takes in the confidence interval and
        data size, so it is independent of assumptions on the distribution of data. So yes I think this equation can be
        used for both classification and regression.

    ref:
        - https://stackoverflow.com/a/15034143/1601580
        - https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/eval/meta_eval.py#L19
    """
    import scipy.stats
    import numpy as np

    a: np.ndarray = 1.0 * np.array(data)
    n: int = len(a)
    if n == 1:
        import logging
        logging.warning('The first dimension of your data is 1, perhaps you meant to transpose your data? or remove the'
                        'singleton dimension?')
    m, se = a.mean(), scipy.stats.sem(a)
    tp = scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    h = se * tp
    return m, h


# def mean_confidence_interval(data, confidence: float = 0.95) -> tuple[float, float]:
#     m, ci = mean_confidence_interval(data, confidence=confidence)
#     return m, ci[0]


def torch_compute_confidence_interval_classification(data: Tensor,
                                                     confidence: float = 0.95,
                                                     item: bool = False,  # setting to true might remove memory issues
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
    if item:
        error: float = error.item()
        ci_interval: float = ci_interval.item()
    return error, ci_interval


def torch_compute_confidence_interval(data: Tensor,
                                      confidence: float = 0.95,
                                      item: bool = False,  # setting to true might remove memory issues
                                      ) -> tuple[Union[Tensor, float], Union[Tensor, float]]:
    """
    Computes the confidence interval for a given survey of a data set.
    """
    n: int = len(data)
    mean: Tensor = data.mean()
    # se: Tensor = scipy.stats.sem(data)  # compute standard error
    # se, mean: Tensor = torch.std_mean(data, unbiased=True)  # compute standard error
    se: Tensor = data.std(unbiased=True) / (n ** 0.5)
    t_p: float = float(scipy.stats.t.ppf((1 + confidence) / 2., n - 1))
    ci = t_p * se
    if item:
        mean: float = mean.item()
        ci: float = ci.item()
    return mean, ci


# -

def prob_of_truth_being_inside_when_using_ci_as_std():
    """
    what is the probability my statement mu_n +- std using the bare std holds. According to this analysis it says that
        Pr[\theta^* \in [mu+- std] ] = int_{std, -std} N(x; mu, std/n**05) dx = int_{+1, -1} N(x; mu, 1/n**0.5) dx
    for n = 25, N(x;0, 1/n**05) = N(x; 0, 0.2) and the probability the true mean is in interval is really high around
    0.9999994266968563.

    todo: - Q: so why don't ppl just report mu +- std? it's more pessimistic and it has a higher chance of cantaining
        the true mean, assuming that is what your trying to "estimate"/bound. Possible due to not dividing by sqrt n,
        it might make most analysis "invalid" since this is a larger confidence interval, so this makes results more
        likely to clash - is my guess.
    """
    from scipy.integrate import quad
    # integration between x1 and x1
    # mean, std = 0.0, 1.0
    # mean, std = 0.0, 1/5.4772255
    n = 25
    mean, std = 0.0, 1 / (n ** 0.5)

    def normal_distribution_function(x):
        import scipy.stats
        value = scipy.stats.norm.pdf(x, mean, std)
        return value

    # x1 = mean - std
    # x2 = mean + std
    x1 = -1
    x2 = 1

    res, err = quad(func=normal_distribution_function, a=x1, b=x2)

    print('\nNormal Distribution (mean,std):', mean, std)
    print('Integration bewteen {} and {} --> '.format(x1, x2), res)


def confidence_intervals_intersect(pair0: tuple[float, float], pair1: tuple[float, float]) -> bool:
    """
    Tells you if the confidence intervals intersect. The first value in the tuple is the expected value and the second
    one is the confidence interval CI. Code will do
    ```python
        larger_adjusted: float = larger[0] - larger[1]
        smaller_adjusted: float = smaller[0] + smaller[1]
        # - they intersect if the larger became smaller and the smaller became larger
        intersect: bool = larger_adjusted < smaller_adjusted
    ```

    Examples:
        - to detect "equivalence" then the CI's intersect (and num tasks is high, since it's trivial for this to be
        true with N~small e.g. 1)
        - to detect statistically significant difference, then the CI's don't intersect (and num tasks is high)
    :param pair0:
    :param pair1:
    :return:
    """
    larger: float
    smaller: float
    if pair0[0] > pair1[0]:
        larger = pair0
        smaller = pair1
    else:
        larger = pair1
        smaller = pair0
    assert larger[0] > smaller[0], f'Err: {larger=}, {smaller=}'
    # - check intersection
    larger_adjusted: float = larger[0] - larger[1]
    smaller_adjusted: float = smaller[0] + smaller[1]
    # - they intersect if the larger became smaller and the smaller became larger
    intersect: bool = larger_adjusted < smaller_adjusted
    return intersect

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
    mean_torch, ci_95_torch = torch_compute_confidence_interval(x, confidence=0.95)
    print(f'{mean=}')
    print(f'{mean_torch=}')
    print(f'{x.std()=}')
    print(f'{ci_95=}')
    print(f'{ci_95_torch=}')
    assert approx_equal(ci_95, ci_95_torch, tolerance=1e-2)

    x: Tensor = torch.randn(n, requires_grad=True) - 10
    mean, ci_95_torch = torch_compute_confidence_interval(x, confidence=0.95)
    print(f'{x.std()=}')
    print(f'{ci_95_torch=}')


def print_tps():
    def print_tp(confidence: float, n: int):
        t_p: float = float(scipy.stats.t.ppf((1 + confidence) / 2., n - 1))
        print(f'{t_p=} at {n=}')

    n = 30
    print_tp(0.9, n)
    print_tp(0.95, n)
    print_tp(0.98, n)
    print_tp(0.99, n)
    n = 100
    print_tp(0.9, n)
    print_tp(0.95, n)
    print_tp(0.98, n)
    print_tp(0.99, n)


def ci_test_float():
    import numpy as np
    # - one WRONG data set of size 1 by N
    data = np.random.randn(1, 30)  # gives an error becuase len sets n=1, so not this shape!
    m, ci = mean_confidence_interval(data)
    print('-- you should get a mean and a list of nan ci (since data is in wrong format, it thinks its 30 data sets of '
          'length 1.')
    print(m, ci)

    # right data as N by 1
    data = np.random.randn(30, 1)
    m, ci = mean_confidence_interval(data)
    print('-- gives a mean and a list of length 1 for a single CI (since it thinks you have a single dat aset)')
    print(m, ci)

    # right data as N by 1
    data = list(np.random.randn(30, 1))
    m, ci = mean_confidence_interval(data)
    print('-- LIST! gives a mean and a list of length 1 for a single CI (since it thinks you have a single dat aset)')
    print(m, ci)

    # multiple data sets (7) of size N (=30)
    data = np.random.randn(30, 7)
    print('-- gives 7 CIs for the 7 data sets of length 30. 30 is the number ud want large if you were using z(p)'
          'due to the CLT.')
    m, ci = mean_confidence_interval(data)
    print(m, ci)


def moments_test():
    from uutils.numpy_uu.common import _my_compute_central_moment
    from uutils.torch_uu import approx_equal
    n: int = 500
    x: np.ndarray = np.random.normal(loc=0.0, scale=1.0, size=n)
    # -
    mom, ci_95_mom = nth_central_moment_and_its_confidence_interval(x, 0, confidence=0.95)
    my_mom = _my_compute_central_moment(x, 0)
    print(f'{mom=}')
    print(f'{my_mom=}')
    assert approx_equal(my_mom, mom, tolerance=1e-2)
    # -
    mean, ci_95 = mean_confidence_interval(x, confidence=0.95)
    mom, ci_95_mom = nth_central_moment_and_its_confidence_interval(x, 1, confidence=0.95)
    my_mom = _my_compute_central_moment(x, 1)
    print('--')
    print(f'{mean=} (not exactly zero because this is the actual mean of the data, not the mean of the centered data)')
    print(f'{mom=} (should be zero)')
    print(f'{my_mom=}')
    assert approx_equal(my_mom, mom, tolerance=1e-2)
    print(f'{ci_95=}')
    print(f'{ci_95_mom=}')
    # the assert is not quite right because we need to do the ci of the centered data but this shift doens't actually matter so it works
    assert approx_equal(ci_95, ci_95_mom, tolerance=1e-2)

    mom, ci_95_mom = nth_central_moment_and_its_confidence_interval(x, 2, confidence=0.95)
    my_mom = _my_compute_central_moment(x, 2)
    print(f'-- should be close to 1.0 since data has std 1.0')
    print(f'{mom=}')
    print(f'{my_mom=}')
    assert approx_equal(my_mom, mom, tolerance=1e-2)
    print(f'{ci_95_mom=}')

    mom, ci_95_mom = nth_central_moment_and_its_confidence_interval(x, 3, confidence=0.95)
    my_mom = _my_compute_central_moment(x, 3)
    print(f'--')
    print(f'{mom=}')
    print(f'{my_mom=}')
    assert approx_equal(my_mom, mom, tolerance=1e-2)
    print(f'{ci_95_mom=}')

    mom, ci_95_mom = nth_central_moment_and_its_confidence_interval(x, 4, confidence=0.95)
    my_mom = _my_compute_central_moment(x, 4)
    print(f'--')
    print(f'{mom=}')
    print(f'{my_mom=}')
    assert approx_equal(my_mom, mom, tolerance=1e-2)
    print(f'{ci_95_mom=}')


def statistically_significant_test():
    """
    For high div expts we need:
        - there is a difference btw the two accuracies
        - so the errors do not intersect

    :return:
    """
    pair1 = (90, 5)
    pair2 = (80, 1)
    assert not confidence_intervals_intersect(pair1, pair2)
    assert not confidence_intervals_intersect(pair2, pair1)
    print(f'{(pair1, pair2)=}')
    print(f'{confidence_intervals_intersect(pair1, pair2)=}')


    pair1 = (93, 3)
    pair2 = (89, 2)
    assert confidence_intervals_intersect(pair1, pair2)
    assert confidence_intervals_intersect(pair2, pair1)
    print(f'{(pair1, pair2)=}')
    print(f'{confidence_intervals_intersect(pair1, pair2)=}')


if __name__ == '__main__':
    # ci_test()
    # ci_test_regression()
    # prob_of_truth_being_inside_when_using_ci_as_std()
    # print_tps()
    # ci_test_float()
    # moments_test()
    statistically_significant_test()
    print('Done, success! \a')
