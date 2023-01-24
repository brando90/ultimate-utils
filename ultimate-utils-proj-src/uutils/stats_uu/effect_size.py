"""
Effect size = is a quantative measure of the strength of a phenomenon. Effect size emphasizes the size of the difference
    or relationship. e.g. the distance between the two means of two hyptoehsis H0, H1 (if they were Gaussian distributions).

Cohen’s d measures the difference between the mean from two Gaussian-distributed variables.
It is a standard score that summarizes the difference in terms of the number of standard deviations.
Because the score is standardized, there is a table for the interpretation of the result, summarized as:
    - Small Effect Size: d=0.20
    - Medium Effect Size: d=0.50
    - Large Effect Size: d=0.80

note:
    - you usually look up the effect size in you application/field (todo why)
    - depends on statistical test/hypothesis decision procedure (e.g. t-test, ANOVA, etc.).
    - large is good if you get a large effect size.
    - Q: why can't we control the effect size? (todo why) I think it's just a function of field/nature & your hypothesis
    - Q: how does it relate to number of samples? does it lead to the issues CIs lead e.g. Pf -> 1 Pm->0?
    - Courtney says, if the effect size is large, then a smaler sample size is needed. My guess is that n (sample size)
    should not affect negatively the effect size.
    - if effect size is small, then a larger sample size is needed (Courney says) (todo: why?)

ref:
    - https://www.youtube.com/watch?v=9LVD9oLg1A0&list=PLljPCllSdtzWop4iDNyyuZ2NZokOHeQqm&index=6
    - https://machinelearningmastery.com/effect-size-measures-in-python/


todo: later
    Two other popular methods for quantifying the difference effect size are:

    Odds Ratio. Measures the odds of an outcome occurring from one treatment compared to another.
    Relative Risk Ratio. Measures the probabilities of an outcome occurring from one treatment compared to another.
"""


# function to calculate Cohen's d for independent samples
def _cohen_d(d1, d2):
    """
    Compute Cohen's d for independent samples.

    ref:
        - from: https://machinelearningmastery.com/effect-size-measures-in-python/

```
# seed random number generator
seed(1)
# prepare data
data1 = 10 * randn(10000) + 60
data2 = 10 * randn(10000) + 55
# calculate cohen's d
d = cohend(data1, data2)
print('Cohens d: %.3f' % d)
```

    Note:
        -   ddof : int, optional
            Means Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            By default `ddof` is zero.
    """
    # calculate the Cohen's d between two samples
    from numpy.random import randn
    from numpy.random import seed
    from numpy import mean
    from numpy import var
    from math import sqrt

    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    cohen_d: float = (u1 - u2) / s
    return cohen_d


def compute_effect_size_t_test_cohens_d(group1: iter, group2: iter) -> tuple[float, float]:
    """
    Compute effect size for a t-test (Cohen's d). The distance in pooled std between the two means.

    Note:
        - The Cohen’s d calculation is not provided in Python; we can calculate it manually. ref: https://machinelearningmastery.com/effect-size-measures-in-python/
        -
    """
    from numpy import mean
    # calculate the pooled standard deviation
    pooled_std: float = compute_pooled_std_two_groups(group1, group2)
    # calculate the means of the samples
    u1, u2 = mean(group1), mean(group2)
    # calculate the effect size
    cohen_d: float = (u1 - u2) / pooled_std
    return cohen_d, pooled_std


def compute_pooled_std_two_groups(group1: iter, group2: iter, ddof: int = 1) -> float:
    """
    Compute pooled std for two groups.

    ref:
        - https://machinelearningmastery.com/effect-size-measures-in-python/
    """
    import numpy as np
    pooled_std: float = np.sqrt((np.std(group1, ddof=ddof) ** 2 + np.std(group2, ddof=ddof) ** 2) / 2)
    return pooled_std


def get_standardized_acceptable_difference(eps: float, group1: iter, group2: iter) -> float:
    """
    Get standardized acceptable difference/eps/epsilon. Recommended use: compare with effect size to see if your effect size is
    in the range of acceptable difference. e.g. if effect size is "low" but it's close to the acceptable difference,
    then there is an argument to accept your hypothesis (reject the null) i.e. there is a noticeable difference.
    """
    # calculate the pooled standard deviation
    pooled_std: float = compute_pooled_std_two_groups(group1, group2)
    # calculate the effect size
    standardized_acceptable_difference: float = eps / pooled_std
    return standardized_acceptable_difference


# - decicion procedures

def print_interpretation_of_cohends_d_decision_procedure(d: float):
    """ Print interpretation of Cohen's d decision procedure. """
    print(
        f'Decision: (is there a minimal difference between the means of two groups or a minimal relationship between two variables?)')
    d: float = abs(d)
    if d < 0.35:
        print(f"Small Effect Size: d~0.20, {d < 0.35=}, {d=}")
    elif 0.35 <= d < 0.65:
        print(f"Medium Effect Size: d~0.50, {0.35 <= d < 0.65=}, {d=}")
    elif d >= 0.65:
        print(f"Large Effect Size: d~0.80, {d >= 0.65=}, {d=}")
    else:
        raise ValueError(f"Unexpected value for d: {d}")


def decision_based_on_effect_size_and_standardized_acceptable_difference(
        effect_size: float,
        standardized_acceptable_difference: float) -> bool:
    """
    Function that accepts or rejects null hypothesis based on effect size and accepted without using confidence intervals.
    i.e. just compares the if the effect size is larger than the standardized acceptable difference.
    Logic/Justification: if people accept papers/knowledge based on the (standardized) acceptable difference then we
    can point out our difference is within that range and therefore we can "accept" our hypothesis (reject the null).

    Similar to the CIs test & CI tests are recommended to be also done

    Note:
        - eps - 1% or 2% based on ywx's suggestion (common acceptance for new method is SOTA in CVPR like papers).
    """
    effect_size: float = abs(effect_size)
    if effect_size >= standardized_acceptable_difference:
        print(f"H1 (Reject the null hypothesis, Effect size is larger than the standardized acceptable difference): "
              f"{effect_size >= standardized_acceptable_difference=}")
    else:
        print(f"H0 (Accept the null hypothesis, Effect size is smaller than the standardized acceptable difference): "
              f"{effect_size < standardized_acceptable_difference=}")
    return effect_size >= standardized_acceptable_difference  # True if H1 else False H0


# - tests

def my_test_using_stds_from_real_expts_():
    """
    std_maml = std_m ~ 0.061377
    std_usl = std_u ~ 0.085221
    """
    import numpy as np

    # Example data
    std_m = 0.061377
    std_u = 0.085221
    N = 100
    N_m = N
    N_u = N
    mu_m = 0.855
    mu_u = 0.893
    group1 = np.random.normal(mu_m, std_m, N_m)
    group2 = np.random.normal(mu_u, std_u, N_u)
    print(f'{std_m=}')
    print(f'{std_u=}')
    print(f'{N_m=}')
    print(f'{N_u=}')
    print(f'{mu_m=}')
    print(f'{mu_u=}')

    # Perform t-test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(group1, group2)

    # Compute Cohen's d
    print('---- effect size analysis ----')
    cohen_d, pooled_std = compute_effect_size_t_test_cohens_d(group1, group2)
    _cohen_d_val: float = _cohen_d(group1, group2)

    print("Cohen's d:", cohen_d)
    print("_cohen_d:", _cohen_d_val)

    # Compare with acceptable difference/epsilon
    acceptable_difference1: float = 0.01  # 1% difference/epsilon
    acceptable_difference2: float = 0.02  # 2% difference/epsilon
    standardized_acceptable_difference1: float = get_standardized_acceptable_difference(acceptable_difference1, group1,
                                                                                        group2)
    standardized_acceptable_difference2: float = get_standardized_acceptable_difference(acceptable_difference2, group1,
                                                                                        group2)
    # print(f'{acceptable_difference1=}')
    # print(f'{acceptable_difference2=}')
    print(f'{standardized_acceptable_difference1=}')
    print(f'{standardized_acceptable_difference2=}')
    decision_based_on_effect_size_and_standardized_acceptable_difference(cohen_d, standardized_acceptable_difference1)
    decision_based_on_effect_size_and_standardized_acceptable_difference(cohen_d, standardized_acceptable_difference2)
    print_interpretation_of_cohends_d_decision_procedure(cohen_d)

    # CIs/Confidence Intervals
    print('---- CIs/Confidence Intervals ----')
    from uutils.stats_uu.ci_uu import mean_confidence_interval, decision_based_on_acceptable_difference_cis
    m1, ci1 = mean_confidence_interval(group1)
    m2, ci2 = mean_confidence_interval(group2)
    decision_based_on_acceptable_difference_cis((m1, ci1), (m2, ci2), 0.0)
    decision_based_on_acceptable_difference_cis((m1, ci1), (m2, ci2), acceptable_difference1)
    decision_based_on_acceptable_difference_cis((m1, ci1), (m2, ci2), acceptable_difference2)

    # Print p-value
    print('---- p-value analysis ----')
    print("p-value: ", p_value)
    alpha: float = 0.01  # significance level
    from uutils.stats_uu.p_values_uu.t_test_uu import print_statistically_significant_decision_procedure
    print_statistically_significant_decision_procedure(p_value, alpha)

    # Print Power P_d (probability of detection, rejecting null if null is false)
    from uutils.stats_uu.power import _compute_power_ttest
    power: float = _compute_power_ttest(cohen_d, N, alpha)
    # from uutils.stats_uu.power import compute_power_posthoc_t_test
    # power2: float = compute_power_posthoc_t_test(cohen_d, N, alpha)
    from uutils.stats_uu.power import _compute_power_ttest
    power2: float = _compute_power_ttest(cohen_d, N, alpha)
    from uutils.stats_uu.power import _compute_power_ttest2
    # power3: float = _compute_power_ttest2(cohen_d, N, alpha)
    print(f'{power=}')
    print(f'{power2=}')
    # print(f'{power3=}')

    from uutils.stats_uu.power import print_interpretation_of_power
    print_interpretation_of_power(power)

    # - guess number of samples
    # print()
    # print('---- estimated number samples needed to get power ----')
    # print(f'true N used {N=}')
    # from uutils.stats_uu.power import get_estimated_number_of_samples_needed_to_reach_certain_power
    # N_estimated: float = get_estimated_number_of_samples_needed_to_reach_certain_power(cohen_d, alpha, power)
    # print(f'N_estimated {N_estimated=}')
    # print('(if gives numerical error ignore)')


# - run it

if __name__ == '__main__':
    import time

    start = time.time()
    # - run it
    my_test_using_stds_from_real_expts_()
    # - Done
    from uutils import report_times

    print(f"\nSuccess Done!: {report_times(start)}\a")
