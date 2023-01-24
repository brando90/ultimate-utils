"""
Power = the ability of a statistical test (decision procedure) to detect a relationship or difference (reject the null hypothesis)
 if the effect is present. i.e. Pr[H(y)=H1 | H=H1] = 1 - beta, beta = Pr[H(y)=H0 | H=H1] = Pr[Type II error] = Pr[false negative]
 related, alpha = size = Pr[H(y)=H1 | H=H0] = Pr[Type I error] = Pr[false positive].

What is acceptable power
According to Courney Donovan (uc denver):
    - acceptable power is 0.8 or higher
    - some say 0.7 is adequate
    - 0.9 is considered excellent
Recommendation: do power analysis before study (can be done after, but contraversial...why? I see no issue e.g. sometimes
you can't know the effect size so you try to do the study anyway with the best sampel you can and then check if the
power was sufficient.

What does power depend on?
- alpha level
- sample size (n)
- effect size (d)
- the type of statistical test (e.g. t-test, ANOVA, etc.)
- the type of design (e.g. one-tailed vs two-tailed, etc.)

# Computing Power & sample size before/apriori a study:

# Computing Power & sample size after/post-hoc a study:
Post-hoc power analysis is a method of determining the power of a study after the data have been collected, given:
    - alpha (todo, I don't think this is false epositive acutally e.g. see p-values, the p-value is the false positive but alpha just determines this threshold for p-values)
    - sample size
    - effect size

Remark: what is amazing for this method is that it's essential exact/correct. This is because the hypothesis we test
wrt to the R.V. y is when y is the sample mean -- which **is** distributed as a Normal (as N->inf).

todo: check CTL convergence conditions.

ref:
    - https://www.youtube.com/watch?v=HPKcmEhP-4s&list=PLljPCllSdtzWop4iDNyyuZ2NZokOHeQqm&index=7
    - https://education.ucdenver.edu/about-us/faculty-directory/Donovan-Courtney-UCD6000147384
"""
import numpy as np


# def _compute_power_posthoc_ttest(effect_size: float, n: int, alpha: float, alternative: str = 'two-sided') -> float:
#     """
#     Compute power after a study has been done.
#
#     - Given N, std -> Power
#         - tails
#         - alpha (significance level)
#         - effect size d (difference between means, e.g. Cohen's d)
#         - Power (probability of detecting a difference, 1 - beta)
#     """
#     # power = norm.sf(effect_size * np.sqrt(sample_size) - norm.ppf(1 - alpha))
#     # compute power with scipy
#     from scipy.stats import norm
#     power = norm.ttest_power(effect_size, nobs=n, alpha=alpha, alternative=alternative)
#     # power = norm.ttest_power(effect_size * np.sqrt(sample_size) - norm.ppf(1 - alpha))
#     return power


def _compute_power_ttest(effect_size: float, n: int, alpha: float, alternative: str = 'two-sided') -> float:
    """
    Compute power for a t-test from effect size, sample size, alpha, and type of alternative hypothesis.

    Note:
        alternative = 'two-sided' | 'smaller' | 'larger'.

    ref:
        - https://stackoverflow.com/questions/54067722/calculate-power-for-t-test-in-python

    Note: uses
        analysis = TTestIndPower()
        result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
    form https://machinelearningmastery.com/statistical-power-and-power-analysis-in-python/ under the hood.
    """
    from statsmodels.stats.power import TTestIndPower
    analysis = TTestIndPower()
    power: float = analysis.power(effect_size, nobs=n, alpha=alpha, alternative=alternative)
    return power


def _compute_power_ttest(effect_size: float, n: int, alpha: float, alternative: str = 'two-sided') -> float:
    """
    Compute power for a t-test from effect size, sample size, alpha, and type of alternative hypothesis.

    Note:
        alternative = 'two-sided' | 'smaller' | 'larger'.

    ref:
        - https://machinelearningmastery.com/statistical-power-and-power-analysis-in-python/
    """
    import statsmodels.stats.power as smp
    power: float = smp.tt_ind_solve_power(effect_size=effect_size, nobs1=n, alpha=alpha, alternative=alternative)
    return power


def _compute_power_ttest2(effect_size: float, n: int, alpha: float, alternative: str = 'two-sided') -> float:
    """
    Compute power for a t-test from effect size, sample size, alpha, and type of alternative hypothesis.

    Note:
        alternative = 'two-sided' | 'smaller' | 'larger'.

    ref:
        - https://stackoverflow.com/questions/54067722/calculate-power-for-t-test-in-python/75215952#75215952
    """
    raise ValueError(
        'Has an error for now see: https://stackoverflow.com/questions/54067722/calculate-power-for-t-test-in-python')
    import statsmodels.stats.power as smp
    power: float = smp.ttest_power(effect_size=effect_size, nobs=n, alpha=alpha, alternative=alternative)
    return power


def get_estimated_number_of_samples_needed_to_reach_certain_power(effect_size: float,
                                                                  power: float,
                                                                  alpha: float,
                                                                  alternative: str = 'two-sided',
                                                                  ) -> float:
    """
    Compute sample size needed to reach a certain power for a t-test. Note the result will be a float.

    Note:
        - Guess: number of observations is the number of observations for RV in question. So if your doing dif = mu1 - mu2
        N estimated or given to power will be the number of difs, not the total from the two groups.

    ref:
        - https://machinelearningmastery.com/statistical-power-and-power-analysis-in-python/
    """
    # perform power analysis
    from statsmodels.stats.power import TTestIndPower
    analysis = TTestIndPower()
    result = analysis.solve_power(effect_size, power=power, nobs1=None, ratio=1.0, alpha=alpha, alternative=alternative)
    return result


def print_interpretation_of_power(power: float):
    """
    Print the interpretation of a power value.

    ref:
        - Courtney Donovan: https://www.youtube.com/watch?v=HPKcmEhP-4s&list=PLljPCllSdtzWop4iDNyyuZ2NZokOHeQqm&index=7
    """
    if power < 0.2:
        print(f'Power is very low (power < 0.2) {power}')
    elif power < 0.5:
        print(f'Power is low (power < 0.5) {power=}')
    elif 0.8 > power >= 0.7:
        print(f'Power is acceptable (>=0.7 is adequate) {power=}')
    elif 0.9 > power >= 0.8:
        print(f'Power is good (>=0.8 is acceptable) {power=}')
    elif 1.0 >= power >= 0.9:
        print(f'Power is excellent (>=0.9 is excellent) {power=}')
    else:
        raise ValueError(f'Power is not in range [0, 1] {power=}')


# -

def compute_power_posthoc_test():
    """
    Compute power after a study has been done.
    """
    pass


if __name__ == '__main__':
    compute_power_posthoc_test()
