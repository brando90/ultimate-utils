"""
Anova
Use case: it's a statistical test to compare the means of two or more groups to determine whether any of their means differ from each other.


refs:
    - anova machine learning mastery:
        - 1 How to Develop a Feature Selection Subspace Ensemble in Python: https://machinelearningmastery.com/feature-selection-with-numerical-input-data/
        - 2 How to Develop a Feature Selection Subspace Ensemble in Python: https://machinelearningmastery.com/feature-selection-subspace-ensemble-in-python/
        - 3 Feature Selection For Machine Learning in Python: https://machinelearningmastery.com/feature-selection-machine-learning-python/
        - 4 SK's link, tests as linear: https://lindeloev.github.io/tests-as-linear/#6_three_or_more_means
    - other nice links:
        - wikipedia: https://en.wikipedia.org/wiki/Analysis_of_variance
        - Contrast Anova with Linear Regression, so read this eventually: https://machinelearningmastery.com/linear-regression-in-r/
        - 17 Statistical Hypothesis Tests in Python (Cheat Sheet): https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
        - Statistics books for machine learning: https://machinelearningmastery.com/statistics-books-for-machine-learning/
        - https://lindeloev.github.io/tests-as-linear/
        - anova r2 & goodness of fit, but this is only goodness of fit: https://lindeloev.github.io/tests-as-linear/#71_goodness_of_fit

Notes & Questions:
1 says:
"
How to Perform Feature Selection With Numerical Input Data
Feature selection is the process of identifying and selecting a subset of input features that are most relevant to the target variable.
Feature selection is often straightforward when working with real-valued input and output data, such as using the Pearson’s correlation coefficient, but can be challenging when working with numerical input data and a categorical target variable.
The two most commonly used feature selection methods for numerical input data when the target variable is categorical (e.g. classification predictive modeling) are the ANOVA f-test statistic and the mutual information statistic.
"
Q1: therefore, is studying emergence (as defined with delta = fs - zs the right technique to use for feature selection?
i.e. is it a numerical input data and a categorical target variable/decide which independent variable matters most?
A1: todo, since anova uses y = categorical, but delta is numerical/real valued, do we really need anova?

Q2: does Anova have p-value -> 0 as n->inf issues like the t-test?
A2: todo, but I think yes. Solution is likely to use eps & effect size based on anova.

Q3: small anova tutorial, idk if it's the right tool if our delta is numeric and anova usually uses y is categorical. Id also want to detect if one specific variables is significant wrt others. Or how significant xi is in predicting y.
A3: todo

---
Hypothesis of how it my be of use for my case (dvs numeric, xs have groups/"categories"):
- Y = f(x1, ..., xn)
- mu_i = E[yi = f(x1, ..., xi = xi, ..., xn)] i.e. we use treatment xi and do multiple measurments of Y
- then we do anova test
- Q: what do we do the other groups? How do we keep them constant or make them zero? what if they aren't constant or zero? Can we still do the anova test?
"""


# -

def two_level_anova_example():
    pass


def anova_example_one_way_from_chat_gpt():
    """
    This example simple shows if we can detect if the means are all the same or not in some data that was sampled.
    H0: mu1 = mu2 = mu3 = mu ?
    H1: mu1 != mu2 != mu3 != mu (all possible alternatives)?

    One way Anovs:
        - One factor = e.g. the outcome of one treatment disease (outcome)
        - @ b (=3) levels = treatment drugs (groups)
    Q: todo there seems to be no dv i.e. no y = f(x1, x2, x3), so does anova work in this case? How does the dv y
        affect if it's one way or two way anova? for y1, y2 does it mean that we have two way anova?
    A:

    The sample data represents the scores of three groups on a test, and the one-way ANOVA test is used to determine
    if there is a significant difference in the mean scores between the groups.
    The F-value and P-value obtained from the stats.f_oneway function are used to make the statistical inference.
    If the P-value is less than 0.05, it means that there is a significant difference in the mean scores between the groups.
    """
    import pandas as pd
    from scipy import stats
    import numpy as np

    # Generate sample data from Gaussians with the same mean
    np.random.seed(0)
    mean = 0
    std = 1
    # - should accept H0 i.e. not statistically significant difference between means of the groups)
    mean1 = mean
    mean2 = mean
    mean3 = mean
    # - should reject H0 (accept H1)  i.e. statistically significant difference between any of the means of the groups)
    mean1 = 0
    mean2 = 5
    mean3 = 10
    # - stds
    std1 = std
    std2 = std
    std3 = std
    N = 150  # number of samples, want it somewhat big but not to big, idk if anova has issues like p-tests with big sample sizes. I suppose yes.
    # N = 300
    group1 = np.random.normal(mean1, std1, N)
    group2 = np.random.normal(mean2, std2, N)
    group3 = np.random.normal(mean3, std3, N)

    # Create dataframe with the sample data
    data = {"Group 1": group1,
            "Group 2": group2,
            "Group 3": group3}
    df = pd.DataFrame(data)

    # Conduct the ANOVA test
    alpha = 0.05
    f_val, p_val = stats.f_oneway(df["Group 1"], df["Group 2"], df["Group 3"])

    # Print the results
    print("F-value:", f_val)
    print("P-value:", p_val)

    # Interpret the results
    if p_val < alpha:
        print("Reject the null hypothesis - significant differences exist between the means.")
    else:
        print("Fail to reject the null hypothesis - no significant difference exists between the means.")
    from uutils.stats_uu.p_values_uu.t_test_uu import decision_procedure_based_on_statistically_significance
    decision_procedure_based_on_statistically_significance(p_val, alpha)


def show_n_does_or_doesnt_affect_p_value_for_anova_one_way_test():
    # """
    # plot p-value vs n. Does n affect p-value?
    # """
    # print('P-values is the probability that you observe the current your sample mean or more extreme under H0.')
    # mu_x, mu_y = 0, 10  # should reject null hypothesis i.e. statistically significant
    # std_x, std_y = 10, 12
    # # ns: list[int] = [2, 10, 30, 100, 500, 1000, 5000, 10_000, 50_000, 100_000]
    # ns: list[int] = [2, 10, 15, 30, 50, 75, 100, 150, 300]
    # p_values: list[float] = []
    # for n in ns:
    #     x = np.random.normal(mu_x, std_x, n)
    #     y = np.random.normal(mu_y, std_y, n)
    #     t, p = stats.ttest_ind(x, y, equal_var=True)
    #     print(f'{p=} (Probability that this sample mean or more extreme is observerd, also type I error)')
    #     p_values.append(p)
    #     from uutils.stats_uu.p_values_uu.common import print_statisticall_significant
    #     print_statisticall_significant(p)
    # print(p_values)
    # # plot p-values vs n, y axis is p value, x axis is n (sample size), smooth curve
    # from uutils.plot import plot
    # plot(ns, p_values, title='p-values vs n', xlabel='n (sample size)', ylabel='p-values')
    # # plot horizonal line at alpha=0.05
    # plt.axhline(y=0.05, color='r', linestyle='-')
    # plt.show()
    pass


def effect_size_anova_chat_gpt():
    """
    todo: NOT BEEN VERIFIED YET.

    One commonly used measure of effect size for a one-way ANOVA is eta-squared (η²). It can be calculated as the ratio of the between-group variance to the total variance in the data. In Python, you can calculate it using the following code:

    In this example, the eta-squared value is calculated using the formula for the ratio of between-group variance to total variance in the data. The larger the eta-squared value, the stronger the effect size. A value of 0.01 is considered a small effect, 0.06 a medium effect, and 0.14 a large effect.
    """
    # import numpy as np
    # import pandas as pd
    # from scipy import stats
    #
    # # Generate sample data from Gaussians with the same mean
    # np.random.seed(0)
    # mean = 0
    # std = 1
    # group1 = np.random.normal(mean, std, 50)
    # group2 = np.random.normal(mean, std, 50)
    # group3 = np.random.normal(mean, std, 50)
    #
    # # Create dataframe with the sample data
    # data = {"Group 1": group1,
    #         "Group 2": group2,
    #         "Group 3": group3}
    # df = pd.DataFrame(data)
    #
    # # Conduct the ANOVA test
    # f_val, p_val = stats.f_oneway(df["Group 1"], df["Group 2"], df["Group 3"])
    #
    # # Calculate eta-squared
    # n = df.shape[0]
    # k = df.shape[1]
    # SS_total = ((df - df.mean().mean()) ** 2).sum().sum()
    # SS_between = ((df.mean() - df.mean().mean()) ** 2 * df.count()).sum()
    # eta_squared = SS_between / SS_total
    #
    # # Print the results
    # print("F-value:", f_val)
    # print("P-value:", p_val)
    # print("Eta-squared:", eta_squared)
    #
    # # Interpret the results
    # if p_val < 0.05:
    #     print("Reject the null hypothesis - significant differences exist between the means.")
    # else:
    #     print("Fail to reject the null hypothesis - no significant difference exists between the means.")
    pass


# - run main, examples, tests, etc.

if __name__ == '__main__':
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    anova_example_one_way_from_chat_gpt()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
