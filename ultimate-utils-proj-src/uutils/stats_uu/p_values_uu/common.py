def print_statisticall_significant(p, alpha=0.05):
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