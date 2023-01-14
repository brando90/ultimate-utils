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

