# Statistical Tests (Hypothesis Testing)

A statistical significance test, like a Chi square test, is attempting to answer a very specific problem: 
"how likely is it that a difference I observe in a random sample is just an artifact of sampling error 
(the error that arises when we try to make generalizations about a population using only a random sample 
of that population)?"

## Small Sample Size

### Summary

What is a small sample size? When power is not too high (e.g. 0.9999) and p-value/CIs are not to small, 
usually n<500 or 300.

Summary what to do for small sample size Hypothesis Testing:
    - p-value: stat test (e.g. t-test) with p-value & significance level
    - effect size: report the effect size (e.g. Cohen's d), see if it falls in the common ~0.2 (small), 0.5 (medium), 0.8 (large)  
    & compare it to `eps/pooled_std(group1, group2)`
    - CI: CI's, do they intersect given the epsilon that matters for your application?
    - Power/sample size: making an estimate of your std (or preliminary data), get Power of your test with a given sample size or compute 
    the sample size you need to achieve good power. 


ref:
    - also fantastic reference: https://stats.stackexchange.com/a/602978/28986 

## Large Sample Size
Same as previous comment but n>500 good rule of thumb.

Summary what to do for large sample size Hypothesis Testing:
    - CI: CI's and use the epsilon valid in your applicaiton
    - Effect size: report effect size, see if it falls in the common ~0.2 (small), 0.5 (medium), 0.8 (large) 
    & compare it to `eps/pooled_std(group1, group2)`
    - LRT: todo
    - eps != 0 p-value: todo

Real conclusion:
    - will look at all numbers but will only use the effect size with common ~0.2 (small), 0.5 (medium), 0.8 (large) to
    make a decision. Looking at `eps/pooled_std(group1, group2)` is worth observing too but won't use for decisions.

## Hands on example

See `my_test_using_stds_from_real_expts_()` function in `effect_size.py`.

### Todo later (for large sample size)

- LRT (theory & python) mainly for large sample size
- hypothesis testing with non-zero epsilon (theory & python) mainly for large sample size

ref:
    - Fantastic reference: https://stats.stackexchange.com/questions/35470/significance-test-for-large-sample-sizes/602978#602978
    - my answer: https://stats.stackexchange.com/a/603027/28986

