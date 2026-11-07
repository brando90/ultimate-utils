# Expectation identity for the weighted squared divergence and KL divergence

**Status:** DONE (11-06-2026)

**Last updated:** 11-06-2026

**Doc link:** <https://github.com/brando90/ultimate-utils/blob/main/experiments/04_brier_vs_weighted_divergence/results.md>

The address resolves after a push to `main`. The notes are `NOTES.md`. The script is `run_expectation_identity.py`, seed 0.

| Phase | State |
| --- | --- |
| Enumeration of both expectations | DONE |
| One-hot collapse | DONE |
| Monte Carlo, 100,000 draws | DONE |
| Finite-difference gradients | DONE |
| Residual split | DONE |

| Quantity | Value |
| --- | --- |
| \(p\), \(q\) | (0.70, 0.20, 0.10), (0.05, 0.90, 0.05) |
| Weighted squared divergence | 0.394 |
| KL divergence | 1.615839 |
| Expected log loss | 2.417658 |
| Monte Carlo absolute error, square | 0.000408, allowance 0.002106 |
| Monte Carlo absolute error, log-likelihood ratio | 0.000147, allowance 0.026291 |
| KL gradient | (-14, -2/9, -2) |
| Weighted-squared gradient | (-0.91, 0.28, -0.01) |
| One-sample log-loss partial on symbol 0 | (-20, 0, 0) |

**TLDR-end:** [ultimate-utils: expectation identity] The weighted squared divergence and KL divergence match their expectation formulas, and a 100,000-draw average of each discrepancy stayed inside 5 standard errors.
