# Weighted squared divergence and KL divergence

**Doc link:** <https://github.com/brando90/ultimate-utils/blob/main/experiments/04_brier_vs_weighted_divergence/NOTES.md>

**TLDR:** The weighted squared divergence and KL divergence are the same kind of expectation, with squared difference and log-likelihood ratio as the discrepancies. On \(p=(0.70,0.20,0.10)\) and \(q=(0.05,0.90,0.05)\) those expectations equal \(0.394\) and \(1.615839\). This file is in the working tree; the address above resolves after a push to `main`.

Checked on 11-06-2026. The TMLR-style report is `tmlr/main.pdf`. The check is `run_expectation_identity.py`.

## The identity

Let \(X\) be drawn from a distribution \(p\) on a finite alphabet, and let \(q\) be a forecast with \(q(a)>0\) whenever \(p(a)>0\). The weighted squared divergence and KL divergence are

\[
\mathrm{WD}(p,q)=\mathbb{E}_{X\sim p}\bigl[(p(X)-q(X))^2\bigr],
\]
\[
D_{\mathrm{KL}}(p\|q)=\mathbb{E}_{X\sim p}\bigl[\log p(X)-\log q(X)\bigr].
\]

The chat notation \(\mathrm{WD}(p\|q)\) is this same function. Both are \(\mathbb{E}_{X\sim p}[d(p(X),q(X))]\). The discrepancy for the weighted squared divergence is \(d(u,v)=(u-v)^2\). The discrepancy for KL divergence is the log-likelihood ratio \(d(u,v)=\log u-\log v\).

The summand of KL divergence still contains \(\log p(X)\). With \(p\) held fixed, \(\mathbb{E}[\log p(X)]=-H(p)\) does not depend on \(q\). Removing it leaves expected log loss, \(-\mathbb{E}[\log q(X)]\). On the pair below, \(H(p)=0.801819\) and expected log loss is \(2.417658\), which equals KL divergence plus that entropy.

## One-hot substitution

If \(p=e_x\), the expectation hits only the observed symbol. The weighted squared divergence collapses to the local square score \((1-q(x))^2\). KL divergence collapses to log loss \(-\log q(x)\). For \(q=(0.05,0.90,0.05)\):

| Observed symbol | Local square score | Log loss |
| --- | --- | --- |
| 0 | 0.9025 | 2.995732 |
| 1 | 0.0100 | 0.105361 |
| 2 | 0.9025 | 2.995732 |

The local square score is proper on a binary alphabet and, for three or more symbols, its population minimizer is not \(p\). Log loss is proper on every alphabet size. The shared expectation form produces those two one-sample scores, and only the log-likelihood ratio stays proper after the substitution.

## Rates

The outer factor \(p(a)\) is the same in both sums. The second derivative of \((p(a)-q(a))^2\) in \(q(a)\) is \(2\), so the Hessian of the weighted squared divergence at \(q=p\) is \(2\,\mathrm{diag}(p)\). The second derivative of \(\log p(a)-\log q(a)\) is \(1/q(a)^2\), so the Hessian of KL divergence at \(q=p\) is \(\mathrm{diag}(1/p)\).

On \(p=(0.70,0.20,0.10)\) the slow eigenvalues on the simplex are \(0.295\) for the weighted squared divergence and \(2.99\) for KL divergence. The slow mode shrinks by a factor \(e\) in time \(3.38\) and in time \(0.334\). Rare symbols are the slow directions of the square and the fast directions of the log-likelihood ratio.

At the interior pair, the Euclidean gradients are

\[
\nabla_q \mathrm{WD}=(-0.91,0.28,-0.01), \qquad \nabla_q D_{\mathrm{KL}}=(-14,-2/9,-2).
\]

The first coordinate of the KL gradient is \(p(0)\) times \(-1/q(0)=-20\). One sample of log loss on that symbol has partial derivative \((-20,0,0)\). The population gradient spreads that one-sample derivative across symbols by the masses \(p(a)\). The weighted squared divergence does the same with the derivative \(2(q(a)-p(a))\), which stays bounded as \(q(a)\to 0\).

## Symbols that did not occur

Log loss of an outcome \(x\) depends on \(q\) only through \(q(x)\). For \(q_{\mathrm{conc}}=(0.40,0.60,0,0)\) and \(q_{\mathrm{spread}}=(0.40,0.20,0.20,0.20)\), log loss is \(0.916291\) on both forecasts and the local square score is \(0.36\) on both. The Brier scores are \(0.72\) and \(0.48\).

Under the uniform distribution on four symbols, the population divergences do see the split. The weighted squared divergence is \(0.0675\) and \(0.0075\). KL divergence of the spread forecast is \(0.049857\). The concentrated forecast puts probability zero on two symbols that the uniform distribution charges, so KL divergence is infinite there, and the weighted squared divergence stays at \(0.0675\).

## What the check ran

`run_expectation_identity.py`, seed \(0\), \(100{,}000\) Monte Carlo draws. The enumeration matched the vectorized sums to \(10^{-12}\). The predicted weighted squared divergence on the interior pair was \(0.394\), and the run returned that value up to floating-point error of size about \(10^{-16}\).

| Discrepancy | Exact | Monte Carlo | Absolute error | Allowance, 5 standard errors |
| --- | --- | --- | --- | --- |
| Squared difference | 0.394 | 0.394408 | 0.000408 | 0.002106 |
| Log-likelihood ratio | 1.615839 | 1.615987 | 0.000147 | 0.026291 |

The same equalities hold on the uniform ternary pair \(p=(1/3,1/3,1/3)\), \(q=(0.50,0.30,0.20)\) and on the binary pair \(p=(0.80,0.20)\), \(q=(0.30,0.70)\). Finite differences matched \(2p(q-p)\) and \(-p/q\) to \(10^{-6}\). Driving the third coordinate of \(q\) to zero leaves the weighted squared divergence at \(0.047\) for the skewed \(p\).

```bash
python experiments/04_brier_vs_weighted_divergence/run_expectation_identity.py
```

**TLDR-end:** [ultimate-utils: WD and KL as expectations] Both divergences equal an expectation under \(p\) of a discrepancy between \(p(X)\) and \(q(X)\), and the square versus the log-likelihood ratio is what reverses the rare-symbol rate.
