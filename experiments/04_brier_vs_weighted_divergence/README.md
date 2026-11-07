# Brier score vs the weighted divergence

Checked on 11-06-2026 against the handwritten notes and against MIT 6.437 / 6.8700 lecture 13 (Inference as Decision). The lecture PDFs stay in `~/Documents/Brandos Textbooks and Notes/6.437 6.8700 2024 notes`. This repository is public, so those course PDFs are not copied in. See `6.437.md` for the lecture list and the two definitions used below.

## What the notes equate, and what the Brier score is

Lecture 13 writes the quadratic cost as a sum over the whole alphabet:

```text
BS(q, x) = sum_a (q(a) - 1{a = x})^2
         = (1 - q(x))^2 + sum_{a != x} q(a)^2
```

The first page of the notes has this decomposition: term 1 is the observed symbol, term 2 is every other symbol. The third page starts from the weighted divergence

```text
WD(p, q) = sum_a p(a) (p(a) - q(a))^2
```

and substitutes the one-hot `p = 1{· = x}`. The weight `p(a)` deletes every summand except `a = x`, leaving `(1 - q(x))^2`. That is term 1 alone. The page then labels it `BS`. The label is the mistake. Over 200 random forecasts of alphabet size 2 through 7, the gap between the Brier score and `(1 - q(x))^2` had maximum 0.952. The two agree when the forecast puts no mass on the symbols that did not occur.

The missing weight is the other half of the comparison. The squared Euclidean distance between distributions is `sum_a (p(a) - q(a))^2`, with no extra factor `p(a)`. Substituting the one-hot into that unweighted distance produces the Brier score.

## Do the minimizers agree?

`WD(p, q)`, expected Brier score, and expected log loss are all uniquely minimized at `q = p` on the probability simplex. SLSQP, from 12 starts, matched that claim to within `1e-6` in Euclidean distance on five distributions (binary uniform, binary skewed, ternary uniform, ternary skewed, five-class skewed).

The local score `(1 - q(x))^2` is a different question, and the answer depends on the alphabet size.

| Alphabet | True `p` | Minimizer of `E[(1 - q(X))^2]` | Distance to `p` |
| --- | --- | --- | --- |
| 2 | (0.80, 0.20) | (0.80, 0.20) | 0 |
| 2 | (0.50, 0.50) | (0.50, 0.50) | 0 |
| 3, uniform | (1/3, 1/3, 1/3) | `p` | 0 |
| 3, skewed | (0.70, 0.20, 0.10) | (7/9, 2/9, 0) ≈ (0.778, 0.222, 0) | 0.1286 |
| 5, skewed | (0.50, 0.25, 0.15, 0.07, 0.03) | (2/3, 1/3, 0, 0, 0) | 0.2510 |

For the skewed ternary distribution the closed form was also the minimum on an 801-by-801 grid of the simplex. The local score puts zero mass on the rare symbol and moves the other two. Expected local loss is lower there than at `p`, so this is the minimum, and the flow is not a slow approach to `p`.

Lecture 13, Theorem 13.1, is the reason the split is at three symbols. For an alphabet of size at least 3, log loss is the only smooth cost that is both local and proper. A cost is local when it depends on the forecast only through the probability of the symbol that occurred. Log loss `-log q(x)` is local. So is `(1 - q(x))^2`. The Brier score is not local, because of the sum over `a ≠ x`. On a binary alphabet the theorem's hypothesis fails, and `(1 - q(x))^2` is proper: its minimizer is `p`. On three or more symbols it cannot be proper, and the table is the counterexample.

## Same target, different speed

Yes, for `WD(p, q)` against the Brier score. Both are minimized at `p`. Their projected Hessians on the simplex differ, so gradient flow does not move at the same rate.

Rates below are the eigenvalues of the projected Hessian. The time column is how long the slowest direction takes to shrink by a factor `e`. The matrix exponential reproduces that factor to within `1e-8`.

| Distribution | Brier rates | Weighted-divergence rates | Slow-time ratio, WD / Brier |
| --- | --- | --- | --- |
| any binary `p` | 2 | 1 | 2 exactly |
| uniform, 3 symbols | 2, 2 | 2/3, 2/3 | 3 exactly |
| (0.70, 0.20, 0.10) | 2, 2 | 0.295, 1.038 | 6.77 |

On a uniform distribution, `WD(p, q) = (1/K) ||q - p||^2`, so the two losses are proportional. The trajectory is the same curve, run 3 times slower for `K = 3`. The error directions stay aligned (minimum cosine 1.000). On `(0.70, 0.20, 0.10)` the paths separate (minimum cosine 0.663). The rare symbol is the slow direction, because the `a`-th derivative of `WD` carries a factor `p(a)`.

From `q(0) = (0.05, 0.90, 0.05)` with `p = (0.70, 0.20, 0.10)`, one projected-gradient trajectory, step checked by halving:

| Loss | Time to distance 0.1 | Time to 0.01 | Time to 0.001 | Distance at time 12 |
| --- | --- | --- | --- | --- |
| Log loss | 0.90 | 1.72 | 2.49 | ~0 |
| Brier | 1.13 | 2.28 | 3.43 | ~0 |
| Weighted divergence | 4.24 | still above 0.01 | still above 0.01 | 0.010 |
| Local `(1 - q(x))^2` | stays above 0.1 | stays above 0.1 | stays above 0.1 | 0.129 |

Log loss is faster than the Brier score near this `p`. Its linearized slow rate is 2.99 against the Brier score's 2. That comparison is local to the optimum. The plotted curves agree from this far start as well.

There is a second, statistical, reading. If `q` is a free probability vector, the empirical minimizer of the Brier score, of log loss, and of `WD(p̂, q)` is exactly the histogram `p̂`. Counting finishes all three. The speed gap shows up when the objective is descended, or when `q` is a restricted parametric model. It does not show up as two different histograms.

## Finite samples of the local score

`p = (0.70, 0.20, 0.10)`, 400 independent multinomial samples per row. The interval is the standard error, sample standard deviation divided by `sqrt(400)`.

The histogram row is also the empirical Brier, log-loss, and weighted-divergence minimizer. The local row is the closed-form minimizer of the local score on that histogram.

| n | Histogram distance to `p` | Local minimizer distance to `p` | Local minimizer distance to (7/9, 2/9, 0) |
| --- | --- | --- | --- |
| 20 | 0.1307 ± 0.0038 | 0.1814 ± 0.0031 | 0.1407 ± 0.0053 |
| 200 | 0.0406 ± 0.0011 | 0.1340 ± 0.0007 | 0.0348 ± 0.0014 |
| 2,000 | 0.0130 ± 0.0003 | 0.1293 ± 0.0002 | 0.0106 ± 0.0004 |
| 20,000 | 0.0042 ± 0.0001 | 0.1286 ± 0.0001 | 0.0034 ± 0.0001 |

The population gap is 0.1286. By `n = 20,000` the local estimator has arrived there, while the histogram is at 0.0042.

## Reproduce

```bash
python experiments/04_brier_vs_weighted_divergence/run_comparison.py
```

Outputs: `results/summary.json`, `results/distance_to_truth.png`.
