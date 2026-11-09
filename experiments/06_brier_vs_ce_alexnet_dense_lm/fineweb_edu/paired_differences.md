# FineWeb-Edu: paired differences from the finished runs

**Doc link:** <https://github.com/brando90/ultimate-utils/blob/cursor/brier-ce-pilot-076e/experiments/06_brier_vs_ce_alexnet_dense_lm/fineweb_edu/paired_differences.md>

**TLDR:** At the matched training budget, cross-entropy training had higher held-out accuracy and lower cross-entropy, perplexity, and full Brier score in all three seed pairs; Brier training had lower top-label expected calibration error. Every exact two-sided sign-flip p-value is 0.25, the minimum attainable with three nonzero pairs, so these results do not establish a general superiority claim.

Analysis date: 11-08-2026. This is post-processing of saved metrics only; no training or evaluation was rerun. The original `results.md`, `state.json`, `protocol.json`, and `tokens_meta.json` are unchanged.

## Scope and method

- Experiment: `experiments/06_brier_vs_ce_alexnet_dense_lm/fineweb_edu`. The 123,551,232-parameter Transformer trained for 57,859 steps and 118,032,360 tokens per final run. Both objectives selected learning rate 0.0003. There are three paired training seeds, 0–2, and 1,992,315 scored targets per run on each split. The target count is not the number of independent replications.
- Difference is cross-entropy-trained minus Brier-trained, paired by seed. Positive accuracy favors cross-entropy training; negative cross-entropy, perplexity, or full Brier score favors cross-entropy training; positive expected calibration error favors Brier training. Accuracy and expected calibration error are fractions, not percentage points. Cross-entropy uses natural logarithms; perplexity is dimensionless. Full Brier score is the unnormalized sum of squared probability errors across classes, with range 0–2.
- The 95% interval is the percentile bootstrap of the paired mean difference, resampling the three training seeds with replacement. This report enumerates all 3³ = 27 equally likely ordered samples and uses inverse empirical cumulative-distribution-function quantiles at 0.025 and 0.975. Each extreme has probability 1/27 > 0.025, so these endpoints equal the minimum and maximum observed paired difference. This computes the exact bootstrap distribution; the older tables in `../metrics.py` use 10,000 pseudorandom draws with PyTorch seed 0. The target resampling distribution is the same, but this report does not claim to reuse that random-number stream.
- The p-value enumerates all 2³ = 8 sign assignments and counts absolute mean differences at least as large as observed (with the existing code's 1e-15 comparison tolerance). The null is that the seed-paired difference distribution is symmetric about zero, so its signs are exchangeable. Every reported test has count 2/8 = 0.25. P-values are unadjusted across five correlated metrics; validation is secondary and was used for selection. No new acceptance threshold is introduced.
- These intervals describe empirical training-seed variability conditional on this architecture, corpus shard, splits, tuning grid, and token budget. With only three seeds the bootstrap is extremely discrete and cannot support reliable population-coverage or broad generalization claims; it does not include corpus, document, architecture, or tuning-procedure uncertainty. An interval excluding zero and a coarse p-value of 0.25 are not contradictory: the small-sample bootstrap interval is not the inversion of the sign-flip test. Failure to reject does not establish equality.
- Top-label expected calibration error uses 15 equal-width bins of maximum softmax probability; it is neither classwise calibration nor full-distribution probability quality. Perplexity means below average each seed's perplexity; they are not the exponential of mean cross-entropy.

## Held-out test

All rows: 3 paired seeds; 1,992,315 scored targets per run.

| Metric | Cross-entropy-trained mean | Brier-trained mean | Paired mean difference [95% seed-bootstrap interval] | Two-sided p-value |
| --- | ---: | ---: | --- | ---: |
| Accuracy | 0.30837744 | 0.238854298 | +0.0695231427 [+0.069161754, +0.0699583148] | 0.25 |
| Cross-entropy | 4.57507279 | 6.07063118 | -1.49555838 [-1.50309969, -1.48877285] | 0.25 |
| Perplexity | 97.0356272 | 432.971989 | -335.936362 [-340.699992, -331.843623] | 0.25 |
| Full Brier score | 0.821049132 | 0.87228104 | -0.0512319079 [-0.0514568729, -0.0508599502] | 0.25 |
| Top-label expected calibration error | 0.00962699134 | 0.00318539177 | +0.00644159957 [+0.00519390715, +0.00745789132] | 0.25 |

| Metric | Seed 0 difference | Seed 1 difference | Seed 2 difference |
| --- | ---: | ---: | ---: |
| Accuracy | +0.069449359162582242 | +0.069958314824713957 | +0.069161754039898327 |
| Cross-entropy | -1.4887728452535907 | -1.4948026125347589 | -1.5030996918158017 |
| Perplexity | -331.84362334918671 | -335.26547161538673 | -340.69999174042539 |
| Full Brier score | -0.050859950171522339 | -0.051456872936602727 | -0.05137890059840089 |
| Top-label expected calibration error | +0.0074578913199512594 | +0.0066730002263069603 | +0.0051939071512485807 |

Sample standard deviations across the same three seeds (denominator n−1):

| Metric | Cross-entropy-trained | Brier-trained | Paired difference |
| --- | ---: | ---: | ---: |
| Accuracy | 0.000427571635 | 0.000425855337 | 0.000403373618 |
| Cross-entropy | 0.00402657885 | 0.0112013847 | 0.00719326248 |
| Perplexity | 0.390989076 | 4.85594228 | 4.4661377 |
| Full Brier score | 0.000130830165 | 0.000451571257 | 0.000324475479 |
| Top-label expected calibration error | 0.000802166033 | 0.000371384572 | 0.00114959374 |

## Validation (secondary; used for selection)

All rows: 3 paired seeds; 1,992,315 scored targets per run.

| Metric | Cross-entropy-trained mean | Brier-trained mean | Paired mean difference [95% seed-bootstrap interval] | Two-sided p-value |
| --- | ---: | ---: | --- | ---: |
| Accuracy | 0.308330259 | 0.238954014 | +0.0693762449 [+0.0693504792, +0.0694202473] | 0.25 |
| Cross-entropy | 4.60430802 | 6.10703181 | -1.50272379 [-1.51307892, -1.49632684] | 0.25 |
| Perplexity | 99.9144023 | 449.029161 | -349.114759 [-355.198531, -344.741314] | 0.25 |
| Full Brier score | 0.821106541 | 0.871946568 | -0.050840027 [-0.0510029679, -0.0505150347] | 0.25 |
| Top-label expected calibration error | 0.0100224439 | 0.00300519999 | +0.00701724395 [+0.00670300198, +0.00747485651] | 0.25 |

| Metric | Seed 0 difference | Seed 1 difference | Seed 2 difference |
| --- | ---: | ---: | ---: |
| Accuracy | +0.069420247300251231 | +0.069358008146302164 | +0.069350479216388983 |
| Cross-entropy | -1.4963268409645307 | -1.4987656005701915 | -1.5130789176061361 |
| Perplexity | -344.7413138549316 | -347.40443163833447 | -355.19853148058519 |
| Full Brier score | -0.050515034715474361 | -0.051002967948531652 | -0.051002078391492733 |
| Top-label expected calibration error | +0.007474856510996726 | +0.0068738733691068327 | +0.0067030019828334259 |

Sample standard deviations across the same three seeds (denominator n−1):

| Metric | Cross-entropy-trained | Brier-trained | Paired difference |
| --- | ---: | ---: | ---: |
| Accuracy | 0.000519315244 | 0.000481306492 | 3.82926942e-05 |
| Cross-entropy | 0.00417971208 | 0.012953651 | 0.0090503284 |
| Perplexity | 0.41753348 | 5.83126531 | 5.43435965 |
| Full Brier score | 0.00022259958 | 0.000497124441 | 0.000281451942 |
| Top-label expected calibration error | 0.000518088542 | 0.000191790065 | 0.000405408679 |

## Provenance and checks

Source revision: `a1fd6973bff8d9da770284050970714d94bb266c`. Protocol and token metadata identify `HuggingFaceFW/fineweb-edu sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The stored per-record `dataset: wikitext103` is a hardcoded label from the reused trainer; selection here uses the FineWeb file and its matching protocol, not that label.

Checks: 20 records = 8 tuning records + validation/test records for 6 final runs; all final seed/loss/split combinations are unique and complete. There are 0 invalid, 0 incomplete, and 0 out-of-memory records. All 12 final rows match the untouched `results.md` at its printed precision; `state.protocol` equals `protocol.json`, whose revision and document-ID hash match `tokens_meta.json`.

SHA-256 file digests:

- `state.json`: `eec2e1453221c15e9d3442753b9e7328a8b110bbd4e9e24b0ad759a8c133891a`
- `protocol.json`: `238a09f945b5b74888472e40371151cfdbd0674799054d3841c55d6887091a74`
- `tokens_meta.json`: `dfd79782478933bf8b5d0bc837d01cacb549cac1a0e6856591263882e6537199`
- `results.md`: `759c50f83b91eb9f9b5df1410f45280427832a50e78e2cbf994f902c9765fcfb`

Reproduce from the designated worktree:

```bash
python3 experiments/06_brier_vs_ce_alexnet_dense_lm/fineweb_edu/compute_paired_differences.py --check
# Omit --check to print full-precision source values and derived statistics as JSON.
# Use --write to regenerate this derived report only.
```

**TLDR-end:** [ultimate-utils: FineWeb-Edu paired differences] `experiments/06_brier_vs_ce_alexnet_dense_lm/fineweb_edu` favors cross-entropy training on held-out accuracy and the two proper scoring rules at this budget, while Brier training has lower top-label calibration error; three seeds limit statistical resolution and scope.
