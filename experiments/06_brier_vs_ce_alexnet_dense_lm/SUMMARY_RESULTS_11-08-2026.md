# Brier versus cross-entropy: five completed study parts

**Doc link:** <https://github.com/brando90/ultimate-utils/blob/cursor/brier-ce-pilot-076e/experiments/06_brier_vs_ce_alexnet_dense_lm/SUMMARY_RESULTS_11-08-2026.md>

**TLDR:** Cross-entropy training had higher accuracy and better probability scores in both language-model comparisons at the tested budget. Brier training had higher accuracy in the longer CIFAR-10 follow-up; the collapsed-bias failure depended on the optimizer and initialization, so the study does not establish a universal ordering of the losses.

Reporting period: 11-07-2026–11-08-2026. Canonical study: `experiments/06_brier_vs_ce_alexnet_dense_lm`. Source results revision: `a1fd6973bff8d9da770284050970714d94bb266c`. All training was already finished. This summary adds saved-data analysis only and leaves the original results and records unchanged.

Google companion: [weekly update, Assistant Results tab](https://docs.google.com/document/d/1xkEn8vW800jBucuJANuXD8ph-sMQMofwNZw_nJzjJls/edit?tab=t.ty5se4uscfip).

## Results across the five parts

The difference is **cross-entropy-trained minus Brier-trained**, paired by training seed. Means are percentages; differences and intervals are **percentage points**. The collapsed-bias rows measure class-0 recall, not overall accuracy. Other rows measure held-out accuracy. Positive differences favor cross-entropy training for these metrics.

| Part and test measurement | Cross-entropy mean | Brier mean | Paired difference [95% seed-bootstrap interval] | Two-sided p-value | Paired seeds |
| --- | ---: | ---: | --- | ---: | ---: |
| 1. WikiText-103 accuracy | 38.400% | 28.553% | +9.847 [9.734, 9.960] | 0.25 | 3 |
| 2. FineWeb-Edu accuracy | 30.838% | 23.885% | +6.952 [6.916, 6.996] | 0.25 | 3 |
| 3. CIFAR-10 follow-up, AdamW accuracy | 79.594% | 81.966% | −2.372 [−2.792, −2.022] | 0.0625 | 5 |
| 3. CIFAR-10 follow-up, stochastic gradient descent accuracy | 82.844% | 84.362% | −1.518 [−2.658, −0.378] | 0.125 | 5 |
| 4. Collapsed bias, stochastic gradient descent class-0 recall | 98.189% | 0.000% | +98.189 [97.296, 99.082] | 0.125 | 4 |
| 4. Collapsed bias, Adam class-0 recall | 96.429% | 95.255% | +1.173 [−0.612, 3.061] | 0.375 | 4 |
| 5. Early AlexNet, MNIST accuracy (2 epochs) | 99.262% | 99.218% | +0.044 [−0.020, 0.108] | 0.3125 | 5 |
| 5. Early AlexNet, CIFAR-10 accuracy (6 epochs) | 81.190% | 79.750% | +1.440 [1.288, 1.588] | 0.0625 | 5 |

Sources by part: [1: WikiText-103](lm_fixed/results.md), [2: FineWeb-Edu raw table](fineweb_edu/results.md) and [new paired analysis](fineweb_edu/paired_differences.md), [3: CIFAR-10 follow-up](stability_cifar/results.md), [4: collapsed-bias counts](collapsed_replication/results.md) and [measurements](collapsed_replication/MEASURED.md), [5: early AlexNet](results.md). Existing-table means are calculated from printed rows and rounded accordingly. FineWeb-Edu uses full-precision state records; collapsed-bias recall uses integer hits divided by 980 test class-0 images. Image accuracy uses 10,000 official test images per run. Neither image nor token counts replace the seed count as the replication unit.

## Missing FineWeb-Edu paired differences

These are new post hoc calculations from `fineweb_edu/state.json`, not new model evaluations. Each of the three final seed pairs trained on 118,032,360 tokens; each held-out evaluation scored 1,992,315 targets. Both losses selected learning rate 0.0003. Accuracy and expected calibration error below are fractions; cross-entropy is in natural-log units per target. Perplexity is the exponential of cross-entropy within each seed, then averaged across seeds. Full Brier score is the sum over all classes of squared probability errors, without division by class count.

| Test metric | Cross-entropy-trained mean | Brier-trained mean | Paired difference [95% seed-bootstrap interval] | Two-sided p-value |
| --- | ---: | ---: | --- | ---: |
| Accuracy | 0.308377440 | 0.238854298 | +0.069523143 [0.069161754, 0.069958315] | 0.25 |
| Cross-entropy | 4.575072795 | 6.070631178 | −1.495558383 [−1.503099692, −1.488772845] | 0.25 |
| Perplexity | 97.035627174 | 432.971989409 | −335.936362235 [−340.699991740, −331.843623349] | 0.25 |
| Full Brier score | 0.821049132 | 0.872281040 | −0.051231908 [−0.051456873, −0.050859950] | 0.25 |
| Top-label expected calibration error | 0.009626991 | 0.003185392 | +0.006441600 [0.005193907, 0.007457891] | 0.25 |

The seed-wise accuracy differences are +0.069449359, +0.069958315, and +0.069161754. Cross-entropy training has the favorable direction on accuracy, cross-entropy, perplexity, and full Brier score in every seed. Brier training has lower top-label expected calibration error in every seed. The [paired analysis](fineweb_edu/paired_differences.md) includes all per-seed differences, sample standard deviations, secondary validation results, and source hashes.

## What the evidence supports

1. **Cross-entropy training is the stronger observed choice for these language-model settings.** Both datasets use the same 123,551,232-parameter Transformer, three paired seeds, 118,032,360 final training tokens, and AdamW with weight decay 0.1. WikiText-103 and FineWeb-Edu both favor cross-entropy training on held-out accuracy, cross-entropy, perplexity, and even full Brier score. This comparison concerns the whole fitted model at this budget, not just the loss value optimized during training. Sources: [language-model setup](LM_PROTOCOL.md), [FineWeb-Edu setup](FINEWEB_PROTOCOL.md), and the result tables above.

2. **Brier training can be competitive or better on accuracy in normally initialized image models.** In the CIFAR-10 follow-up, Brier training has higher mean accuracy and lower full Brier score under both optimizers. Under AdamW, cross-entropy training still has lower test cross-entropy; the accuracy advantage does not make Brier training better on every metric. The follow-up changes learning-rate selection, schedule horizon, early stopping, and checkpoint selection relative to the six-epoch comparison. The reversal therefore cannot be attributed to more epochs alone. It also does not establish convergence: training stops by validation patience or the 30-epoch cap. Sources: [follow-up setup](STABILITY_PROTOCOL.md) and [results](stability_cifar/results.md).

3. **The collapsed-bias result is an optimizer-and-initialization interaction.** With class-0 bias initialized to −12, stochastic gradient descent (SGD) with Brier training leaves that bias unchanged and class-0 recall at zero in all 16 learning-rate/seed combinations. Other parameters still move. At the published SGD learning rate 0.05, cross-entropy training recovers class-0 recall on all four seeds. With Adam at learning rate 0.001, both losses recover class-0 recall on all four seeds. The wider Adam grid recovers in all 64 tested combinations. Tiny Brier gradients and zero recorded bias steps support a specific optimization failure; they do not establish that the entire model is motionless or that Brier training must fail under ordinary initialization. This study used zero weight decay, so weight decay is not necessary for this observed failure. Seed 0 was not rerun. Sources: [measured replication](collapsed_replication/MEASURED.md) and [setup](collapsed_replication/PROTOCOL.md).

4. **Lower calibration error is a different result from better predictive probabilities.** The language-model Brier-trained runs have lower top-label expected calibration error, using 15 equal-width bins of maximum predicted probability. Cross-entropy-trained runs nevertheless have lower held-out cross-entropy and lower full Brier score. This calibration measure does not evaluate every class probability and cannot by itself establish better full-distribution probability estimates. Source: [metric definitions](metrics.py).

5. **The study establishes neither a universal winner nor a weight-decay explanation.** Learning rates and, for the follow-up, checkpoints were selected by each loss's own validation objective, not by a shared accuracy objective. The language models cover one architecture and short fixed budgets; FineWeb-Edu uses one fixed shard and document-ID-based splits, not its whole corpus. Weight decay was fixed within each comparison, so there is no measured decay ablation here. The same coefficient also means different operations for AdamW's decoupled decay and SGD's coupled penalty. These runs do not establish common converged minima, explain historical cross-entropy adoption, or rule out better Brier results under other budgets, architectures, or tuning choices.

## Statistical limits and corrections

All p-values are exact, two-sided sign-flip tests of a seed-paired difference distribution symmetric about zero; signs must be exchangeable under the null. They are unadjusted for multiple comparisons. The smallest attainable two-sided p-values are 0.25 for three seeds, 0.125 for four, and 0.0625 for five. Thus no comparison here can reach p < 0.05 with that test, even when every observed seed favors one loss. This does not establish equality.

The 95% bootstrap intervals resample training seeds. Existing WikiText-103, CIFAR-10, and early AlexNet intervals are reproduced from their original tables, which use 10,000 draws with seed 0. New FineWeb-Edu intervals enumerate all 27 ordered three-seed resamples; new collapsed-bias recall intervals enumerate all 256 ordered four-seed resamples, using inverse empirical cumulative-distribution quantiles. With only three to five seeds, these are coarse, conditional estimates of seed variability, not reliable guarantees of population coverage. They do not capture dataset, architecture, or tuning uncertainty; the CIFAR-10 follow-up also reuses seed 0's selected tuning run among its five final seeds. A bootstrap interval excluding zero is not a replacement for the sign-flip test; these intervals are not its inversion. Do not pool unrelated settings into one larger seed sample.

Inherited presentation issues are recorded here without changing the source files: the collapsed-bias prose says the minimum four-seed sign-flip p-value is 0.0625, but 0.125 is correct for its two-sided test; FineWeb-Edu record labels still say `wikitext103` because the trainer function was reused, while its matching protocol and token provenance identify FineWeb-Edu. FineWeb-Edu's setup prose says test encoding occurs after selection, but its code and saved metadata store the split earlier and evaluate it after selection; encoding alone is not evidence of test-driven selection. The old top-level language-model BLOCKED entry describes an earlier download failure; the separate WikiText-103 and FineWeb-Edu runs subsequently finished. The old language-model setup's proposed decay comparison is superseded by the actual fixed-decay CIFAR-10 setup and outputs.

## Reproduce the added calculations

From `/Users/brandomiranda/.worktrees/ultimate-utils/brier-ce-pilot`:

```bash
python3 experiments/06_brier_vs_ce_alexnet_dense_lm/fineweb_edu/compute_paired_differences.py --check
python3 experiments/06_brier_vs_ce_alexnet_dense_lm/collapsed_replication/compute_paired_recall.py
```

Both commands read saved records only. Original trainer-output files remain unchanged from `a1fd697`. No training, model evaluation, cluster operation, or email is part of this analysis.

**Copy-ready message:** I finished the five-part comparison: cross-entropy training has higher accuracy and better probability scores in both language-model datasets at our tested budget, but Brier training has higher accuracy in the longer CIFAR-10 follow-up. The collapsed-bias failure depends on the optimizer and initialization; with only three to five seeds per comparison, these results support conditional empirical findings rather than a universal ranking of the losses.

**TLDR-end:** [ultimate-utils: Brier versus cross-entropy] `experiments/06_brier_vs_ce_alexnet_dense_lm` supports cross-entropy training for the two tested language-model settings and a conditional Brier advantage on CIFAR-10 accuracy. FineWeb-Edu's new paired analysis comes entirely from saved records; optimizer, initialization, metric, and training procedure limit the conclusion.
