# Brier score versus cross-entropy on AlexNet and a dense language model

**Status:** DONE

**Last updated:** 2026-11-06T23:53:08-08:00

**Doc link:** https://github.com/brando90/ultimate-utils/blob/main/experiments/06_brier_vs_ce_alexnet_dense_lm/results.md

The link resolves after this branch is on `main`. Definitions and the procedure fixed before the run are in `README.md` in this folder.

## Phases

| phase | state |
| --- | --- |
| controls | DONE |
| mnist_tune | DONE |
| cifar10_tune | DONE |
| mnist_final | DONE |
| cifar10_final | DONE |
| lm_tune | BLOCKED |
| lm_final | BLOCKED |

## Placement

- hostname: mercury2.stanford.edu
- cuda_visible_devices: GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697
- gpu_uuid_env: GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697
- device: cuda
- torch: 2.7.0+cu126
- started: 2026-11-06T23:01:37-08:00
- device_count: 1
- device_name: NVIDIA RTX A4000

## Notes

- 2026-11-06T23:01:37-08:00 collapsed-logit check {"cross_entropy": {"loss": 8.200767517089844, "bias_gradient": -0.4999995827674866, "q0_mean": 6.80456537338614e-07}, "brier": {"loss": 1.0002005100250244, "bias_gradient": -7.545760354332742e-07, "q0_mean": 6.80456537338614e-07}}
- 2026-11-06T23:30:06-08:00 epoch rule: first CIFAR epoch 16.6s, CIFAR epochs 6, MNIST epochs 2, CNN seeds 5
- 2026-11-06T23:53:08-08:00 language-model data blocked: no WikiText corpus could be stored: wikitext-103-raw-v1: File is not a zip file | wikitext-2-raw-v1: File is not a zip file

## Calibration

A predicted probability is calibrated, in the sense used here, when the model's top label is correct about as often as that top probability says. If the largest softmax probability is 0.7, the predicted class should be right on about 70% of those examples.

The ECE column is top-label expected calibration error. It is not classwise calibration, and the bins are equal width, not equal count. The AlexNet numbers were produced by `_ece` in `train.py`. `metrics.expected_calibration_error` uses the same rule.

- Confidence is the maximum softmax probability. An example is correct when the argmax class equals the label.
- There are 15 bins. The edges are `torch.linspace(0, 1, 16)` in float32, so the nominal width is 1/15.
- The first bin includes both endpoints. Each later bin is open on the left and closed on the right, so a confidence that lands exactly on an interior edge stays in the lower bin.
- An empty bin is skipped. It adds no gap and does not change the denominator. A nonempty bin is weighted by its count divided by the number of examples in that evaluation.
- The gap in a bin is the absolute difference between the fraction correct and the mean confidence. ECE is the sum of gap times weight. It is between 0 and 1.

Each held-out row is one training seed on the official test split, 10,000 images. The paired ECE row is the mean of the seed-wise differences, cross-entropy-trained minus Brier-trained. It is not one ECE pooled across seeds. With 5 seeds the smallest nonzero sign-flip p-value is 0.0625.

On CIFAR-10 the five raw Brier-trained test ECE values average 0.054083, and the five cross-entropy-trained values average 0.073726. The paired mean difference is 0.019643. On MNIST those averages are 0.004715 and 0.003570, and the paired difference is −0.001145.

A lower ECE is not, by itself, a better probability. On CIFAR-10 the cross-entropy-trained models have higher accuracy and a lower full Brier score, while their top-label ECE is higher. Their cross-entropy on this test split is nearly the same as the Brier-trained models. Accuracy, cross-entropy, the full Brier score, and this ECE are different measurements.

Per-example confidences and per-bin counts were not saved for the AlexNet runs, so there is no reliability diagram for those rows. The scalar ECE in the tables is the whole calibration record.

## Held-out test

Learning rates were selected on the validation split. These rows are the official test split, scored after that selection.

### mnist held-out test

| loss | seed | learning rate | accuracy | cross-entropy | perplexity | Brier score | local square | ECE | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| brier | 0 | 0.001 | 0.9924 | 0.0380995 | 1.03883 | 0.013094 | 0.007006 | 0.00490368 | 10000 |
| brier | 1 | 0.001 | 0.9921 | 0.0366507 | 1.03733 | 0.0129909 | 0.0071944 | 0.00423882 | 10000 |
| brier | 2 | 0.001 | 0.9926 | 0.029327 | 1.02976 | 0.0117422 | 0.00636548 | 0.00394489 | 10000 |
| brier | 3 | 0.001 | 0.9916 | 0.0471383 | 1.04827 | 0.0142542 | 0.00766324 | 0.00541602 | 10000 |
| brier | 4 | 0.001 | 0.9922 | 0.0377845 | 1.03851 | 0.012947 | 0.00705048 | 0.00507039 | 10000 |
| cross_entropy | 0 | 0.003 | 0.9935 | 0.0264184 | 1.02677 | 0.0106209 | 0.00571879 | 0.00378652 | 10000 |
| cross_entropy | 1 | 0.003 | 0.9927 | 0.0262783 | 1.02663 | 0.0117845 | 0.0067078 | 0.00360555 | 10000 |
| cross_entropy | 2 | 0.003 | 0.9921 | 0.029486 | 1.02993 | 0.0122253 | 0.00690836 | 0.00228483 | 10000 |
| cross_entropy | 3 | 0.003 | 0.9929 | 0.0299018 | 1.03035 | 0.0114751 | 0.00632615 | 0.00374996 | 10000 |
| cross_entropy | 4 | 0.003 | 0.9919 | 0.0298732 | 1.03032 | 0.012257 | 0.00663889 | 0.00442224 | 10000 |

Paired difference is the cross-entropy-trained run minus the Brier-trained run. The 95% interval resamples training seeds. p-val is a two-sided sign-flip test.

| metric | mean difference | 95% interval | p-val | n seeds |
| --- | ---: | --- | ---: | ---: |
| accuracy | 0.00044 | [-0.0002, 0.00108] | 0.3125 | 5 |
| cross_entropy | -0.00940846 | [-0.0142604, -0.00431529] | 0.125 | 5 |
| perplexity | -0.0097403 | [-0.0147977, -0.00445537] | 0.125 | 5 |
| brier | -0.00133311 | [-0.00234215, -0.000324069] | 0.125 | 5 |
| local_square | -0.000595925 | [-0.00114704, -2.40099e-05] | 0.25 | 5 |
| expected_calibration_error | -0.00114494 | [-0.00155388, -0.000735996] | 0.0625 | 5 |

### cifar10 held-out test

| loss | seed | learning rate | accuracy | cross-entropy | perplexity | Brier score | local square | ECE | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| brier | 0 | 0.001 | 0.7981 | 0.646324 | 1.90851 | 0.292354 | 0.18195 | 0.0534947 | 10000 |
| brier | 1 | 0.001 | 0.7988 | 0.64042 | 1.89728 | 0.289451 | 0.17966 | 0.0544354 | 10000 |
| brier | 2 | 0.001 | 0.7956 | 0.650892 | 1.91725 | 0.295662 | 0.184989 | 0.055171 | 10000 |
| brier | 3 | 0.001 | 0.7966 | 0.649283 | 1.91417 | 0.294763 | 0.182287 | 0.0593104 | 10000 |
| brier | 4 | 0.001 | 0.7984 | 0.625813 | 1.86976 | 0.288478 | 0.182173 | 0.0480037 | 10000 |
| cross_entropy | 0 | 0.001 | 0.8136 | 0.635083 | 1.88718 | 0.277102 | 0.161867 | 0.0726046 | 10000 |
| cross_entropy | 1 | 0.001 | 0.813 | 0.643711 | 1.90353 | 0.27845 | 0.161905 | 0.0762901 | 10000 |
| cross_entropy | 2 | 0.001 | 0.8126 | 0.624216 | 1.86678 | 0.276343 | 0.163145 | 0.0636894 | 10000 |
| cross_entropy | 3 | 0.001 | 0.8084 | 0.660884 | 1.9365 | 0.285469 | 0.16627 | 0.0800575 | 10000 |
| cross_entropy | 4 | 0.001 | 0.8119 | 0.651466 | 1.91835 | 0.282159 | 0.163912 | 0.0759865 | 10000 |

Paired difference is the cross-entropy-trained run minus the Brier-trained run. The 95% interval resamples training seeds. p-val is a two-sided sign-flip test.

| metric | mean difference | 95% interval | p-val | n seeds |
| --- | ---: | --- | ---: | ---: |
| accuracy | 0.0144 | [0.01288, 0.01588] | 0.0625 | 5 |
| cross_entropy | 0.000525799 | [-0.0159334, 0.0155599] | 0.9375 | 5 |
| perplexity | 0.00107508 | [-0.0300803, 0.0296195] | 0.9375 | 5 |
| brier | -0.0122371 | [-0.0165008, -0.00844564] | 0.0625 | 5 |
| local_square | -0.0187921 | [-0.0204354, -0.017161] | 0.0625 | 5 |
| expected_calibration_error | 0.0196426 | [0.0134098, 0.0247611] | 0.0625 | 5 |

### wikitext held-out test

| loss | seed | learning rate | accuracy | cross-entropy | perplexity | Brier score | local square | ECE | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

No paired held-out test rows yet.

The language-model rows are empty. `lm_tune` and `lm_final` are BLOCKED. Both legacy research.metamind.io downloads were 467-byte S3 redirect documents, not zip files, so no language-model training ran. The tables above are the AlexNet comparison only.

**TLDR-end:** [ultimate-utils: brier vs cross-entropy] Status DONE for the AlexNet runs. The dense language model did not train.

