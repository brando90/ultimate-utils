# Brier score versus cross-entropy on AlexNet and a dense language model

**Status:** RUNNING

**Last updated:** 2026-11-06T23:03:42-08:00

**Doc link:** https://github.com/brando90/ultimate-utils/blob/main/experiments/06_brier_vs_ce_alexnet_dense_lm/results.md

The link resolves after this branch is on `main`. Definitions and the procedure fixed before the run are in `README.md` in this folder.

## Phases

| phase | state |
| --- | --- |
| controls | DONE |
| mnist_tune | RUNNING |

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

## Held-out test

Learning rates were selected on the validation split. These rows are the official test split, scored after that selection.

### mnist held-out test

| loss | seed | learning rate | accuracy | cross-entropy | perplexity | Brier score | local square | ECE | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

No paired held-out test rows yet.

### cifar10 held-out test

| loss | seed | learning rate | accuracy | cross-entropy | perplexity | Brier score | local square | ECE | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

No paired held-out test rows yet.

### wikitext held-out test

| loss | seed | learning rate | accuracy | cross-entropy | perplexity | Brier score | local square | ECE | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

No paired held-out test rows yet.

This is one architecture, one corpus, and the token count written in the phase notes. It is not a result about a frontier language model.

**TLDR-end:** [ultimate-utils: brier vs cross-entropy] Status RUNNING. AlexNet on MNIST and CIFAR-10, and one dense causal language model, trained with the full Brier score and with cross-entropy under matched data, initialization, and step budgets.

