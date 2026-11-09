# Brier score versus log loss, CNN and tiny language model

**Status:** DONE (11-06-2026)

**Last updated:** 11-06-2026

**Doc link:** <https://github.com/brando90/ultimate-utils/blob/main/experiments/05_brier_vs_logloss_mnist_cifar_tinylm/results.md>

The address resolves after a push to `main`. Predictions were written in `README.md` before the run. One seed, seed 0. No error bars.

Machine: SNAP `mercury2`, one RTX A4000, UUID `GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697`. Main job `brier_logloss_05`, 6:10:08 PM to 6:21:35 PM PT, exit 0. Frozen-head probe on a second idle A4000, UUID `GPU-87a5b8eb-3e08-4391-6434-e73da2a03519`, 6:16:51 PM to 6:17:09 PM PT, exit 0. Both tmux sessions were closed after the logs were read. Python was the existing veribench environment, torch 2.7.0+cu126. Weight decay is AdamW on weights only. Biases are not decayed.

| Phase | State |
| --- | --- |
| Gradient check at bias \(-12\) | DONE |
| MNIST, 8 conditions, 2 epochs | DONE |
| CIFAR-10, 8 conditions, 4 epochs | DONE |
| Byte transformer, 6 conditions, 1500 steps | DONE |
| Frozen-head probe, 4 conditions, 2 epochs | DONE |

## Gradient check

On a linear layer with class-0 bias \(-12\), half the labels were class 0. The log-loss gradient on that bias was \(-0.500\). The Brier gradient was \(-7.59 \times 10^{-7}\). Mean \(q(0)\) was \(6.86 \times 10^{-7}\).

A separate laptop check, 512 random features, AdamW at learning rate \(10^{-3}\), 900 steps: both biases moved from \(-12\) to about \(-11.03\) (log loss) and \(-11.04\) (Brier). Adam's step follows the sign of a steady gradient, so the million-fold smaller Brier gradient did not produce a million-fold smaller step.

## MNIST, final test metrics

| Loss | Weight decay | Bias | Accuracy | Class-0 recall | Expected log loss | Brier | Mean class-0 logit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| log loss | 0 | 0 | 0.9516 | 0.9745 | 0.1714 | 0.0754 | 1.12 |
| log loss | 0 | \(-12\) | 0.9194 | 0.9112 | 0.2820 | 0.1260 | \(-0.82\) |
| log loss | \(10^{-4}\) | 0 | 0.9494 | 0.9663 | 0.1735 | 0.0771 | 1.03 |
| log loss | \(10^{-4}\) | \(-12\) | 0.9235 | 0.9214 | 0.2666 | 0.1188 | 0.05 |
| Brier | 0 | 0 | 0.9534 | 0.9714 | 0.1825 | 0.0728 | 2.47 |
| Brier | 0 | \(-12\) | 0.9577 | 0.9714 | 0.1618 | 0.0674 | 3.48 |
| Brier | \(10^{-4}\) | 0 | 0.9539 | 0.9704 | 0.1796 | 0.0725 | 2.23 |
| Brier | \(10^{-4}\) | \(-12\) | 0.9575 | 0.9684 | 0.1628 | 0.0669 | 3.59 |

## CIFAR-10, final test metrics

| Loss | Weight decay | Bias | Accuracy | Class-0 recall | Expected log loss | Brier | Mean class-0 logit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| log loss | 0 | 0 | 0.4442 | 0.673 | 1.530 | 0.689 | 0.21 |
| log loss | 0 | \(-12\) | 0.3504 | 0.807 | 1.715 | 0.776 | \(-1.10\) |
| log loss | \(10^{-4}\) | 0 | 0.4512 | 0.672 | 1.522 | 0.687 | 0.18 |
| log loss | \(10^{-4}\) | \(-12\) | 0.3510 | 0.804 | 1.719 | 0.776 | \(-1.05\) |
| Brier | 0 | 0 | 0.4208 | 0.640 | 1.624 | 0.706 | \(-0.36\) |
| Brier | 0 | \(-12\) | 0.4110 | 0.703 | 1.651 | 0.720 | \(-0.20\) |
| Brier | \(10^{-4}\) | 0 | 0.4221 | 0.651 | 1.620 | 0.706 | \(-0.38\) |
| Brier | \(10^{-4}\) | \(-12\) | 0.4179 | 0.742 | 1.638 | 0.724 | 0.20 |

Four epochs of this small net stay in the mid-0.4s. That is under-training of this architecture. Both losses sit in the same band.

## Tiny Shakespeare bytes, step 1500

Validation next-byte scores. The suppressed coordinate is the space byte.

| Loss | Weight decay | Bias | Expected log loss | Brier | \(q(\text{space})\) when the target is space | Mean space logit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| log loss | 0 | 0 | 2.015 | 0.728 | 0.597 | 3.71 |
| log loss | 0 | \(-12\) | 2.035 | 0.732 | 0.589 | 0.52 |
| log loss | \(10^{-4}\) | \(-12\) | 2.035 | 0.732 | 0.589 | 0.52 |
| Brier | 0 | 0 | 2.136 | 0.714 | 0.603 | 2.23 |
| Brier | 0 | \(-12\) | 2.185 | 0.719 | 0.592 | 0.13 |
| Brier | \(10^{-4}\) | \(-12\) | 2.185 | 0.719 | 0.592 | 0.13 |

At step 1 with bias \(-12\), \(q(\text{space})\) on space targets was \(4.4 \times 10^{-8}\) for log loss and \(3.1 \times 10^{-8}\) for the Brier score. At step 250 those probabilities were \(0.261\) and \(0.00147\). At step 500 the Brier run was at \(0.445\).

## Frozen convolutional features

The head alone was trained for 2 epochs on features from one frozen random MNIST network.

| Loss | Start mean logit | Class-0 recall | Mean class-0 logit | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| log loss, bias 0 | \(-0.083\) | 0.871 | 0.008 | 0.2775 |
| log loss, bias \(-12\) | \(-12.017\) | 0.000 | \(-8.15\) | 0.2304 |
| Brier, bias 0 | \(-0.083\) | 0.848 | 0.035 | 0.1964 |
| Brier, bias \(-12\) | \(-12.017\) | 0.000 | \(-4.99\) | 0.2031 |

## Verdict against the written predictions

The gradient check matches the prediction. The network predictions do not. On MNIST, CIFAR-10, and the byte transformer, a bias of \(-12\) was revived by both losses. Weight decay \(10^{-4}\) copies the matching no-decay run. With the features frozen, neither loss restored class-0 recall in 2 epochs at learning rate \(10^{-3}\). The Brier mean logit moved from \(-12.02\) to \(-4.99\), and the log-loss mean logit moved to \(-8.15\).

Default MNIST accuracy was \(0.9516\) and \(0.9534\), short of the predicted \(0.97\). CIFAR-10 accuracy was \(0.444\) and \(0.421\), short of the predicted \(0.55\).

**TLDR-end:** [ultimate-utils: brier vs log loss nets] On mercury2, log loss and the Brier score both trained the small CNN and the byte transformer, including from a bias of \(-12\). Weight decay \(10^{-4}\) did not change the comparison. The stall predicted for the Brier score shows up in the raw gradient, and Adam walks out of it.
