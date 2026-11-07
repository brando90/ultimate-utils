# SGD versus Adam on a collapsed softmax coordinate

**Status:** DONE (11-06-2026)

**Last updated:** 11-06-2026

**Doc link:** <https://github.com/brando90/ultimate-utils/blob/main/experiments/06_sgd_vs_adam_collapsed_bias/results.md>

The address resolves after a push to `main`. Predictions were written in `README.md` before the run. Seed 0. One run per condition.

Machine: SNAP mercury2, RTX A4000 `GPU-87a5b8eb-3e08-4391-6434-e73da2a03519`. Job `sgd_adam_06`, 11:03:34 PM to 11:06:17 PM PT, exit 0. The session was closed afterward. GPU 0 was already busy and was not used.

| Phase | State |
| --- | --- |
| Pure logits, 2,000 steps | DONE |
| Frozen MNIST head, 8 epochs | DONE |
| Full MNIST CNN, 3 epochs | DONE |

## Pure logits

Class 0 started at \(-12\). Its share of the labels was \(1/10\). The initial gradient was \(-0.100\) for log loss and \(-1.52 \times 10^{-7}\) for the Brier score.

| Loss | Optimizer | Learning rate | Momentum | Change in logit 0 |
| --- | --- | ---: | ---: | ---: |
| log loss | SGD | 0.1 | 0 | \(+10.80\) |
| log loss | SGD | 0.1 | 0.9 | \(+10.80\) |
| Brier | SGD | 0.1 | 0 | \(0\) |
| Brier | SGD | 0.1 | 0.9 | \(0\) |
| log loss | Adam | 0.01 | 0 | \(+6.00\) |
| Brier | Adam | 0.01 | 0 | \(+9.69\) |
| log loss | Adam | 0.001 | 0 | \(+2.00\) |
| Brier | Adam | 0.001 | 0 | \(+3.04\) |

Every logit prediction matched. SGD moved log loss and left the Brier coordinate at \(-12\) to the printed precision. Momentum did not change that. Adam moved both, and the two changes stayed within a factor of 1.7.

## Frozen head

| Loss | Optimizer | Learning rate | Momentum | Bias change | Class-0 recall | Mean class-0 logit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| log loss | SGD | 0.1 | 0 | \(+8.68\) | 0.769 | \(-1.24\) |
| log loss | SGD | 0.1 | 0.9 | \(+5.92\) | 0.588 | \(-1.37\) |
| Brier | SGD | 0.1 | 0 | \(0\) | 0.000 | \(-12.02\) |
| Brier | SGD | 0.1 | 0.9 | \(0\) | 0.000 | \(-12.02\) |
| log loss | Adam | 0.001 | 0 | \(+1.93\) | 0.780 | \(-3.57\) |
| Brier | Adam | 0.001 | 0 | \(+2.87\) | 0.831 | \(-0.33\) |
| log loss | Adam | 0.01 | 0 | \(+0.49\) | 0.671 | \(-4.06\) |
| Brier | Adam | 0.01 | 0 | \(+1.25\) | 0.743 | \(-1.07\) |

The SGD rows matched. Adam at \(10^{-3}\) matched. Adam at \(10^{-2}\) recovered class-0 recall for both losses, which was predicted, and did not move either bias by more than 5, which was also predicted and did not happen. The weights carried that recovery.

## Full MNIST CNN, 3 epochs

| Loss | Optimizer | Starting bias | Accuracy | Class-0 recall | Mean class-0 logit | Final bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| log loss | SGD, momentum 0.9, lr 0.05 | 0 | 0.9717 | 0.988 | 0.94 | \(-0.50\) |
| Brier | SGD, momentum 0.9, lr 0.05 | 0 | 0.9643 | 0.982 | 2.51 | \(-0.37\) |
| log loss | SGD, momentum 0.9, lr 0.05 | \(-12\) | 0.9717 | 0.993 | 1.38 | \(-8.07\) |
| Brier | SGD, momentum 0.9, lr 0.05 | \(-12\) | 0.8694 | 0.000 | \(-12.83\) | \(-12.00\) |
| log loss | Adam, lr 0.001 | \(-12\) | 0.9385 | 0.972 | 0.96 | \(-11.99\) |
| Brier | Adam, lr 0.001 | \(-12\) | 0.9567 | 0.978 | 4.34 | \(-11.91\) |

The initial collapsed gradient on the CNN bias was \(-0.117\) for log loss and \(-1.75 \times 10^{-7}\) for the Brier score. Every full-network prediction matched. With a normal initialization both losses reached accuracy above 0.96 under SGD. With bias \(-12\), SGD log loss recovered class 0 and SGD Brier never predicted it. Adam recovered both.

## Verdict

The claim holds for SGD, including on the full network: a collapsed Brier coordinate stays at its start value, and log loss moves it. Adam moves both. The one missed prediction is the size of the Adam bias step on the frozen head at learning rate \(10^{-2}\).

**TLDR-end:** [ultimate-utils: sgd vs adam collapsed bias] SGD with the Brier score left a coordinate that started at \(-12\) exactly there, on the logits, on a frozen head, and on the MNIST CNN, where class-0 recall stayed 0. Log loss and Adam both recovered it.
