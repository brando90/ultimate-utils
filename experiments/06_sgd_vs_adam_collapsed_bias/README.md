# SGD versus Adam on a collapsed softmax coordinate

**Doc link:** <https://github.com/brando90/ultimate-utils/blob/main/experiments/06_sgd_vs_adam_collapsed_bias/README.md>

**TLDR:** The claim under test is that a Brier score on a softmax has a near-zero gradient on a collapsed coordinate, so plain SGD leaves that coordinate frozen, while log loss moves it. Adam takes a step of about the learning rate whenever the gradient keeps one sign, so it moves both. The address above resolves after a push to `main`.

Checked out on 11-06-2026, before the run. Seed 0. Class 0, or the pure logit coordinate 0, starts at bias or value \(-12\).

## Measurement contract

| Level | What is trained | Metric | Predicted |
| --- | --- | --- | --- |
| Logits | One 10-way softmax, parameters are the logits, 2,000 SGD or Adam steps, batch of fixed labels with class 0 on 10% of rows | Change in logit 0 | SGD, log loss: increase by more than 5. SGD, Brier: increase by less than 0.05. Adam, learning rate \(10^{-2}\), both losses: each increases by more than 5, and the two increases differ by less than a factor of 3. Adam, learning rate \(10^{-3}\): both increases are between 0.5 and 5 and within a factor of 3 of each other |
| Frozen head | MNIST features from one frozen random CNN, 8 epochs, batch 128 | Class-0 bias change and class-0 recall | SGD learning rate \(0.1\), momentum 0 and momentum 0.9: log loss bias increases by more than 5 and recall exceeds 0.50. Brier bias increases by less than 0.5 and recall stays under 0.05. Adam learning rate \(10^{-3}\): both bias changes are between 0.5 and 8 and within a factor of 3 of each other. Adam learning rate \(10^{-2}\): both bias increases exceed 5, stay within a factor of 3, and both recalls exceed 0.30 |
| Full CNN | The small MNIST CNN, 3 epochs | Class-0 recall and mean class-0 logit | SGD, momentum 0.9, learning rate \(0.05\), bias 0: both losses reach accuracy above 0.90. Same optimizer, bias \(-12\): log-loss recall above 0.70. Brier recall under 0.20 and mean logit under \(-5\). Adam, learning rate \(10^{-3}\), bias \(-12\): both recalls above 0.50 |

On every run that starts at \(-12\), the initial gradient on that coordinate is near \(-0.1\) for log loss, because class 0 is about a tenth of the labels, and smaller than \(10^{-4}\) in absolute value for the Brier score.

## What falsifies the claim

- SGD on the Brier score moves the collapsed logit or bias by more than 1 at the logit level or the frozen-head level.
- SGD on log loss moves it by less than 1. That means the learning rate was too small, and the comparison is inconclusive.
- On the full CNN, SGD with the Brier score reaches class-0 recall above 0.50. Representation learning then revives the class without a live gradient on that coordinate, and the practical claim fails for this network.

## Run

```bash
python experiments/06_sgd_vs_adam_collapsed_bias/train_validate.py --suite all --device cuda
```
