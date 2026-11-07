# CIFAR-10 convergence and optimizer follow-up

The closed AlexNet comparison used AdamW only, weight decay 5e-4, and 6 CIFAR-10 epochs. Those 6 epochs are not a converged run. The held-out scores from that run do not choose the settings below.

This is not a rerun of `experiments/06_sgd_vs_adam_collapsed_bias` on `main` commit `50d908b`. That study is seed 0, plain SGD and plain Adam, no weight decay, and a collapsed bias of -12. It is finished. The multi-seed extension of that study is a separate job in `collapsed_replication/`. It does not replace this file. Weight decay is not the question here. The coefficient stays 5e-4 for every run in this CIFAR-10 follow-up.

## Conditions

| optimizer | loss | learning rates, seed 0 |
| --- | --- | --- |
| AdamW, betas (0.9, 0.95) | cross-entropy | 3e-4, 1e-3, 3e-3, 1e-2 |
| AdamW, betas (0.9, 0.95) | full Brier score | 3e-4, 1e-3, 3e-3, 1e-2 |
| SGD, momentum 0.9 | cross-entropy | 1e-2, 3e-2, 1e-1, 3e-1 |
| SGD, momentum 0.9 | full Brier score | 1e-2, 3e-2, 1e-1, 3e-1 |

Four learning rates for each optimizer. The numerical values differ because a shared number is not a matched step size. AdamW's weight decay is decoupled. SGD's weight decay is the coupled penalty on the same parameter groups. The coefficient is 5e-4 in both, and it is not varied.

## Initialization and data

`AlexNetSmallImage.reset_parameters`: Kaiming normal convolutions, linear weights with standard deviation 0.01, every bias 0. No coordinate starts at -12. The trainer rejects a bias that is not 0.

CIFAR-10, no augmentation, the same 5,000-image validation split, seed 123, as the closed run. Batch 128. Gradient clip 1. bfloat16 autocast, loss in float32. Cosine schedule on a 30-epoch horizon, down to 0.1 times the peak. Early stopping does not compress that horizon.

## Rules

Maximum 30 epochs. Patience 3 on the validation value of the loss being trained. The scored checkpoint is the best validation epoch.

- Divergence: non-finite loss or gradient.
- Stall: best validation accuracy below 0.20, or validation loss improved by less than 1% from epoch 0.
- Success: neither.

Seed 0 maps the learning rates. Seeds 0 through 4 train at the selected rate. Seed 0 is the map run. Selection is the lowest validation value of the training loss among successes, then higher validation accuracy, then the smaller learning rate. The official test split is scored once after those four selections are stored.

Paired differences use the training seed. With 5 seeds the smallest nonzero sign-flip p-value is 0.0625. Compare cross-entropy-trained minus Brier-trained inside one optimizer, and AdamW minus SGD inside one loss. Do not compare the raw loss values across the two losses.

## Budget

One GPU that is not the WikiText GPU `GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697`. Upper bound 16 map runs plus 16 extra final seeds, 30 epochs each. The watchdog is twice the measured epoch time, times 30, times 32 runs, plus one hour. Kept checkpoints stay under 6 GB. The run refuses to start below 8 GB free.
