# Collapsed-bias replication: new seeds and learning rates

Created: 11-07-2026

This is an add-on. It does not replace the CIFAR-10 follow-up in `STABILITY_PROTOCOL.md`, and it does not change the running WikiText-103 protocol.

The seed-0 code and results stay where they are: `experiments/06_sgd_vs_adam_collapsed_bias/` on `main` commit `50d908b`. That directory is not edited. Seed 0 is not trained again. The numbers below are the values printed in that `results.md`.

## Seed 0, full MNIST CNN, already measured

Architecture in `train_validate.py`: three convolutions, 32 then 64 then 128, each 3×3 with padding 1, ReLU, two stride-2 pools, adaptive average pool to 1×1, then `Linear(128, 10)`. `torch.manual_seed(0)`, default module initialization, then the final bias is set to 0 and, when the row says so, class 0 is set to -12. Convolution biases stay at the default. Three epochs. MNIST, `ToTensor` only, train batch 128, test batch 256.

Optimizers, with no weight-decay argument: `torch.optim.SGD(lr=0.05, momentum=0.9)` and `torch.optim.Adam(lr=0.001)`. These are not AdamW.

| Loss | Optimizer | Start bias | Accuracy | Class-0 recall | Final bias |
| --- | --- | ---: | ---: | ---: | ---: |
| log loss | SGD | 0 | 0.9717 | 0.988 | -0.50 |
| Brier score | SGD | 0 | 0.9643 | 0.982 | -0.37 |
| log loss | SGD | -12 | 0.9717 | 0.993 | -8.07 |
| Brier score | SGD | -12 | 0.8694 | 0.000 | -12.00 |
| log loss | Adam | -12 | 0.9385 | 0.972 | -11.99 |
| Brier score | Adam | -12 | 0.9567 | 0.978 | -11.91 |

The printed initial class-0 bias gradient was -0.117 for log loss and -1.75e-7 for the Brier score. The Brier SGD row is the stall: class-0 recall 0 and the bias still -12. Log loss under SGD moved that bias to -8.07. Adam recovered class 0 for both losses. The integer class-0 count was not stored; the code divides by the number of class-0 images in the official test split. Adam with start bias 0 was not a seed-0 CNN row.

Experiment 05 is a different optimizer. It uses AdamW, and weight decay 0 or 1e-4 applies only to weights. Biases are excluded. Both losses recovered there. Do not cite experiment 05 as this stall.

## What this job trains

Same `SmallCNN`, same MNIST loaders, same three epochs, same two losses, same absence of weight decay. `torch.manual_seed(seed)` is called in the same place, immediately before `SmallCNN()`.

Seeds 1, 2, 3, and 4. Seed 0 is copied from the table above and is not executed.

Full-network learning rates, four per optimizer, published value included:

| Optimizer | Momentum or betas | Learning rates |
| --- | --- | --- |
| `torch.optim.SGD` | momentum 0.9, library weight decay 0 | 0.01, 0.05, 0.1, 0.2 |
| `torch.optim.Adam` | library defaults, betas (0.9, 0.999), eps 1e-8, weight decay 0 | 1e-4, 3e-4, 1e-3, 3e-3 |

Start bias 0 and -12 for every optimizer, loss, and seed. Bias 0 is the normal control. Adam at bias 0 is new; seed 0 did not run it.

That is 2 optimizers × 4 learning rates × 2 start biases × 2 losses × 4 seeds = 128 CNN runs. Every cell is reported. The official test split is scored the way seed 0 scored it. There is no learning-rate selection and no validation split in the source experiment, so none is added here.

Two cheaper levels use only the published settings, seeds 1–4, and are not given the wide grid:

- Pure logits, 2,000 steps, class 0 starts at -12. SGD learning rate 0.1 with momentum 0 and 0.9. Adam learning rates 0.01 and 0.001.
- Frozen head, 8 epochs, final bias class 0 set to -12. Same four optimizer settings.

32 logit runs and 32 frozen-head runs.

## Labels and measurements

Training always runs the fixed epoch count. These labels are applied afterward:

- Stalled: class-0 recall is 0 and the absolute bias change is under 1e-3.
- Recovered: class-0 recall is at least 0.5.
- Partial: anything else.

Each CNN run stores the probe gradient of the class-0 bias, the first training step's gradient, the first-step absolute bias change, the first-step parameter-update L2 norm, the mean absolute bias change, and the class-0 bias, class-0 hit count, class-0 count, recall, and accuracy after each epoch.

Paired comparisons at the published learning rates use seeds 1–4 only. With 4 seeds the smallest nonzero sign-flip p-value is 0.0625. Seed 0 stays in the table as the committed row.

## Compute

One GPU that is not `GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697` and not `GPU-071c409d-a1bb-9fd7-1239-a271b8ce1889`. The first of those is the WikiText run. The second is the CIFAR-10 follow-up.

Smoke times one CNN run after MNIST is loaded: seed 1, SGD, learning rate 0.05, Brier score, start bias -12. The watchdog is twice that measured run time, times 160, plus 15 minutes, and it is capped at 6 hours. It stops a hung process. It is not a 12-hour cutoff of a healthy run. A crash retries up to 4 times and skips finished keys. Refuse to start below 2 GB free. Checkpoints of this network are small; the output stays under 1 GB.

The 123,551,232-parameter WikiText run is the other requested piece. It is already running. Its optimizer is AdamW with normal initialization. It is not this stall, and this job does not add a collapsed coordinate to it.
