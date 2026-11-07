# CIFAR-10 stability follow-up

This is the longer CIFAR-10 study. It replaces a separate 30-epoch rerun at the already selected learning rate. The 6-epoch held-out test from 11-06-2026 was already scored. Those scores do not choose the learning rates, the patience, or the epoch cap below.

The question is whether the full Brier score is more stable than cross-entropy, and whether the AlexNet weight decay of 5e-4 is what makes cross-entropy train. A finite loss that stays near the uniform predictor is stalled learning.

## Conditions

Two losses, two weight-decay values, every combination:

| loss | weight decay |
| --- | ---: |
| cross-entropy | 0 |
| cross-entropy | 5e-4 |
| full Brier score | 0 |
| full Brier score | 5e-4 |

There is no third weight-decay value. Two values answer whether this baseline changes training. Another value would double the map.

The full Brier score is the mean of the sum over all 10 classes. It is not `(1 - q_y)^2`.

## What is matched

Same AlexNet used for 32×32 images, same initialization for a seed, same CIFAR-10 official training images, same 5,000-image validation split from seed 123, same batch order, AdamW betas (0.9, 0.95), gradient clip 1, bfloat16 autocast, loss in float32, batch 128, no augmentation. The cosine schedule is defined on a 30-epoch horizon for every run. Early stopping does not compress that horizon. Weight decay applies to parameters with two or more dimensions, as in the first study.

## Learning rates

Each condition sees the same six learning rates: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2. The count is equal. The selected rate can differ. A shared numerical learning rate is not treated as a matched update, because the two losses have different scales. Each run records the gradient norm and the realized parameter-update norm.

Seed 0 maps all 24 runs. The successful range is that one-seed map. Seeds 0 through 4 are trained only at the selected learning rate. Seed 0 is the map run, not a second training.

## Rules fixed before the final seeds

Maximum 30 epochs. After each epoch, score the validation value of the loss being trained. Stop when that value fails to improve for 3 epochs. The scored checkpoint is the best validation epoch, not the last epoch.

- Divergence: a non-finite loss or a non-finite gradient norm. That run stops and stays in the table.
- Stall: the best epoch has validation accuracy below 0.20, or the validation loss has improved by less than 1% relative to epoch 0. Accuracy 0.20 is twice the 10-class chance rate. A flat Brier predictor is this outcome.
- Success: neither of the above.

Selection uses seed 0 and validation only. Among successes, take the lowest validation value of the training loss. Ties break toward higher validation accuracy, then the smaller learning rate. If a condition has no success, and some run stayed finite, the final seeds still use that least-bad learning rate and the condition is labeled as having no successful learning rate. If every learning rate diverged, that condition has no final seeds.

The official test split is scored once, after the four selections are on disk, on the best validation checkpoint. It is not used to change a selection.

## What is reported

For every learning rate: outcome, best epoch, steps, wall seconds, gradient-norm mean and maximum, update-norm mean and maximum, fraction of steps whose gradient norm exceeds 10 times the epoch median, and the count of confident wrong validation examples. A confident wrong example has maximum softmax probability at least 0.99 and the wrong class. The diagnostic is the mean true-class logit gradient of each loss on those examples. It is not a selection criterion.

The paired differences use the training seed as the unit. With 5 seeds the smallest nonzero sign-flip p-value is 0.0625. The comparisons are, within one loss, weight decay 5e-4 minus weight decay 0, and, within one weight decay, cross-entropy-trained minus Brier-trained. Validation loss values are not compared across the two losses.

## Budget

One GPU. The upper bound is 24 map runs plus 16 extra final seed runs, each at most 30 epochs. The watchdog is twice the measured epoch time, times 30 epochs, times those 40 runs, plus one hour. It kills a hung process. It is not the scientific stopping rule. Unselected map checkpoints are deleted after selection. The kept files are the metric record and the best checkpoint of each final seed. The CIFAR-10 AlexNet has 35,855,178 parameters, about 144 MB per checkpoint. Twenty final checkpoints are about 2.9 GB. The plan stays under 6 GB. The run refuses to start below 8 GB free. It does not delete the WikiText job or any preexisting files.

Placement is one idle GPU that is not the WikiText GPU `GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697`.
