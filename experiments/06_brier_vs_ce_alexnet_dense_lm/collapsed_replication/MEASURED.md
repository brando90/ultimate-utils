# Measured replication, seeds 1–4

Finished 11-07-2026 at 08:57:48 PT. Exit 0. Status DONE. 128 full-CNN runs, 32 logit runs, and 32 frozen-head runs. Seed 0 was not trained. The machine was mercury1, GPU `GPU-0ec99cca-57d0-31e9-eb6a-c27a4061554a`. The class-0 count on the official MNIST test split is 980.

## Published full-CNN point

`torch.optim.SGD`, learning rate 0.05, momentum 0.9, start bias −12:

| Seed | Log-loss recall | Log-loss final bias | Brier recall | Brier final bias |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9704 (951/980) | −8.4208 | 0.0000 (0/980) | −12.0000 |
| 2 | 0.9755 (956/980) | −8.5570 | 0.0000 (0/980) | −12.0000 |
| 3 | 0.9908 (971/980) | −10.5800 | 0.0000 (0/980) | −12.0000 |
| 4 | 0.9908 (971/980) | −8.7339 | 0.0000 (0/980) | −12.0000 |

The four log-loss-minus-Brier recall differences are all positive. Their mean is 0.9819. With 4 seeds the smallest nonzero sign-flip p-value is 0.0625.

`torch.optim.Adam`, learning rate 0.001, start bias −12: both losses recovered on every seed. Log-loss recall was 0.9408, 0.9694, 0.9643, and 0.9827. Brier recall was 0.9010, 0.9510, 0.9786, and 0.9796. The mean recall difference is 0.0117, and the four signs are not the same.

The Brier probe gradient on the class-0 bias was between −2.28e-7 and −1.41e-7. The log-loss probe gradient was about −0.09 to −0.14. For SGD and the Brier score, the first class-0 bias step was 0. The rest of the parameter update was not: its L2 norm was between 0.00112 and 0.00157 at learning rate 0.05. Adam's first class-0 bias step was about 0.0009 for the Brier score and 0.0010 for log loss, and the full update L2 was about 0.28 to 0.30 for both losses.

## The wider grid

SGD, Brier score, start bias −12, learning rates 0.01, 0.05, 0.1, and 0.2, all 4 seeds: 16 of 16 stalled. The final bias change was 0, and the first bias step was 0. Other parameters still moved. The largest first-step update L2 in those 16 runs was 0.00627.

Adam recovered in all 64 of its full-CNN cells: 4 learning rates, 4 seeds, both losses, and both start biases.

Normal initialization at the published learning rates recovered for both optimizers and both losses. SGD log-loss accuracy was 0.9746, 0.9742, 0.9710, and 0.9725.

Cells that did not follow the main pattern:

- SGD, log loss, learning rate 0.2, start bias 0, seed 4: class-0 recall 0, accuracy 0.1135, final bias 0.0006. The prespecified rule calls this stalled. Accuracy is near 0.1, so the whole run failed at that learning rate.
- SGD, log loss, learning rate 0.2, start bias −12, seeds 1 and 3: the bias moved by about +10.8, accuracy fell to about 0.10, and class-0 recall stayed 0. These are partial, not a frozen coordinate.
- SGD, Brier score, start bias 0, learning rate 0.01, seed 1: recall 0.2694, accuracy 0.2711. Partial.
- SGD, Brier score, start bias 0, learning rate 0.2, seed 1: class-0 recall 0, accuracy 0.8788, final bias −0.6710. The bias moved, so this is not the frozen-coordinate stall.

## Logits and frozen head

At the published settings, seeds 1–4: SGD with the Brier score stayed at a change of 0 for momentum 0 and momentum 0.9, on the logits and on the frozen head. Log loss under SGD moved. Adam moved the logits and recovered the frozen head for both losses.
