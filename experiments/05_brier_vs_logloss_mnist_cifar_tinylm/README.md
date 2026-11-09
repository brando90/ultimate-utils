# Brier score versus log loss on a CNN and a small language model

**Doc link:** <https://github.com/brando90/ultimate-utils/blob/main/experiments/05_brier_vs_logloss_mnist_cifar_tinylm/README.md>

**TLDR:** Log loss is the training loss in deep learning because the softmax gradient is \(q - e_y\), which stays alive on a collapsed coordinate. The Brier score's softmax gradient is multiplied by that tiny probability. This run compares the two losses on MNIST, CIFAR-10, and a byte-level transformer, with and without weight decay, including a coordinate whose bias starts at \(-12\). The address above resolves after a push to `main`.

Checked out on 11-06-2026. The population fact from `experiments/04_brier_vs_weighted_divergence/` is that a free probability vector fit by either proper score is the histogram. A neural net is not a free probability vector. Weight decay is not the reason log loss became usable: AlexNet trained softmax log loss with weight decay in 2012.

## Measurement contract

Written before the cluster run.

| Metric | Definition | Range | Predicted, default init | Predicted, class or byte bias \(-12\), no bias decay |
| --- | --- | --- | --- | --- |
| Test accuracy | argmax matches the label | 0 to 1 | MNIST, both losses, above 0.97 after 2 epochs | Log loss above 0.90. Brier below 0.85, because class 0 stays weak |
| Collapsed-label recall | accuracy on the examples whose label is the suppressed coordinate | 0 to 1 | Both above 0.90 on MNIST | Log loss above 0.80. Brier below 0.20 |
| Mean logit of the suppressed coordinate | mean pre-softmax logit on the test set | real | Both near the other logits | Log loss above \(-2\). Brier below \(-8\) |
| Test expected log loss | mean \(-\log q(y)\) | at least 0 | Lower for the log-loss run than for the Brier run | Log loss finite and much smaller than Brier |
| Test Brier score | mean \(\|q - e_y\|_2^2\) | 0 to 2 | Lower or close for the Brier run if it converges | Higher for the Brier run, because \(q\) stays away from \(e_y\) on the suppressed label |

Weight decay is AdamW on weights only. Biases are not decayed, so decay cannot drag a bias of \(-12\) back to \(0\) by itself. Prediction: weight decay \(10^{-4}\) does not swap the collapsed-label recall ordering.

CIFAR-10 uses the same predictions with lower absolute accuracy: both default runs above \(0.55\) after 4 epochs of the small CNN. The collapsed-label gap has the same direction.

The byte transformer predicts the next byte on Tiny Shakespeare. Six runs: both losses at bias \(0\) with no decay, and both losses at bias \(-12\) with decay \(0\) and \(10^{-4}\). Default init: log loss has the lower validation expected log loss. Suppressed byte is the space character, bias \(-12\): under log loss its mean probability on positions where the target is space exceeds \(0.05\); under the Brier score it stays below \(0.01\). Weight decay does not swap that ordering.

A local gradient check, before any training, must show that at bias \(-12\) the log-loss gradient on that bias is near \(-1\) times the frequency of the label, and the Brier gradient on that bias is smaller than \(10^{-4}\).

A second probe freezes the convolutional features and trains only the linear head. Prediction, written before that probe: with bias \(-12\), log loss reaches class-0 recall above \(0.50\) and a mean class-0 logit above \(-2\) after 2 epochs. The Brier head stays under class-0 recall \(0.20\), and its mean class-0 logit stays below \(-8\). The full network can move the representation, so it is allowed to revive a class even when that class's own gradient is tiny. The frozen head cannot.

## What would falsify the claim

If the Brier run with bias \(-12\) and no bias decay reaches collapsed-label recall within \(0.10\) of the log-loss run on MNIST, the softmax-Jacobian stall does not survive this CNN. Report that rather than explaining it away.

## Run

```bash
python experiments/05_brier_vs_logloss_mnist_cifar_tinylm/train_compare.py --suite check
python experiments/05_brier_vs_logloss_mnist_cifar_tinylm/train_compare.py --suite all --device cuda
```

The cluster job writes `results/summary.json` and `results.md`.
