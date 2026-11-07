# Brier score versus cross-entropy: AlexNet and a dense language model

**Doc link:** https://github.com/brando90/ultimate-utils/blob/main/experiments/06_brier_vs_ce_alexnet_dense_lm/README.md

**TLDR:** The full multiclass Brier score and cross-entropy are both proper scores, and they are not the same training loss. This run trains an AlexNet adapted to MNIST and CIFAR-10, and one dense causal language model of about 100M to 300M parameters, with a separate learning-rate grid for each loss and the same data, initialization, and step count. The address above resolves after a push to `main`.

Checked before the GPU run on 11-06-2026. The population identities are in `experiments/04_brier_vs_weighted_divergence/`. A short CNN and byte-model run from the same day is in `experiments/05_brier_vs_logloss_mnist_cifar_tinylm/` and is a different comparison: it does not tune the learning rate separately, and it is not this architecture.

## Losses

Cross-entropy is the mean of \(-\log q_y\), natural log, no label smoothing, not divided by the class count. \(q\) is the softmax of the logits.

The Brier score is the mean of the sum over every class,

\[
\sum_k (q_k - 1\{k = y\})^2 = \|q\|^2 + 1 - 2 q_y.
\]

It is not divided by the class count. The local square \((1 - q_y)^2\) drops every class that did not occur. It is logged as a diagnostic and is never the training loss.

On 10 classes the Brier score of one example lies in \([0, 2]\). A uniform forecast has Brier score \((K-1)/K\) and cross-entropy \(\ln K\).

## Why cross-entropy was already the standard, and why weight decay is not the reason

For an alphabet of at least 3 symbols, the logarithmic score is the only smooth local proper score. Local means the score looks only at the probability of the symbol that occurred. The Brier score is proper and is not local. That characterization is the one recorded from MIT 6.437 lecture 13 in experiment 04.

AlexNet (Krizhevsky, Sutskever, and Hinton, 2012) trained with multinomial logistic loss, which is cross-entropy, and it also used weight decay 0.0005. Weight decay was already on. The Transformer (Vaswani et al., 2017) also trains with cross-entropy. The AdamW weight decay 0.1 used below for the language model is a later default (Loshchilov and Hutter, 2019). Both losses in this run share the weight decay of that model, so a gap between them is not a weight-decay gap.

The learning signal is the gradient of the training loss with respect to the logits. For cross-entropy it is \(q - e_y\). The observed-class component is \(q_y - 1\), which stays near \(-1\) when \(q_y\) is near 0. For the Brier score the softmax Jacobian multiplies that error by factors of \(q\), so a collapsed coordinate has a tiny Brier gradient. Adam can still step on a tiny steady gradient. A small gradient is not, by itself, a stuck weight, and a flat loss is not evidence that two constrained networks share a minimum.

## Models

CIFAR-10 images stay 32×32. The 2012 AlexNet first convolution is 11×11 with stride 4 and assumes a 224×224 image; on 32×32 it collapses the map. This adaptation uses five 3×3 stride-1 convolutions, channels 64, 192, 384, 256, 256, three 2×2 stride-2 pools, an adaptive average pool to 4×4, and the 4096-4096 classifier. Local response normalization is omitted. Classifier dropout is 0.5. MNIST uses the same network with one input channel. There is no data augmentation, so both losses see the same pixels.

The 2012 optimizer was SGD with momentum. Both losses here use AdamW, betas (0.9, 0.95), so the optimizer is not a confound. CNN weight decay is \(5 \times 10^{-4}\), the 2012 magnitude. Language-model weight decay is 0.1. Matrix weights are decayed. Biases and RMSNorm scales are not.

The language model is a dense causal Transformer: RMSNorm, rotary positions, SwiGLU, 12 layers, width 768, 12 heads, sequence length 256. The token embedding and the output layer share one matrix. The tokenizer is the fixed GPT-2 byte-pair encoding (50257 tokens). It is not fit on this corpus. The parameter count must land between 100M and 300M. This is not a frontier model and not a mixture of experts. Training uses bfloat16 autocast. The loss is computed in float32. Gradients are clipped at global norm 1. The learning rate warms up and then follows a cosine down to 10% of the peak.

## Data

MNIST and CIFAR-10 use the official training images. A validation split of 5000 images is carved with seed 123 and saved. The official test split is not constructed until that dataset's learning rate is on disk.

The corpus is WikiText-103 raw (Salesforce; text from Wikipedia). WikiText-2 raw is the fallback only if WikiText-103 cannot be stored, and that choice is made before any language-model step. Train, validation, and test files are encoded separately. The test file is not read until the language-model learning rate is on disk. Headers in the raw files are left as they are.

The add-one unigram is fit on the training tokens only and scored on the validation targets. It is the language-model baseline.

## Learning rates, seeds, and the clock

Each loss gets the same grid and the same number of trials. The selected learning rate is the one with the lowest validation value of that loss. Ties break toward higher validation accuracy, then the smaller learning rate.

- CNN grid, seed 0, one epoch: \(3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}, 10^{-2}\).
- Language-model grid, seed 0, 2M tokens: \(10^{-4}, 3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}\).

After the first CIFAR-10 tune trial, let \(t\) be its wall time in seconds.

| \(t\) | CIFAR-10 final epochs | MNIST final epochs |
| --- | ---: | ---: |
| \(t \le 60\) | 6 | 2 |
| \(60 < t \le 180\) | 3 | 2 |
| \(t > 180\) | 1 | 1 |

CNN seeds are 0 through 4 if the projected final CNN time at 5 seeds is under 2.5 hours, and 0 through 2 otherwise. The projection uses \(t\) for both datasets, which overestimates MNIST.

Language-model final seeds are 3 if each run can see at least 20M tokens in the time left, and 2 otherwise. Every final language-model run uses the same step count. A step is one batch. Tokens are batch × 255. If fewer than 20 steps fit, that phase is skipped and marked skipped.

The full job's deadline is 12 hours after its first start, including a restart. One GPU. A restart skips finished markers and does not add another 12 hours.

Batch size starts at 128 for the CNN and 8 for the language model. One out-of-memory retry drops the language-model batch to 4 for both losses. A non-finite loss stops that run and keeps the row. Nothing else stops a run early.

## Measurement contract

Written before the GPU run. Code paths: `losses.py`, `metrics.py`, `train.py`.

| Quantity | Oracle, \(q = e_y\) | Uniform | Intended, this budget | Degenerate, all-zero logits |
| --- | --- | --- | --- | --- |
| Accuracy | 1 | about \(1/K\), argmax ties break to class 0 | MNIST validation above 0.95 for both losses when the final has 2 epochs. CIFAR-10 validation above 0.45 at 6 epochs, above 0.35 at 3, above 0.10 at 1. No prediction of which loss is higher. | fraction of class 0 |
| Brier score | 0 | \((K-1)/K\) | finite, and below the uniform value if accuracy is above chance | 0.9 for 10 classes |
| Cross-entropy | 0 | \(\ln K\) | finite. Language model: below the add-one unigram on validation if each final run has at least 20M tokens; otherwise only below \(\ln 50257 \approx 10.825\) | \(\ln 10\) for 10 classes |
| Perplexity | 1 | \(K\) | \(\exp(\)cross-entropy\()\) on the same rows | 10 for 10 classes |
| Expected calibration error, top label, 15 equal-width bins of the maximum probability | 0 | 0 when the top-label accuracy equals that confidence | in \([0, 1]\) | absolute gap between the class-0 frequency and 0.1, because every probability is 0.1 and argmax ties break to class 0 |
| Gradient of the mean loss w.r.t. a class-0 bias set to \(-12\), half the labels class 0 |  |  | cross-entropy \(< -0.2\) (near \(-0.5\)); absolute Brier gradient \(< 10^{-3}\) |  |
| Non-finite losses | 0 | 0 | 0 | 0 |

Calibration here means the top label is correct about as often as its probability. The error is top-label expected calibration error, not classwise and not equal-count. Fifteen bins use `torch.linspace(0, 1, 16)`. The first bin includes both ends. Later bins are open on the left and closed on the right. Empty bins are omitted from the sum, and each nonempty bin is weighted by its count over the evaluation size. The gap is the absolute difference between the fraction correct and the mean confidence in the bin. The reported ECE is that weighted sum for one seed and one split. The paired table then averages cross-entropy-trained minus Brier-trained across seeds. A lower ECE does not by itself mean higher accuracy, lower cross-entropy, or a lower full Brier score. The all-zero-logit cell above is only that special case of the same formula.

An invalid or incomplete row stays in the denominator of the phase table. The paired difference is the cross-entropy-trained seed minus the Brier-trained seed. The 95% interval is a percentile bootstrap over training seeds, 10000 draws, seed 0. The p-value is a two-sided sign-flip test. With 3 seeds the p-value sits on a grid of eighths and cannot resolve 0.05. This language-model budget is not a result about a frontier language model.

## Machine

Open SNAP nodes have no `squeue`. The job is a direct process on one idle GPU, checked again at launch. The shared shell sets `CUDA_VISIBLE_DEVICES` to the least-used index; the launcher overrides that for this process only and does not edit the shell startup file.

On 11-06-2026 at 22:41 PT, mercury2's ten RTX A4000 GPUs were idle and its existing `veribench` environment had PyTorch 2.7.0+cu126. Local scratch had 15596736512 bytes free. That is not enough spare room to raise the file cap. This job plans at most 6GB of new files under `/lfs/mercury2/0/brando9/brier_ce_pilot_076e` and refuses to start the full run below 8GB free. It does not delete preexisting files. Checkpoints of optimizer state are not kept; the retained record is the metric JSON. A killed seed is trained again from its matched initialization.

## Run

```bash
python experiments/06_brier_vs_ce_alexnet_dense_lm/test_pilot.py
bash experiments/06_brier_vs_ce_alexnet_dense_lm/run_bounded.sh smoke
bash experiments/06_brier_vs_ce_alexnet_dense_lm/run_bounded.sh full
```

Smoke must print `SMOKE_PASS` before the 12-hour run. Smoke does not open a test split.

**TLDR-end:** [ultimate-utils: brier vs cross-entropy] The full Brier score and cross-entropy are trained on an adapted AlexNet and one dense causal language model, with separate learning-rate grids, matched seeds, and the test split held out until selection.
