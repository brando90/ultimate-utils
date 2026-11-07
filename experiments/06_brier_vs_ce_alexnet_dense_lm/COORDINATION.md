# Coordination: two Brier studies, two owners

Created: 2026-11-07 08:02 PST
Last updated: 2026-11-07 08:02 PST

This file is the handoff. It is not a message in the local Cursor chat. The committed copy is `experiments/06_brier_vs_ce_alexnet_dense_lm/COORDINATION.md` on branch `cursor/brier-ce-pilot-076e`. A readable copy sits untracked at `experiments/COORDINATION_brier_ce_076e.md` in the `main` checkout on this Mac. It was not committed on `main`.

## Ownership

| Work | Owner | Where |
| --- | --- | --- |
| Math, KL, weighted divergence, preprint | local tab, branch `main` | `experiments/04_brier_vs_weighted_divergence/`, commit `50d908b` |
| AdamW small CNN, MNIST, CIFAR-10, byte transformer, bias -12, weight decay 0 and 1e-4 | local tab, branch `main` | `experiments/05_brier_vs_logloss_mnist_cifar_tinylm/` |
| Seed-0 SGD versus Adam, collapsed bias -12 | local tab, branch `main` | `experiments/06_sgd_vs_adam_collapsed_bias/` |
| AlexNet AdamW multi-seed comparison, closed | this branch | `experiments/06_brier_vs_ce_alexnet_dense_lm/` on `cursor/brier-ce-pilot-076e` |
| WikiText-103 language model, running | this branch | mercury2 pid 1065070, tmux `brier-ce-lm-076e` |
| CIFAR-10 convergence and optimizer follow-up | this branch | `STABILITY_PROTOCOL.md` |

The two `experiments/06_*` directories are different studies. Keep both names. This branch does not contain `04`, `05`, or `06_sgd_vs_adam_collapsed_bias`. Do not rename either directory to remove the collision.

## What main already showed

Commit `50d908b5429a7f621d095d54f60b895e968c95ff` on `origin/main` (merge; the experiment commit is `36ab43c`). Read from that tree:

- `experiments/06_sgd_vs_adam_collapsed_bias/results.md`: status DONE. SGD momentum 0.9, learning rate 0.05, bias -12, 3 MNIST epochs: cross-entropy class-0 recall 0.993, accuracy 0.9717; Brier class-0 recall 0, accuracy 0.8694. Adam, learning rate 0.001, bias -12: recalls 0.972 and 0.978. Normal initialization, SGD, both losses above 0.96 accuracy. GPU `GPU-87a5b8eb-3e08-4391-6434-e73da2a03519`. The results file says the tmux session was closed. A live `tmux ls` on mercury2 on 11-07-2026 shows only `brier-ce-lm-076e`.
- `experiments/05_brier_vs_logloss_mnist_cifar_tinylm/results.md`: seed 0, AdamW. Both losses trained, including bias -12. Weight decay 1e-4 matched weight decay 0. CIFAR-10 at 4 epochs stayed near accuracy 0.42 to 0.45. Frozen features, 2 epochs, bias -12: class-0 recall 0 for both losses.
- That tab dropped the guess that AlexNet's 2012 weight decay of 5e-4 is why cross-entropy is used.

The collapsed-gradient gap is a seed-0 mechanism. Adam moving both coordinates there is not a guarantee for every optimizer, width, or training length.

## What this branch actually trains

Closed CNN, from `train.py` and `/lfs/mercury2/0/brando9/brier_ce_pilot_076e/full/state.json`:

- Optimizer AdamW, betas (0.9, 0.95), gradient clip 1, batch 128.
- Weight decay 0.0005 on parameters with ndim at least 2. Recorded on the CIFAR-10 test rows.
- Biases initialized to 0 in `AlexNetSmallImage.reset_parameters`. No training run starts a bias at -12. A separate linear check sets a class-0 bias to -12 only to measure the gradient.
- Cosine schedule to 0.1 times the peak. CIFAR-10 finals: 6 epochs. MNIST finals: 2 epochs. Seeds 0 through 4.
- Selected CIFAR-10 learning rate 0.001 for both losses. MNIST: cross-entropy 0.003, Brier 0.001.
- 35,855,178 parameters.

Running language model, from `train_lm_fixed.py` and `lm_fixed/protocol.json` read on 11-07-2026:

- AdamW, betas (0.9, 0.95), weight decay 0.1, clip 1, batch 8, sequence 256.
- Embedding initialization standard deviation 0.02. Attention and SwiGLU projections have no bias.
- WikiText-103 raw, `Salesforce/wikitext` config `wikitext-103-raw-v1`, tiktoken `gpt2`.
- Tune 8,000,000 tokens, 3,921 steps, learning rates 1e-4, 3e-4, 1e-3, 3e-3. Final 118,032,360 tokens, 57,859 steps, 3 seeds.
- Not Tiny Shakespeare. Not a collapsed byte.

## Not duplicated

This branch will not rerun the seed-0 collapsed-bias SGD-versus-Adam diagnostic, the small-CNN weight-decay pair at 1e-4, or the byte transformer. The weight-decay factorial that had been downloading CIFAR-10 on mercury1 was stopped before `protocol.json` existed and before any epoch. The replacement is the convergence and optimizer follow-up in `STABILITY_PROTOCOL.md`: normal initialization, fixed weight decay 5e-4, AdamW and SGD, separate learning-rate grids, up to 30 epochs, 5 seeds.

## Active job

mercury2, tmux `brier-ce-lm-076e`, pid 1065070, GPU `GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697`. Do not kill it. Do not use that GPU.
