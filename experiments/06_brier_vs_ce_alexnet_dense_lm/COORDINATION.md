# Coordination: two Brier studies, two owners

Created: 11-07-2026
Last updated: 11-07-2026

This file is the handoff. It is not a message in the local Cursor chat. The committed copy is `experiments/06_brier_vs_ce_alexnet_dense_lm/COORDINATION.md` on branch `cursor/brier-ce-pilot-076e`. A readable copy sits untracked at `experiments/COORDINATION_brier_ce_076e.md` in the `main` checkout on this Mac. It was not committed on `main`.

## Ownership

| Work | Owner | Where |
| --- | --- | --- |
| Math, KL, weighted divergence, preprint | local tab, branch `main` | `experiments/04_brier_vs_weighted_divergence/`, commit `50d908b` |
| AdamW small CNN, MNIST, CIFAR-10, byte transformer, bias -12, weight decay 0 and 1e-4 | local tab, branch `main` | `experiments/05_brier_vs_logloss_mnist_cifar_tinylm/` |
| Seed-0 plain SGD versus plain Adam, collapsed bias -12, no weight decay | local tab, branch `main` | `experiments/06_sgd_vs_adam_collapsed_bias/` |
| AlexNet AdamW multi-seed comparison, closed | this branch | `experiments/06_brier_vs_ce_alexnet_dense_lm/` on `cursor/brier-ce-pilot-076e` |
| WikiText-103 language model, running, AdamW, normal initialization | this branch | mercury2 pid 1065070, tmux `brier-ce-lm-076e` |
| CIFAR-10 convergence follow-up, normal initialization, running | this branch | `STABILITY_PROTOCOL.md`, mercury1 pid 3621040 |
| Collapsed-bias seeds 1–4 and learning-rate grid | this branch | `collapsed_replication/` |

The two `experiments/06_*` directories are different studies. Keep both names. This branch does not contain `04`, `05`, or `06_sgd_vs_adam_collapsed_bias`. Do not rename either directory to remove the collision.

## What main already showed

Commit `50d908b5429a7f621d095d54f60b895e968c95ff` on `origin/main` (merge; the experiment commit is `36ab43c`). Read from that tree:

- `experiments/06_sgd_vs_adam_collapsed_bias/train_validate.py` and `results.md`, status DONE, seed 0 only. The network is three convolutions, 32, 64, and 128, then `Linear(128, 10)`. MNIST, 3 epochs. `torch.manual_seed(0)`, default initialization, final bias set to 0, then class 0 set to -12 on the collapsed rows. Optimizers are `torch.optim.SGD(lr=0.05, momentum=0.9)` and `torch.optim.Adam(lr=0.001)`. No weight-decay argument, so both stay at 0. This is not AdamW. Printed full-CNN results: SGD Brier score, start -12, class-0 recall 0.000, final bias -12.00, accuracy 0.8694. SGD log loss, start -12, class-0 recall 0.993, final bias -8.07, accuracy 0.9717. Adam, start -12, class-0 recall 0.972 and 0.978. Normal-init SGD accuracy 0.9717 and 0.9643. The integer class-0 count was not stored. GPU `GPU-87a5b8eb-3e08-4391-6434-e73da2a03519`. That session is closed.
- `experiments/05_brier_vs_logloss_mnist_cifar_tinylm/train_compare.py`: AdamW only. `param_groups` puts every bias, and every 1-dimensional parameter, in the group with weight decay 0. The other group gets 0 or 1e-4. Both losses recovered from bias -12. Do not cite experiment 05 as the SGD stall.
- The historical guess that weight decay explains the use of cross-entropy is dropped. AlexNet in 2012 already used weight decay 5e-4, and experiment 06 used none.

The collapsed-gradient gap is a seed-0 mechanism. Adam taking a step there is not a guarantee for every seed, learning rate, or network.

## What this branch actually trains

Closed CNN, from `train.py` and `/lfs/mercury2/0/brando9/brier_ce_pilot_076e/full/state.json`:

- Optimizer AdamW, betas (0.9, 0.95), gradient clip 1, batch 128.
- Weight decay 0.0005 on parameters with ndim at least 2. Recorded on the CIFAR-10 test rows.
- Biases initialized to 0 in `AlexNetSmallImage.reset_parameters`. No training run starts a bias at -12. A separate linear check sets a class-0 bias to -12 only to measure the gradient.
- Cosine schedule to 0.1 times the peak. CIFAR-10 finals: 6 epochs. MNIST finals: 2 epochs. Seeds 0 through 4.
- Selected CIFAR-10 learning rate 0.001 for both losses. MNIST: cross-entropy 0.003, Brier 0.001.
- 35,855,178 parameters.

Running language model, from `train_lm_fixed.py`, `models.py`, and `lm_fixed/protocol.json`. The protocol is not being changed.

- Class: `torch.optim.AdamW`. Not `torch.optim.SGD` and not `torch.optim.Adam`.
- Betas (0.9, 0.95). Gradient clip 1. Batch 8. Sequence 256. Loss in float32 under bfloat16 autocast.
- Weight decay 0.1 on parameters with ndim at least 2. Weight decay 0 on ndim below 2, which here are the RMSNorm scales. `adamw_groups` in `models.py`.
- Output projection: `nn.Linear(d, vocab, bias=False)`. Attention and SwiGLU projections also have `bias=False`. No output bias exists, and no coordinate is set to -12. `collapsed_logit_check` runs on a separate linear layer inside the objective check. It is not a training condition.
- Embedding initialization: normal, standard deviation 0.02. RMSNorm weights start at 1.
- Schedule: warmup `max(20, min(steps // 20, 500))`, then cosine from the peak down to 0.1 times the peak (`lr_at`).
- Peaks on the validation split: 1e-4, 3e-4, 1e-3, 3e-3. Selection is the lower validation value of the loss being trained, then higher validation accuracy, then the smaller peak. Selection has not run yet.
- WikiText-103 raw, `Salesforce/wikitext`, config `wikitext-103-raw-v1`, tiktoken `gpt2`. Tune 8,000,000 tokens, 3,921 steps. Final 118,032,360 tokens, 57,859 steps, seeds 0, 1, and 2.
- This is a normal-initialization pretraining comparison. It does not replicate the SGD saturation stall. A collapsed language-model condition would be a separate prespecified run. It is not started.

## Not duplicated

This branch does not rerun seed 0 of `experiments/06_sgd_vs_adam_collapsed_bias`, and it does not edit that directory. The extension is `collapsed_replication/`: seeds 1–4, the same network, plain SGD and plain Adam, no weight decay, the same bias conditions, and wider learning-rate grids. The design is in `collapsed_replication/PROTOCOL.md`.

The CIFAR-10 job in `STABILITY_PROTOCOL.md` keeps running. It is normal initialization, AdamW against SGD, fixed weight decay 5e-4. It is not the collapsed-bias replication.

Experiment 05 is not repeated.

## Active jobs, checked 11-07-2026

- WikiText-103: mercury2, tmux `brier-ce-lm-076e`, pid 1065070, GPU `GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697`. Do not kill it. Do not use that GPU. Do not change its protocol.
- CIFAR-10 follow-up: mercury1, tmux `brier-ce-stability-076e`, pid 3621040, `CUDA_VISIBLE_DEVICES=GPU-071c409d-a1bb-9fd7-1239-a271b8ce1889`, still in smoke. Do not kill it. Do not use that GPU.
- Collapsed-bias replication: separate output `collapsed_out`, separate session `brier-ce-collapsed-076e`, a different idle GPU.
