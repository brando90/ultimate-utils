# DRAFT — do not send

Status: 11-07-2026. The WikiText-103 run and the collapsed-bias replication are not finished. Send this only after both have final numbers, in the existing thread `1a1b4d2c17576f10`, from brandojazz@gmail.com. To brando.science@gmail.com. BCC brandojazz@gmail.com, brando.science@gmail.com, brando9@stanford.edu, and brando@vals.ai.

## Distinctions that stay in the sent version

Three different comparisons are in progress or already finished. They do not substitute for each other.

1. Seed 0 on `main` commit `50d908b`, directory `experiments/06_sgd_vs_adam_collapsed_bias/`, is finished and is not being rerun. The network is three convolutions, 32, 64, and 128, then a linear layer of 10. MNIST, 3 epochs. `torch.optim.SGD(lr=0.05, momentum=0.9)` and `torch.optim.Adam(lr=0.001)`. No weight decay. Default initialization, then the final bias is 0, and the collapsed rows set class 0 to -12. SGD with the Brier score left class-0 recall at 0 and the bias at -12, accuracy 0.8694. SGD log loss reached class-0 recall 0.993. Adam reached 0.972 and 0.978. Normal-init SGD accuracy was 0.9717 and 0.9643. That stall is a seed-0 mechanism. Adam taking a normal step there is not a guarantee for every seed or learning rate.

2. Experiment 05 is AdamW. Weight decay 0 or 1e-4 is applied to weights only. Biases are excluded. Both losses recovered from bias -12. It is not the SGD stall.

3. The running 123,551,232-parameter WikiText-103 model is `torch.optim.AdamW`, betas (0.9, 0.95), weight decay 0.1 on parameters with ndim at least 2, and weight decay 0 on the rest. The output projection has `bias=False`. Nothing in that run is initialized to -12. The schedule is a warmup of `max(20, min(steps // 20, 500))` and then a cosine down to 0.1 times the selected peak. The peak is chosen on the validation split from 1e-4, 3e-4, 1e-3, and 3e-3. This is a normal-initialization pretraining comparison. It does not replicate the SGD saturation stall. A collapsed language-model condition would be a separate, prespecified run. It has not been started.

The historical guess that weight decay is why cross-entropy is standard is dropped. AlexNet in 2012 already used weight decay 5e-4. Experiment 06 used none.

The CIFAR-10 AlexNet follow-up is a fourth, separate job: normal initialization, fixed weight decay 5e-4, AdamW against SGD with momentum, up to 30 epochs. It is not the collapsed-bias replication.

## Collapsed replication, filled 11-07-2026

Seeds 1–4 are done. At SGD learning rate 0.05 and start bias −12, Brier class-0 recall was 0/980 on every seed and the bias stayed −12. Log loss recovered, recalls 0.9704 to 0.9908. The same Brier stall held at learning rates 0.01, 0.1, and 0.2, 16 of 16 runs, with a class-0 bias step of 0 while other parameters still moved. Adam at learning rate 0.001 recovered both losses. The Brier probe gradient was about −1e-7. Adam's first full update had L2 about 0.29 for both losses. Two SGD log-loss runs at learning rate 0.2 dropped to accuracy near 0.10. That is a high-learning-rate failure, not the frozen coordinate. The full table is `collapsed_replication/results.md`.

## Numbers still to fill in before sending

- WikiText-103: realized steps, tokens, seeds, validation and held-out accuracy, cross-entropy, perplexity, full Brier score, top-label 15-bin ECE, and failures. At 09:11 PT the log was still in the final phase: `final cross_entropy seed 0 lr 0.0003 step 1600`.
- CIFAR-10 follow-up, if it has finished by then. At 09:11 PT it was still in the seed-0 map.

Do not describe the 6-epoch AlexNet result as a converged minimum. Do not describe n=3 or n=4 sign-flip tests as significance at 0.05.
