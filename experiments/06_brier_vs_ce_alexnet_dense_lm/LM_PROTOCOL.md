# Language-model protocol, fixed before the final runs

The AlexNet comparison in `full/` is closed. Its weights were not kept. CIFAR-10 training loss was still falling at epoch 6, so those six epochs are a fixed short budget, not a converged minimum. MNIST validation accuracy after two epochs is about 0.99. A longer CNN run would be a new matched training job, labeled as a follow-up, using the learning rates already selected. It is not part of this launch. The one GPU goes to the language model first.

## Corpus

Salesforce WikiText-103 raw, config `wikitext-103-raw-v1`, from the authors' Hugging Face dataset `Salesforce/wikitext`. License CC-BY-SA-3.0 and GFDL. Paper: Merity, Xiong, Bradbury, and Socher, 2016, arXiv:1609.07843.

The legacy `research.metamind.io` zip URLs return a 467-byte S3 redirect document. Those files are left in place and are not read.

Parquet rows are the raw lines, joined with newlines. Expected sizes and row counts are in `data_util.py`. The test parquet is size-checked when it is downloaded. Its text is decoded only after the learning-rate selection is on disk.

Tokenizer: `tiktoken` encoding `gpt2`, vocabulary 50257, not fit on this corpus.

## Model

Decoder-only Transformer, 123,551,232 parameters: 12 pre-norm blocks, width 768, 12 heads, rotary positions, RMSNorm, causal attention, SwiGLU hidden size 2048, sequence length 256, tied token embedding and output matrix. AdamW, betas (0.9, 0.95), weight decay 0.1, gradient clip 1, bfloat16 autocast, loss in float32.

## Budget rule

This rule is the one the trainer applies. The resulting token count is written to `protocol.json` before any final step, and a restart uses that file.

- Tune: learning rates 1e-4, 3e-4, 1e-3, 3e-3, both losses, 8,000,000 tokens each, seed 0, one shared initialization.
- Selection: lowest validation value of the loss being trained. Ties break toward higher validation accuracy, then the smaller learning rate. The test split is not in this choice.
- Final: 3 seeds. Prefer one packed epoch of the training split when the eight tune runs plus six final runs fit in 48 hours at the slower loss's measured tokens per second. Otherwise the largest 5,000,000-token multiple that fits in 48 hours. If that multiple is under 20,000,000 and 20,000,000 tokens still fits in 72 hours, use 20,000,000. Below that, do not start.
- Every final run uses the same step count. A seed builds one initialization and one token permutation. Both losses load them.
- The watchdog is 1.5 times the projected program plus one hour, taken from the slower loss. It can stop a hung process. It is not a per-loss deadline. A killed run resumes from its own checkpoint to the same step count.
- A non-finite loss stops that run and keeps the row. The other loss still runs the full step count.

## CNN follow-up

The separate 30-epoch CIFAR-10 rerun and the 5-epoch MNIST rerun are not launched. CIFAR-10 is the stability study in `STABILITY_PROTOCOL.md`: cross-entropy and the full Brier score, at weight decay 0 and at 5e-4, with its own learning-rate map and the same 30-epoch patience. MNIST is not rerun. Both losses already reached validation accuracy about 0.99 in two epochs, so a longer MNIST run would not test that premise.
