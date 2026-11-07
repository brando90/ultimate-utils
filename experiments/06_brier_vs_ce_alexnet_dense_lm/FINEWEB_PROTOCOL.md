# FineWeb-Edu replication

Status 11-07-2026: code is `train_fineweb.py`. The run is placed on mercury1. It is not placed on mercury2, whose `/lfs` is nearly full. The final token count is the WikiText realized count, 118,032,360. A run that stays finite for that count is not labeled a stall. Non-finite loss or gradient stops that run only.

This run waits until the WikiText-103 language-model comparison and the CIFAR-10 stability study are not asking for the same GPU. It does not replace either one. A second corpus is a check of external validity. It is not a measure of dataset quality, and it is not the weight-decay study.

## Source

Hugging Face dataset `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, license ODC-By. The card lists `sample-10BT`, `sample-100BT`, and `sample-350BT`. Each of those configs has a train split only. The card was read on 11-07-2026. The default feature columns include `text` and `id`.

`sample-10BT` is 14 parquet shards. The replication needs the same training-token count as the WikiText protocol, 118,032,360 tokens, plus a validation split and a held-out split of 2,000,000 tokens each. That is early training for a 123,551,232-parameter model. Shards are read in sorted filename order and only until those counts are filled. The other shards are not downloaded.

## Split

There is no official validation or test split. Documents are assigned by `sha256(id)` interpreted as an integer, modulo 1000:

- 0 through 7: validation
- 8 through 15: held-out test
- the rest: train

Duplicate `id` values are dropped. Reading stops when the train pack reaches 118,032,360 tokens and both held-out splits have 2,000,000 tokens. The tokenizer is tiktoken `gpt2`, vocabulary 50257, the same encoder as WikiText. The dataset's own `token_count` field is not the training count. The list of accepted document ids is hashed and stored before training.

## Training

Same decoder-only Transformer, same seeds, same learning-rate grid, same AdamW settings, and the same weight decay 0.1 as the WikiText comparison. This is cross-entropy versus the full Brier score at that one weight decay. It is not the CIFAR-10 factorial. The test documents are encoded only after the learning-rate selection is on disk. The watchdog is computed from the measured tokens per second, as in the WikiText runner.

Mayee, Zhang, Roth, and Hashimoto, arXiv:2306.13840, measure variability across datasets. FineWeb-Edu is not one of the datasets in that paper. This replication does not extend that measurement.
