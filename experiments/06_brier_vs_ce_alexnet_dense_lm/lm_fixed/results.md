# WikiText-103 raw: Brier score versus cross-entropy

**Status:** DONE

**Last updated:** 2026-11-08T00:23:35-0800

The AlexNet tables are in the earlier `full/` run. This file is the language-model run only.

## Protocol

```
{
  "final_tokens_per_run": 118032360,
  "tune_tokens_per_trial": 8000000,
  "learning_rates": [
    0.0001,
    0.0003,
    0.001,
    0.003
  ],
  "seeds": 3,
  "epoch_tokens": 118032360,
  "tokens_per_sec": 16198.750876862452,
  "projected_seconds": 47669.98183193042,
  "watchdog_seconds": 75104.97274789563,
  "rule": "one packed epoch of the WikiText-103 raw training split",
  "objective_checks": {
    "brier_identity_ok": true
  },
  "batch": 8,
  "seq_len": 256,
  "tokens_per_step": 2040,
  "final_steps": 57859,
  "tune_steps": 3921,
  "final_tokens_realized": 118032360,
  "tune_tokens_realized": 7998840,
  "corpus": "Salesforce/wikitext wikitext-103-raw-v1",
  "n_train_tokens": 118496151,
  "n_valid_tokens": 248461,
  "tokenizer": "tiktoken gpt2",
  "parameters": 123551232,
  "smoke": {
    "per_loss": {
      "cross_entropy": {
        "seconds_per_step": 0.11220774551232655,
        "tokens_per_sec": 18180.563121428157,
        "grad_norm": 1.4471614360809326
      },
      "brier": {
        "seconds_per_step": 0.12593563636526084,
        "tokens_per_sec": 16198.750876862452,
        "grad_norm": 0.005125016905367374
      }
    },
    "slower_tokens_per_sec": 16198.750876862452,
    "batch": 8
  },
  "written": "2026-11-07T07:40:20-0800"
}
```

## Validation and test

### validation

| loss | seed | learning rate | steps | tokens | accuracy | cross-entropy | perplexity | Brier | ECE | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cross_entropy | 0 | 0.0003 | 57859 | 118032360 | 0.381876 | 3.92668 | 50.738 | 0.744614 | 0.0127068 | False |
| brier | 0 | 0.0003 | 57859 | 118032360 | 0.288361 | 5.6346 | 279.946 | 0.820669 | 0.00322312 | False |
| cross_entropy | 1 | 0.0003 | 57859 | 118032360 | 0.382458 | 3.91998 | 50.3995 | 0.744553 | 0.011763 | False |
| brier | 1 | 0.0003 | 57859 | 118032360 | 0.288365 | 5.6311 | 278.968 | 0.82074 | 0.00389503 | False |
| cross_entropy | 2 | 0.0003 | 57859 | 118032360 | 0.382276 | 3.92682 | 50.7452 | 0.7453 | 0.0101772 | False |
| brier | 2 | 0.0003 | 57859 | 118032360 | 0.288187 | 5.63717 | 280.667 | 0.820929 | 0.00238801 | False |

### test

| loss | seed | learning rate | steps | tokens | accuracy | cross-entropy | perplexity | Brier | ECE | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cross_entropy | 0 | 0.0003 | 57859 | 118032360 | 0.384727 | 3.94223 | 51.5336 | 0.743927 | 0.0149959 | False |
| brier | 0 | 0.0003 | 57859 | 118032360 | 0.285131 | 5.66837 | 289.563 | 0.823326 | 0.00430662 | False |
| cross_entropy | 1 | 0.0003 | 57859 | 118032360 | 0.384109 | 3.94102 | 51.4709 | 0.743708 | 0.0122009 | False |
| brier | 1 | 0.0003 | 57859 | 118032360 | 0.285636 | 5.65472 | 285.638 | 0.822608 | 0.00391187 | False |
| cross_entropy | 2 | 0.0003 | 57859 | 118032360 | 0.383159 | 3.94522 | 51.6875 | 0.744795 | 0.0115541 | False |
| brier | 2 | 0.0003 | 57859 | 118032360 | 0.285816 | 5.6608 | 287.379 | 0.823183 | 0.00332192 | False |

Paired difference is cross-entropy-trained minus Brier-trained. The interval resamples seeds. With 3 seeds the sign-flip p-value cannot resolve 0.05.

| metric | mean difference | 95% interval | p-value | n |
| --- | ---: | --- | ---: | ---: |
| accuracy | 0.098471 | [0.0973439, 0.0995958] | 0.25 | 3 |
| cross_entropy | -1.71848 | [-1.72614, -1.71371] | 0.25 | 3 |
| perplexity | -235.962 | [-238.029, -234.167] | 0.25 | 3 |
| brier | -0.0788955 | [-0.0793989, -0.0783877] | 0.25 | 3 |
| expected_calibration_error | 0.00907014 | [0.00823216, 0.0106892] | 0.25 | 3 |

Invalid runs: 0.

ECE is top-label expected calibration error: 15 equal-width bins of the maximum softmax probability, first bin closed on both ends, later bins open on the left and closed on the right, empty bins omitted from the sum. It is not classwise and not equal-count. A lower ECE is not, by itself, better accuracy, cross-entropy, or full Brier score. Bin rows, when evaluation ran, are under metrics.calibration_bins in state.json.

**TLDR-end:** [ultimate-utils: brier vs cross-entropy] Language-model status DONE. WikiText-103 raw, 123551232-parameter Transformer, matched token budget.
