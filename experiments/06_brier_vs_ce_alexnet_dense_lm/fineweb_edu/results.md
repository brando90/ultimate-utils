# FineWeb-Edu: Brier score versus cross-entropy

**Status:** DONE

**Last updated:** 2026-11-08T05:18:35-0800

Matched to the WikiText-103 run: 118,032,360 final tokens, AdamW, weight decay 0.1, normal initialization.
The held-out split is scored after learning-rate selection.

## Validation and test

### validation

| loss | seed | learning rate | steps | tokens | accuracy | cross-entropy | perplexity | Brier | ECE | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cross_entropy | 0 | 0.0003 | 57859 | 118032360 | 0.308916 | 4.60001 | 99.4851 | 0.820864 | 0.0105316 | False |
| brier | 0 | 0.0003 | 57859 | 118032360 | 0.239496 | 6.09633 | 444.226 | 0.871379 | 0.00305678 | False |
| cross_entropy | 1 | 0.0003 | 57859 | 118032360 | 0.308149 | 4.60456 | 99.9391 | 0.821155 | 0.0100398 | False |
| brier | 1 | 0.0003 | 57859 | 118032360 | 0.238791 | 6.10333 | 447.344 | 0.872158 | 0.00316593 | False |
| cross_entropy | 2 | 0.0003 | 57859 | 118032360 | 0.307926 | 4.60836 | 100.319 | 0.821301 | 0.0094959 | False |
| brier | 2 | 0.0003 | 57859 | 118032360 | 0.238576 | 6.12143 | 455.518 | 0.872303 | 0.00279289 | False |

### test

| loss | seed | learning rate | steps | tokens | accuracy | cross-entropy | perplexity | Brier | ECE | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cross_entropy | 0 | 0.0003 | 57859 | 118032360 | 0.308769 | 4.57162 | 96.7008 | 0.8209 | 0.0102483 | False |
| brier | 0 | 0.0003 | 57859 | 118032360 | 0.23932 | 6.06039 | 428.544 | 0.87176 | 0.00279039 | False |
| cross_entropy | 1 | 0.0003 | 57859 | 118032360 | 0.308442 | 4.5741 | 96.9408 | 0.821102 | 0.0099113 | False |
| brier | 1 | 0.0003 | 57859 | 118032360 | 0.238484 | 6.0689 | 432.206 | 0.872559 | 0.0032383 | False |
| cross_entropy | 2 | 0.0003 | 57859 | 118032360 | 0.307921 | 4.5795 | 97.4653 | 0.821145 | 0.00872139 | False |
| brier | 2 | 0.0003 | 57859 | 118032360 | 0.238759 | 6.0826 | 438.165 | 0.872524 | 0.00352749 | False |

Invalid runs: 0.
