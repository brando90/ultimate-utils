# CIFAR-10 stability follow-up

**Status:** DONE

**Last updated:** 2026-11-07T11:09:24-08:00

This is a follow-up. The 6-epoch test scores were already observed and were not used to choose these learning rates, the patience, or the 30-epoch cap.

A lower validation loss is compared only inside one training loss. Cross-entropy and the Brier score are different scales.

## Selection

```
{
  "adamw|cross_entropy": {
    "learning_rate": 0.0003,
    "label": "success",
    "outcome": "success"
  },
  "adamw|brier": {
    "learning_rate": 0.0003,
    "label": "success",
    "outcome": "success"
  },
  "sgd|cross_entropy": {
    "learning_rate": 0.03,
    "label": "success",
    "outcome": "success"
  },
  "sgd|brier": {
    "learning_rate": 0.1,
    "label": "success",
    "outcome": "success"
  }
}
```

## Map, seed 0

| optimizer | loss | learning rate | outcome | best epoch | best validation objective | best accuracy | seconds |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| adamw | cross_entropy | 0.0003 | success | 7 | 0.643535 | 0.7886 | 171.0 |
| adamw | cross_entropy | 0.001 | success | 7 | 0.653777 | 0.8014 | 170.7 |
| adamw | cross_entropy | 0.003 | success | 7 | 0.801013 | 0.735 | 170.7 |
| adamw | cross_entropy | 0.01 | success | 29 | 1.26644 | 0.5322 | 464.4 |
| adamw | brier | 0.0003 | success | 20 | 0.274467 | 0.829 | 375.4 |
| adamw | brier | 0.001 | success | 12 | 0.288192 | 0.8086 | 251.1 |
| adamw | brier | 0.003 | success | 28 | 0.297488 | 0.813 | 472.3 |
| adamw | brier | 0.01 | stall | 0 | 0.76397 | 0.3364 | 62.6 |
| sgd | cross_entropy | 0.01 | success | 29 | 0.667026 | 0.77 | 431.5 |
| sgd | cross_entropy | 0.03 | success | 21 | 0.513838 | 0.8326 | 360.1 |
| sgd | cross_entropy | 0.1 | success | 12 | 0.531251 | 0.8288 | 230.2 |
| sgd | cross_entropy | 0.3 | success | 4 | 1.14272 | 0.612 | 115.6 |
| sgd | brier | 0.01 | success | 29 | 0.333378 | 0.7652 | 438.4 |
| sgd | brier | 0.03 | success | 27 | 0.236486 | 0.8348 | 437.4 |
| sgd | brier | 0.1 | success | 29 | 0.211402 | 0.8562 | 437.6 |
| sgd | brier | 0.3 | success | 28 | 0.233788 | 0.8402 | 438.5 |

## Held-out test at the selected learning rates

| optimizer | loss | seed | learning rate | accuracy | cross-entropy | Brier score | ECE | outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| adamw | cross_entropy | 0 | 0.0003 | 0.795 | 0.613315 | 0.2911 | 0.0442776 | success |
| adamw | cross_entropy | 1 | 0.0003 | 0.7952 | 0.604836 | 0.289924 | 0.0382059 | success |
| adamw | cross_entropy | 2 | 0.0003 | 0.8 | 0.615524 | 0.286264 | 0.0474642 | success |
| adamw | cross_entropy | 3 | 0.0003 | 0.7878 | 0.654234 | 0.303083 | 0.0575607 | success |
| adamw | cross_entropy | 4 | 0.0003 | 0.8017 | 0.641582 | 0.290151 | 0.0718421 | success |
| adamw | brier | 0 | 0.0003 | 0.8217 | 0.767003 | 0.284299 | 0.108324 | success |
| adamw | brier | 1 | 0.0003 | 0.8166 | 0.639876 | 0.274995 | 0.0767617 | success |
| adamw | brier | 2 | 0.0003 | 0.8183 | 0.665808 | 0.278539 | 0.0821356 | success |
| adamw | brier | 3 | 0.0003 | 0.8183 | 0.711496 | 0.280168 | 0.0986452 | success |
| adamw | brier | 4 | 0.0003 | 0.8234 | 0.672611 | 0.273085 | 0.0885248 | success |
| sgd | cross_entropy | 0 | 0.03 | 0.8356 | 0.519233 | 0.241617 | 0.045314 | success |
| sgd | cross_entropy | 1 | 0.03 | 0.8304 | 0.521866 | 0.246706 | 0.0478586 | success |
| sgd | cross_entropy | 2 | 0.03 | 0.8196 | 0.542672 | 0.260279 | 0.0280132 | success |
| sgd | cross_entropy | 3 | 0.03 | 0.8242 | 0.525286 | 0.251612 | 0.033238 | success |
| sgd | cross_entropy | 4 | 0.03 | 0.8324 | 0.524683 | 0.245389 | 0.0564588 | success |
| sgd | brier | 0 | 0.1 | 0.8539 | 0.478301 | 0.215431 | 0.0520637 | success |
| sgd | brier | 1 | 0.1 | 0.8524 | 0.493367 | 0.220237 | 0.0579393 | success |
| sgd | brier | 2 | 0.1 | 0.8549 | 0.48675 | 0.217407 | 0.053161 | success |
| sgd | brier | 3 | 0.1 | 0.8236 | 0.553667 | 0.254875 | 0.0376232 | success |
| sgd | brier | 4 | 0.1 | 0.8333 | 0.520006 | 0.240029 | 0.0410654 | success |

Paired differences use the training seed. With 5 seeds the smallest nonzero sign-flip p-value is 0.0625.

| comparison | metric | mean difference | 95% interval | p-value | n |
| --- | --- | ---: | --- | ---: | ---: |
| adamw, cross-entropy minus Brier | accuracy | -0.02372 | [-0.02792, -0.02022] | 0.0625 | 5 |
| adamw, cross-entropy minus Brier | cross_entropy | -0.0654606 | [-0.110673, -0.0364843] | 0.0625 | 5 |
| adamw, cross-entropy minus Brier | brier | 0.0138873 | [0.00879633, 0.0189784] | 0.0625 | 5 |
| adamw, cross-entropy minus Brier | expected_calibration_error | -0.0390081 | [-0.0535788, -0.0259377] | 0.0625 | 5 |
| sgd, cross-entropy minus Brier | accuracy | -0.01518 | [-0.02658, -0.00378] | 0.125 | 5 |
| sgd, cross-entropy minus Brier | cross_entropy | 0.0203298 | [-0.00790685, 0.0444414] | 0.1875 | 5 |
| sgd, cross-entropy minus Brier | brier | 0.0195248 | [0.00440841, 0.0329734] | 0.125 | 5 |
| sgd, cross-entropy minus Brier | expected_calibration_error | -0.006194 | [-0.0173157, 0.00586993] | 0.4375 | 5 |
| cross_entropy, AdamW minus SGD | accuracy | -0.0325 | [-0.03784, -0.02602] | 0.0625 | 5 |
| cross_entropy, AdamW minus SGD | cross_entropy | 0.0991503 | [0.0811452, 0.117343] | 0.0625 | 5 |
| cross_entropy, AdamW minus SGD | brier | 0.0429838 | [0.0344397, 0.0494227] | 0.0625 | 5 |
| cross_entropy, AdamW minus SGD | expected_calibration_error | 0.00969359 | [-0.00210869, 0.0205862] | 0.25 | 5 |
| brier, AdamW minus SGD | accuracy | -0.02396 | [-0.0354, -0.01232] | 0.0625 | 5 |
| brier, AdamW minus SGD | cross_entropy | 0.184941 | [0.151212, 0.238335] | 0.0625 | 5 |
| brier, AdamW minus SGD | brier | 0.0486212 | [0.0340133, 0.0629514] | 0.0625 | 5 |
| brier, AdamW minus SGD | expected_calibration_error | 0.0425077 | [0.0283403, 0.0564047] | 0.0625 | 5 |

**TLDR-end:** [ultimate-utils: brier vs cross-entropy] CIFAR-10 stability follow-up. Status DONE. AdamW and SGD, both losses, fixed weight decay 0.0005, normal initialization.
