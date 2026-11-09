"""Post-hoc paired recall analysis of saved counts; no training or evaluation.

Run with Python's standard library. Each bootstrap sample contains four paired
training seeds, sampled with replacement. Enumerate all 4**4 ordered samples;
take the inverse empirical cumulative distribution at 0.025 and 0.975. The
two-sided sign-flip test enumerates all 2**4 signs, under the null that paired
differences are symmetric about zero. Small-seed intervals are descriptive and
do not cover uncertainty across datasets or a repeated hyperparameter search.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from itertools import product
from math import ceil
from pathlib import Path


def read_counts(source: Path) -> dict:
    groups = {name: {loss: {} for loss in ("log_loss", "brier")}
              for name in ("sgd", "adam")}
    published = {"sgd": Fraction("0.05"), "adam": Fraction("0.001")}
    for line in source.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 11 or cells[0] not in ("1", "2", "3", "4"):
            continue
        seed, loss, optimizer, rate, bias = cells[:5]
        if optimizer not in published or loss not in groups[optimizer]:
            continue
        if Fraction(rate) != published[optimizer] or Fraction(bias) != -12:
            continue
        hits, count = map(int, cells[6].split("/"))
        if count != 980 or not 0 <= hits <= count:
            raise ValueError(f"Unexpected class-0 counts: {cells[6]}")
        seed = int(seed)
        if seed in groups[optimizer][loss]:
            raise ValueError(f"Duplicate row: {optimizer}, {loss}, seed {seed}")
        groups[optimizer][loss][seed] = (hits, count)
    for optimizer, by_loss in groups.items():
        for loss, rows in by_loss.items():
            if set(rows) != {1, 2, 3, 4}:
                raise ValueError(f"Expected seeds 1-4: {optimizer}, {loss}, {rows}")
    return groups


def compare(by_loss: dict) -> dict:
    seeds = [1, 2, 3, 4]
    left = [Fraction(*by_loss["log_loss"][seed]) for seed in seeds]
    right = [Fraction(*by_loss["brier"][seed]) for seed in seeds]
    differences = [a - b for a, b in zip(left, right)]
    n = len(seeds)
    observed = sum(differences) / n
    bootstrap = sorted(
        sum(differences[index] for index in indices) / n
        for indices in product(range(n), repeat=n)
    )
    interval = [bootstrap[ceil(q * len(bootstrap)) - 1]
                for q in (Fraction(25, 1000), Fraction(975, 1000))]
    permutations = [
        sum(sign * value for sign, value in zip(signs, differences)) / n
        for signs in product((-1, 1), repeat=n)
    ]
    extreme = sum(abs(value) >= abs(observed) for value in permutations)
    return {
        "seeds": seeds,
        "n_seeds": n,
        "cross_entropy_hits": [by_loss["log_loss"][s][0] for s in seeds],
        "brier_hits": [by_loss["brier"][s][0] for s in seeds],
        "class0_count_per_seed": 980,
        "cross_entropy_mean_recall": float(sum(left) / n),
        "brier_mean_recall": float(sum(right) / n),
        "paired_differences": [float(value) for value in differences],
        "mean_difference": float(observed),
        "mean_difference_fraction": str(observed),
        "ci95": [float(value) for value in interval],
        "ci95_fractions": [str(value) for value in interval],
        "bootstrap_samples": len(bootstrap),
        "sign_flip_extreme": extreme,
        "sign_flip_total": len(permutations),
        "p_value": extreme / len(permutations),
    }


def main() -> None:
    source = Path(__file__).resolve().with_name("results.md")
    groups = read_counts(source)
    output = {
        "analysis": "post-hoc analysis of saved counts; no new runs",
        "source": "collapsed_replication/results.md",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "difference": "cross-entropy-trained minus Brier-trained class-0 recall",
        "resampling_unit": "paired training seed",
        "interval": "exact 4^4 ordered seed bootstrap; inverse empirical CDF 0.025/0.975",
        "test": "exact two-sided sign-flip; null: paired differences symmetric about zero",
        "multiplicity": "unadjusted p-values for two reported optimizer comparisons",
        "sgd_lr_0.05_bias_minus12": compare(groups["sgd"]),
        "adam_lr_0.001_bias_minus12": compare(groups["adam"]),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
