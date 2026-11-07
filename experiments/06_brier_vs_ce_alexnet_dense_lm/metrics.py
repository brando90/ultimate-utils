"""Evaluation metrics shared by the vision runs and the language-model runs.

Accuracy is the fraction of argmax predictions that match the label.
Cross-entropy and the Brier score use the definitions in losses.py.
Perplexity is exp(cross-entropy) on the same tokens. Expected calibration
error uses 15 equal-width bins of the maximum predicted probability.
"""

from __future__ import annotations

import math

import torch

from losses import brier_per_example, local_square, probabilities


def expected_calibration_error(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=-1)
    correct = pred.eq(y).float()
    total = conf.numel()
    if total == 0:
        return float("nan")
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = probs.new_zeros(())
    for i in range(n_bins):
        if i == 0:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf > edges[i]) & (conf <= edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        gap = (correct[mask].mean() - conf[mask].mean()).abs()
        ece = ece + gap * (count / total)
    return float(ece.detach().cpu())


def classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict:
    logits = logits.float()
    q = probabilities(logits)
    q_y = q.gather(-1, y[:, None]).squeeze(-1)
    ce = float((-q_y.clamp_min(1e-12).log()).mean().detach().cpu())
    brier = float(brier_per_example(logits, y).mean().detach().cpu())
    local = float(local_square(logits, y).detach().cpu())
    pred = logits.argmax(dim=-1)
    acc = float(pred.eq(y).float().mean().detach().cpu())
    return {
        "accuracy": acc,
        "cross_entropy": ce,
        "perplexity": float(math.exp(ce)) if ce < 50 else float("inf"),
        "brier": brier,
        "local_square": local,
        "brier_minus_local_square": brier - local,
        "expected_calibration_error": expected_calibration_error(q, y),
        "mean_true_class_probability": float(q_y.mean().detach().cpu()),
        "mean_true_class_logit": float(logits.gather(-1, y[:, None]).mean().detach().cpu()),
        "mean_abs_logit": float(logits.abs().mean().detach().cpu()),
        "n": int(y.numel()),
    }


def paired_seed_comparison(a: list[float], b: list[float], seed: int = 0, draws: int = 10000) -> dict:
    """Paired difference a - b across training seeds.

    The interval resamples seeds with replacement. The p-value is an exact
    two-sided sign-flip test. With few seeds the p-value sits on a coarse grid.
    """
    if len(a) != len(b) or len(a) == 0:
        return {"n": len(a), "mean_diff": None, "ci95": None, "p_value": None}
    d = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    n = int(d.numel())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (draws, n), generator=g)
    means = d[idx].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    obs = d.mean().abs()
    count = 0
    total = 1 << n
    for mask in range(total):
        signs = torch.tensor(
            [1.0 if (mask >> i) & 1 else -1.0 for i in range(n)], dtype=torch.float64
        )
        if (d * signs).mean().abs() + 1e-15 >= obs:
            count += 1
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "p_value": count / total,
        "sign_flip_count": count,
        "sign_flip_total": total,
        "resampling_unit": "training seed",
        "interval": "percentile bootstrap of the paired mean difference, 10000 draws, seed 0",
        "test": "two-sided sign-flip permutation, null that the paired difference is symmetric about 0",
    }
