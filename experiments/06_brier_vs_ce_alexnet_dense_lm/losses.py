"""Training losses for the Brier score and cross-entropy comparison.

Cross-entropy is the batch mean of -log q_y, natural log, no label smoothing,
and it is not divided by the class count. q is the softmax of the logits.

The Brier score is the batch mean of the sum over every class,
sum_k (q_k - 1{k = y})^2. That is the full multiclass Brier score.
The fast identity is ||q||^2 + 1 - 2 q_y. It is not divided by the class count.
The local square (1 - q_y)^2 is not the training loss. It is reported only
as a diagnostic, because it drops the other classes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().softmax(dim=-1)


def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), y)


def brier_per_example(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    q = probabilities(logits)
    q_y = q.gather(-1, y[:, None]).squeeze(-1)
    return q.square().sum(dim=-1) + 1.0 - 2.0 * q_y


def brier_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return brier_per_example(logits, y).mean()


def brier_explicit(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference sum over classes, used to check the fast identity."""
    q = probabilities(logits)
    one_hot = torch.zeros_like(q)
    one_hot.scatter_(-1, y[:, None], 1.0)
    return (q - one_hot).square().sum(dim=-1).mean()


def local_square(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Diagnostic only: (1 - q_y)^2. This is not a training loss."""
    q_y = probabilities(logits).gather(-1, y[:, None]).squeeze(-1)
    return (1.0 - q_y).square().mean()


LOSSES = {
    "cross_entropy": cross_entropy,
    "brier": brier_loss,
}


def collapsed_logit_check(device: torch.device, seed: int = 0) -> dict:
    """A linear classifier whose class-0 bias starts at -12.

    Half of the labels are class 0. The reported value is the gradient of the
    mean loss with respect to that shared bias. Cross-entropy stays near -0.5.
    The Brier score's bias gradient is tiny because the softmax Jacobian
    multiplies by q.
    """
    torch.manual_seed(seed)
    n, k, d = 200, 10, 16
    features = torch.randn(n, d, device=device)
    layer = torch.nn.Linear(d, k).to(device)
    with torch.no_grad():
        layer.bias.zero_()
        layer.bias[0] = -12.0
        layer.weight.normal_(0.0, 0.01)
    y = torch.zeros(n, dtype=torch.long, device=device)
    y[: n // 2] = 0
    y[n // 2 :] = 1
    rows = {}
    for name, fn in LOSSES.items():
        layer.zero_grad(set_to_none=True)
        logits = layer(features)
        loss = fn(logits, y)
        loss.backward()
        rows[name] = {
            "loss": float(loss.detach().cpu()),
            "bias_gradient": float(layer.bias.grad[0].cpu()),
            "q0_mean": float(probabilities(logits.detach())[:, 0].mean().cpu()),
        }
    ce = rows["cross_entropy"]["bias_gradient"]
    br = abs(rows["brier"]["bias_gradient"])
    if ce > -0.2:
        raise AssertionError(f"cross-entropy bias gradient was {ce}, expected < -0.2")
    if br > 1e-3:
        raise AssertionError(f"Brier bias gradient was {br}, expected < 1e-3")
    return rows
