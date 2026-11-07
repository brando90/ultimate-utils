"""CPU checks for the loss identity, the split, and the language-model block.

These run before any GPU job. They do not download CIFAR-10, MNIST, or WikiText.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_util import carve_validation
from losses import brier_explicit, brier_loss, collapsed_logit_check, cross_entropy, local_square
from metrics import paired_seed_comparison
from models import AlexNetSmallImage, DenseCausalLM, count_parameters
from train import TestGate


def test_brier_is_the_sum_over_classes() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 0.0, 0.0]])
    y = torch.tensor([0, 2])
    fast = brier_loss(logits, y)
    slow = brier_explicit(logits, y)
    if not torch.allclose(fast, slow, atol=1e-6):
        raise AssertionError(f"fast {fast} explicit {slow}")
    q = torch.softmax(logits[:1], dim=-1)[0]
    local = (1 - q[0]) ** 2
    full = (q[0] - 1) ** 2 + q[1] ** 2 + q[2] ** 2
    if not torch.allclose(full, brier_loss(logits[:1], y[:1]), atol=1e-6):
        raise AssertionError("full sum does not match the training loss")
    if torch.allclose(local, full):
        raise AssertionError("local square matched the full Brier score on a spread-out forecast")
    uniform = cross_entropy(torch.zeros(4, 10), torch.tensor([1, 2, 3, 4]))
    if abs(float(uniform) - torch.log(torch.tensor(10.0))) > 1e-5:
        raise AssertionError(f"uniform cross-entropy was {uniform}")


def test_collapsed_learning_signal() -> None:
    rows = collapsed_logit_check(torch.device("cpu"))
    ce = rows["cross_entropy"]["bias_gradient"]
    br = abs(rows["brier"]["bias_gradient"])
    if ce > -0.2 or br > 1e-3:
        raise AssertionError(rows)


def test_split_is_disjoint() -> None:
    train_idx, val_idx = carve_validation(1000, 50, 123)
    if set(train_idx) & set(val_idx):
        raise AssertionError("overlap")
    if len(train_idx) + len(val_idx) != 1000:
        raise AssertionError("coverage")
    again, _ = carve_validation(1000, 50, 123)
    if again != train_idx:
        raise AssertionError("split was not fixed by the seed")


def test_test_gate() -> None:
    gate = TestGate()
    try:
        gate.open("mnist")
    except RuntimeError:
        pass
    else:
        raise AssertionError("test split opened before selection")
    gate.allow("mnist")
    gate.open("mnist")


def test_alexnet_and_causal_lm() -> None:
    net = AlexNetSmallImage(3)
    y = net(torch.randn(2, 3, 32, 32))
    if y.shape != (2, 10):
        raise AssertionError(y.shape)
    mnist = AlexNetSmallImage(1)
    if mnist(torch.randn(2, 1, 28, 28)).shape != (2, 10):
        raise AssertionError("MNIST shape")
    torch.manual_seed(0)
    a = AlexNetSmallImage(3)
    torch.manual_seed(0)
    b = AlexNetSmallImage(3)
    if not torch.equal(a.features[0].weight, b.features[0].weight):
        raise AssertionError("AlexNet init was not matched by the seed")
    small = DenseCausalLM(vocab=128, d=32, n_layer=2, n_head=4, seq_len=16)
    small.tie_weights()
    if small.head.weight is not small.tok_emb.weight:
        raise AssertionError("output layer is not tied")
    idx = torch.randint(0, 128, (2, 16))
    first = small(idx)
    changed = idx.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 128
    second = small(changed)
    if not torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5):
        raise AssertionError("a future token changed an earlier logit")
    loss = brier_loss(first[:, :-1].reshape(-1, 128), idx[:, 1:].reshape(-1))
    loss.backward()
    full = DenseCausalLM(vocab=50257, d=768, n_layer=12, n_head=12, seq_len=256)
    n_params = count_parameters(full)
    if not (100_000_000 <= n_params <= 300_000_000):
        raise AssertionError(f"parameter count {n_params}")
    print(f"full language model parameters {n_params}")


def test_paired_report() -> None:
    comp = paired_seed_comparison([0.2, 0.3, 0.4], [0.1, 0.2, 0.3])
    if comp["n"] != 3 or abs(comp["mean_diff"] - 0.1) > 1e-12:
        raise AssertionError(comp)
    if comp["p_value"] is None or not (0 < comp["p_value"] <= 1):
        raise AssertionError(comp)


def main() -> None:
    test_brier_is_the_sum_over_classes()
    test_collapsed_learning_signal()
    test_split_is_disjoint()
    test_test_gate()
    test_paired_report()
    test_alexnet_and_causal_lm()
    # local_square is the diagnostic, and it is smaller than the Brier score here.
    logits = torch.tensor([[0.0, 2.0, -2.0]])
    y = torch.tensor([0])
    if float(local_square(logits, y)) >= float(brier_loss(logits, y)):
        raise AssertionError("expected the local square to drop the other classes")
    print("CPU_CHECKS_PASS")


if __name__ == "__main__":
    main()
