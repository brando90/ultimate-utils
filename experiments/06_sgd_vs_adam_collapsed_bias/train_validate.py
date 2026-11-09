"""Test whether SGD leaves a collapsed Brier coordinate frozen.

Predictions were written in README.md before this run. Seed 0.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "summary.json"


def brier_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    q = logits.softmax(dim=-1)
    q_y = q.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (q.square().sum(dim=-1) + 1.0 - 2.0 * q_y).mean()


def log_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


LOSSES = {"log_loss": log_loss, "brier": brier_loss}


def make_opt(params, name: str, lr: float, momentum: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(name)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).flatten(1))


def pure_logits(device: torch.device) -> list[dict]:
    torch.manual_seed(0)
    n, k = 1000, 10
    y = torch.arange(n, device=device) % k
    rows = []
    specs = [
        ("sgd", 0.1, 0.0),
        ("sgd", 0.1, 0.9),
        ("adam", 1e-2, 0.0),
        ("adam", 1e-3, 0.0),
    ]
    for loss_name, fn in LOSSES.items():
        for opt_name, lr, momentum in specs:
            theta = torch.zeros(k, device=device, requires_grad=True)
            with torch.no_grad():
                theta.zero_()
                theta[0] = -12.0
            opt = make_opt([theta], opt_name, lr, momentum)
            grad_logits = theta.detach().clone().requires_grad_(True)
            g = fn(grad_logits[None].expand(n, -1), y)
            g.backward()
            grad0 = float(grad_logits.grad[0])
            for _ in range(2000):
                opt.zero_grad(set_to_none=True)
                loss = fn(theta[None].expand(n, -1), y)
                loss.backward()
                opt.step()
            rows.append(
                {
                    "level": "logits",
                    "loss": loss_name,
                    "optimizer": opt_name,
                    "lr": lr,
                    "momentum": momentum,
                    "steps": 2000,
                    "grad0": grad0,
                    "logit0": float(theta.detach()[0]),
                    "delta": float(theta.detach()[0] + 12.0),
                }
            )
            print(rows[-1], flush=True)
    return rows


def load_mnist(device: torch.device):
    from torchvision import datasets, transforms

    root = HERE / "results" / "data"
    tfm = transforms.ToTensor()
    train_set = datasets.MNIST(str(root), train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(str(root), train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    return train_loader, test_loader


@torch.no_grad()
def head_metrics(head, features, labels, device) -> dict:
    head.eval()
    logits = head(features.to(device))
    pred = logits.argmax(-1)
    y = labels.to(device)
    mask = y == 0
    return {
        "accuracy": float((pred == y).float().mean()),
        "collapsed_recall": float((pred[mask] == 0).float().mean()) if int(mask.sum()) else 0.0,
        "mean_logit0": float(logits[:, 0].mean()),
        "bias0": float(head.bias.detach()[0]),
    }


def frozen_head(device: torch.device) -> list[dict]:
    torch.manual_seed(0)
    train_loader, test_loader = load_mnist(device)
    encoder = SmallCNN().to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    def embed(loader):
        xs, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                xs.append(encoder.features(x.to(device)).flatten(1).cpu())
                ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    train_h, train_y = embed(train_loader)
    test_h, test_y = embed(test_loader)
    rows = []
    specs = [
        ("sgd", 0.1, 0.0, 8),
        ("sgd", 0.1, 0.9, 8),
        ("adam", 1e-3, 0.0, 8),
        ("adam", 1e-2, 0.0, 8),
    ]
    for loss_name in ("log_loss", "brier"):
        for opt_name, lr, momentum, epochs in specs:
            torch.manual_seed(0)
            head = nn.Linear(train_h.shape[1], 10).to(device)
            with torch.no_grad():
                head.bias.zero_()
                head.bias[0] = -12.0
            opt = make_opt(head.parameters(), opt_name, lr, momentum)
            fn = LOSSES[loss_name]
            sample = train_h[:256].to(device)
            sample_y = train_y[:256].to(device)
            opt.zero_grad(set_to_none=True)
            fn(head(sample), sample_y).backward()
            grad0 = float(head.bias.grad[0])
            head.zero_grad(set_to_none=True)
            t0 = time.time()
            for _ in range(epochs):
                head.train()
                perm = torch.randperm(train_h.shape[0])
                for i in range(0, perm.numel() - 128, 128):
                    batch = perm[i : i + 128]
                    opt.zero_grad(set_to_none=True)
                    loss = fn(head(train_h[batch].to(device)), train_y[batch].to(device))
                    loss.backward()
                    opt.step()
            metrics = head_metrics(head, test_h, test_y, device)
            row = {
                "level": "frozen_head",
                "loss": loss_name,
                "optimizer": opt_name,
                "lr": lr,
                "momentum": momentum,
                "epochs": epochs,
                "grad0": grad0,
                "seconds": time.time() - t0,
                **metrics,
                "delta_bias": metrics["bias0"] + 12.0,
            }
            rows.append(row)
            print(row, flush=True)
    return rows


@torch.no_grad()
def cnn_metrics(model, loader, device) -> dict:
    model.eval()
    correct = total = hit = collapsed = 0
    logit_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(-1)
        total += y.numel()
        correct += int((pred == y).sum())
        mask = y == 0
        collapsed += int(mask.sum())
        hit += int((pred[mask] == 0).sum())
        logit_sum += float(logits[:, 0].sum())
    return {
        "accuracy": correct / total,
        "collapsed_recall": hit / max(collapsed, 1),
        "mean_logit0": logit_sum / total,
        "bias0": float(model.fc.bias.detach()[0]),
    }


def full_cnn(device: torch.device) -> list[dict]:
    train_loader, test_loader = load_mnist(device)
    rows = []
    specs = [
        ("sgd", 0.05, 0.9, 0.0),
        ("sgd", 0.05, 0.9, -12.0),
        ("adam", 1e-3, 0.0, -12.0),
    ]
    for loss_name in ("log_loss", "brier"):
        for opt_name, lr, momentum, bias in specs:
            torch.manual_seed(0)
            model = SmallCNN().to(device)
            with torch.no_grad():
                model.fc.bias.zero_()
                model.fc.bias[0] = bias
            opt = make_opt(model.parameters(), opt_name, lr, momentum)
            fn = LOSSES[loss_name]
            x0, y0 = next(iter(train_loader))
            opt.zero_grad(set_to_none=True)
            fn(model(x0.to(device)), y0.to(device)).backward()
            grad0 = float(model.fc.bias.grad[0])
            model.zero_grad(set_to_none=True)
            t0 = time.time()
            for _ in range(3):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad(set_to_none=True)
                    fn(model(x), y).backward()
                    opt.step()
            metrics = cnn_metrics(model, test_loader, device)
            row = {
                "level": "full_cnn",
                "loss": loss_name,
                "optimizer": opt_name,
                "lr": lr,
                "momentum": momentum,
                "collapsed_bias": bias,
                "epochs": 3,
                "grad0": grad0,
                "seconds": time.time() - t0,
                **metrics,
                "delta_bias": metrics["bias0"] - bias,
            }
            rows.append(row)
            print(row, flush=True)
    return rows


def write_results(rows: list[dict], done: bool) -> None:
    status = "DONE" if done else "RUNNING"
    lines = [
        "# SGD versus Adam on a collapsed softmax coordinate",
        "",
        f"**Status:** {status} (11-06-2026)",
        "",
        "**Last updated:** 11-06-2026",
        "",
        "**Doc link:** <https://github.com/brando90/ultimate-utils/blob/main/experiments/06_sgd_vs_adam_collapsed_bias/results.md>",
        "",
        "The address resolves after a push to `main`. Predictions are in `README.md`. Seed 0.",
        "",
        "| Level | Loss | Optimizer | lr | Momentum | Bias | Grad at start | Final collapsed coordinate | Recall or delta |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        if row["level"] == "logits":
            final = f"logit0 {row['logit0']:.4f}"
            extra = f"delta {row['delta']:.4f}"
            bias = "-12"
        elif row["level"] == "frozen_head":
            final = f"bias {row['bias0']:.4f}, mean logit {row['mean_logit0']:.3f}"
            extra = f"recall {row['collapsed_recall']:.4f}, delta bias {row['delta_bias']:.4f}"
            bias = "-12"
        else:
            final = f"bias {row['bias0']:.4f}, mean logit {row['mean_logit0']:.3f}, acc {row['accuracy']:.4f}"
            extra = f"recall {row['collapsed_recall']:.4f}, delta bias {row['delta_bias']:.4f}"
            bias = str(row["collapsed_bias"])
        lines.append(
            f"| {row['level']} | {row['loss']} | {row['optimizer']} | {row['lr']} | {row['momentum']} | {bias} | {row['grad0']:.6g} | {final} | {extra} |"
        )
    lines += [
        "",
        "**TLDR-end:** [ultimate-utils: sgd vs adam collapsed bias] "
        + (
            "The table is the whole run. Compare each delta with the predictions in README.md."
            if done
            else "The table lists finished rows only."
        ),
        "",
    ]
    (HERE / "results.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["logits", "head", "cnn", "all"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    rows: list[dict] = []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if args.suite in ("logits", "all"):
        rows.extend(pure_logits(device))
        OUT.write_text(json.dumps(rows, indent=2) + "\n")
        write_results(rows, done=False)
    if args.suite in ("head", "all"):
        rows.extend(frozen_head(device))
        OUT.write_text(json.dumps(rows, indent=2) + "\n")
        write_results(rows, done=False)
    if args.suite in ("cnn", "all"):
        rows.extend(full_cnn(device))
        OUT.write_text(json.dumps(rows, indent=2) + "\n")
    write_results(rows, done=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
