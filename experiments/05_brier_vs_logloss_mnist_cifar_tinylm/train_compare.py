"""Brier score versus log loss on MNIST, CIFAR-10, and a byte transformer.

The local check does not train. It compares the gradient of the two losses
on a bias that sits at -12. Training runs are the cluster job.

Predictions, written before those runs, are in README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"


def brier_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean ||softmax(logits) - e_y||^2 = ||q||^2 + 1 - 2 q(y)."""
    q = logits.softmax(dim=-1)
    q_y = q.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (q.square().sum(dim=-1) + 1.0 - 2.0 * q_y).mean()


def log_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


LOSSES = {"log_loss": log_loss, "brier": brier_loss}


def check_collapsed_gradient() -> dict:
    """One batch of a 10-way softmax. Class 0 has bias -12 and is the label half the time."""
    torch.manual_seed(0)
    n, k = 200, 10
    features = torch.randn(n, 16)
    layer = nn.Linear(16, k)
    with torch.no_grad():
        layer.bias.zero_()
        layer.bias[0] = -12.0
        layer.weight.normal_(0, 0.01)
    y = torch.zeros(n, dtype=torch.long)
    y[: n // 2] = 0
    y[n // 2 :] = 1
    rows = {}
    for name, fn in LOSSES.items():
        layer.zero_grad(set_to_none=True)
        loss = fn(layer(features), y)
        loss.backward()
        rows[name] = {
            "loss": float(loss.detach()),
            "bias0_gradient": float(layer.bias.grad[0]),
            "bias1_gradient": float(layer.bias.grad[1]),
            "q0_mean": float(layer(features).softmax(-1)[:, 0].mean()),
        }
    brier_g = abs(rows["brier"]["bias0_gradient"])
    log_g = rows["log_loss"]["bias0_gradient"]
    if brier_g > 1e-4:
        raise AssertionError(f"Brier gradient at bias -12 was {brier_g}, expected < 1e-4")
    if log_g > -0.2:
        raise AssertionError(f"log-loss gradient at bias -12 was {log_g}, expected < -0.2")
    return rows


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).flatten(1))


class ByteTransformer(nn.Module):
    def __init__(self, n_layer: int = 4, d_model: int = 128, n_head: int = 4, vocab: int = 256, seq: int = 128):
        super().__init__()
        self.seq = seq
        self.token = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=n_layer)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.token(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        h = self.blocks(h, mask=mask)
        return self.head(self.norm(h))


def param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith("bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def set_collapsed_bias(module: nn.Linear, index: int, value: float) -> None:
    with torch.no_grad():
        module.bias.zero_()
        module.bias[index] = value


@torch.no_grad()
def vision_metrics(model: nn.Module, loader, device: torch.device, collapsed: int) -> dict:
    model.eval()
    total = 0
    correct = 0
    collapsed_total = 0
    collapsed_correct = 0
    nll_sum = 0.0
    brier_sum = 0.0
    logit_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        total += y.numel()
        correct += int((pred == y).sum())
        mask = y == collapsed
        collapsed_total += int(mask.sum())
        collapsed_correct += int((pred[mask] == y[mask]).sum())
        nll_sum += float(F.cross_entropy(logits, y, reduction="sum"))
        q = logits.softmax(-1)
        q_y = q.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        brier_sum += float((q.square().sum(-1) + 1 - 2 * q_y).sum())
        logit_sum += float(logits[:, collapsed].sum())
    return {
        "accuracy": correct / total,
        "collapsed_recall": collapsed_correct / max(collapsed_total, 1),
        "expected_log_loss": nll_sum / total,
        "brier": brier_sum / total,
        "mean_collapsed_logit": logit_sum / total,
    }


def run_vision(dataset: str, epochs: int, device: torch.device, seed: int, loss_name: str, weight_decay: float, collapsed_bias: float) -> dict:
    from torchvision import datasets, transforms

    torch.manual_seed(seed)
    if dataset == "mnist":
        tfm = transforms.ToTensor()
        train_set = datasets.MNIST(str(OUT_DIR / "data"), train=True, download=True, transform=tfm)
        test_set = datasets.MNIST(str(OUT_DIR / "data"), train=False, download=True, transform=tfm)
        in_ch = 1
    elif dataset == "cifar10":
        tfm = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        )
        test_tfm = transforms.ToTensor()
        train_set = datasets.CIFAR10(str(OUT_DIR / "data"), train=True, download=True, transform=tfm)
        test_set = datasets.CIFAR10(str(OUT_DIR / "data"), train=False, download=True, transform=test_tfm)
        in_ch = 3
    else:
        raise ValueError(dataset)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    model = SmallCNN(in_ch).to(device)
    if collapsed_bias != 0.0:
        set_collapsed_bias(model.fc, 0, collapsed_bias)
    opt = torch.optim.AdamW(param_groups(model, weight_decay), lr=1e-3)
    fn = LOSSES[loss_name]
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = fn(model(x), y)
            loss.backward()
            opt.step()
        metrics = vision_metrics(model, test_loader, device, collapsed=0)
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        print(dataset, loss_name, "wd", weight_decay, "bias", collapsed_bias, metrics, flush=True)
    final = history[-1]
    return {
        "dataset": dataset,
        "loss": loss_name,
        "weight_decay": weight_decay,
        "collapsed_bias": collapsed_bias,
        "seed": seed,
        "seconds": time.time() - t0,
        "final": final,
        "history": history,
    }


def load_bytes(path: Path) -> torch.Tensor:
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path,
        )
    data = path.read_bytes()
    return torch.tensor(list(data), dtype=torch.long)


def run_lm(device: torch.device, seed: int, loss_name: str, weight_decay: float, collapsed_bias: float, steps: int) -> dict:
    torch.manual_seed(seed)
    data = load_bytes(OUT_DIR / "data" / "tinyshakespeare.txt")
    n = int(0.9 * data.numel())
    train, val = data[:n], data[n:]
    model = ByteTransformer().to(device)
    suppressed = 32  # space
    if collapsed_bias != 0.0:
        set_collapsed_bias(model.head, suppressed, collapsed_bias)
    opt = torch.optim.AdamW(param_groups(model, weight_decay), lr=3e-4)
    fn = LOSSES[loss_name]
    seq = model.seq
    t0 = time.time()

    def batch(src: torch.Tensor, n_batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(0, src.numel() - seq - 1, (n_batch,))
        x = torch.stack([src[i : i + seq] for i in ix]).to(device)
        y = torch.stack([src[i + 1 : i + 1 + seq] for i in ix]).to(device)
        return x, y

    history = []
    model.train()
    for step in range(1, steps + 1):
        x, y = batch(train, 32)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = fn(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        opt.step()
        if step == 1 or step % 250 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(val, 16)
                vlogits = model(vx).reshape(-1, 256)
                vy = vy.reshape(-1)
                nll = float(F.cross_entropy(vlogits, vy))
                q = vlogits.softmax(-1)
                q_y = q.gather(-1, vy.unsqueeze(-1)).squeeze(-1)
                brier = float((q.square().sum(-1) + 1 - 2 * q_y).mean())
                mask = vy == suppressed
                space_prob = float(q[mask, suppressed].mean()) if int(mask.sum()) else math.nan
                space_logit = float(vlogits[:, suppressed].mean())
            row = {
                "step": step,
                "expected_log_loss": nll,
                "brier": brier,
                "space_probability_when_target": space_prob,
                "mean_space_logit": space_logit,
            }
            history.append(row)
            print("lm", loss_name, "wd", weight_decay, "bias", collapsed_bias, row, flush=True)
            model.train()
    return {
        "dataset": "tinyshakespeare_bytes",
        "loss": loss_name,
        "weight_decay": weight_decay,
        "collapsed_bias": collapsed_bias,
        "suppressed_byte": suppressed,
        "seed": seed,
        "seconds": time.time() - t0,
        "final": history[-1],
        "history": history,
    }


def conditions(dataset: str) -> list[dict]:
    rows = []
    for loss_name in ("log_loss", "brier"):
        for weight_decay in (0.0, 1e-4):
            for bias in (0.0, -12.0):
                rows.append({"loss": loss_name, "weight_decay": weight_decay, "collapsed_bias": bias})
    if dataset == "lm":
        # Both losses at bias 0 with no decay, and both losses at bias -12 with and without decay.
        return [
            row
            for row in rows
            if row["collapsed_bias"] == -12.0 or (row["weight_decay"] == 0.0 and row["collapsed_bias"] == 0.0)
        ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["check", "mnist", "cifar10", "lm", "all"], default="check")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"runs": []}

    if args.suite in ("check", "all"):
        summary["gradient_check"] = check_collapsed_gradient()
        print("gradient_check", summary["gradient_check"], flush=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        write_results(summary, done=False)
        if args.suite == "check":
            return

    device = torch.device(args.device)
    jobs = []
    if args.suite in ("mnist", "all"):
        jobs.append(("mnist", 2))
    if args.suite in ("cifar10", "all"):
        jobs.append(("cifar10", 4))
    for dataset, epochs in jobs:
        for spec in conditions(dataset):
            summary["runs"].append(
                run_vision(
                    dataset,
                    epochs,
                    device,
                    args.seed,
                    loss_name=spec["loss"],
                    weight_decay=spec["weight_decay"],
                    collapsed_bias=spec["collapsed_bias"],
                )
            )
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            write_results(summary, done=False)
    if args.suite in ("lm", "all"):
        for spec in conditions("lm"):
            summary["runs"].append(
                run_lm(
                    device,
                    args.seed,
                    loss_name=spec["loss"],
                    weight_decay=spec["weight_decay"],
                    collapsed_bias=spec["collapsed_bias"],
                    steps=1500,
                )
            )
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            write_results(summary, done=False)
    print("wrote", summary_path, flush=True)
    write_results(summary, done=True)


def write_results(summary: dict, done: bool) -> None:
    status = "DONE" if done else "RUNNING"
    lines = [
        "# Brier score versus log loss, CNN and tiny language model",
        "",
        f"**Status:** {status} (11-06-2026)",
        "",
        "**Last updated:** 11-06-2026",
        "",
        "**Doc link:** <https://github.com/brando90/ultimate-utils/blob/main/experiments/05_brier_vs_logloss_mnist_cifar_tinylm/results.md>",
        "",
        "The address resolves after a push to `main`. Predictions are in `README.md`.",
        "",
    ]
    check = summary.get("gradient_check")
    if check:
        lines.append("## Gradient check")
        lines.append("")
        lines.append("| Loss | Bias-0 gradient | Mean q(0) |")
        lines.append("| --- | --- | --- |")
        for name, row in check.items():
            lines.append(f"| {name} | {row['bias0_gradient']:.6g} | {row['q0_mean']:.6g} |")
        lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| Data | Loss | Weight decay | Collapsed bias | Final metric | Seconds |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for run in summary.get("runs", []):
        final = run["final"]
        if "accuracy" in final:
            metric = (
                f"accuracy {final['accuracy']:.4f}, collapsed recall {final['collapsed_recall']:.4f}, "
                f"log loss {final['expected_log_loss']:.4f}, Brier {final['brier']:.4f}, "
                f"mean collapsed logit {final['mean_collapsed_logit']:.3f}"
            )
        else:
            metric = (
                f"log loss {final['expected_log_loss']:.4f}, Brier {final['brier']:.4f}, "
                f"space probability {final['space_probability_when_target']:.4f}, "
                f"mean space logit {final['mean_space_logit']:.3f}"
            )
        lines.append(
            f"| {run['dataset']} | {run['loss']} | {run['weight_decay']} | {run['collapsed_bias']} | {metric} | {run['seconds']:.0f} |"
        )
    lines.append("")
    lines.append(
        "**TLDR-end:** [ultimate-utils: brier vs log loss nets] "
        + (
            "The table is complete. Compare collapsed-label recall and the suppressed logit with the predictions in README.md."
            if done
            else "The table lists finished runs only. Later conditions are still training."
        )
    )
    lines.append("")
    (HERE / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
