"""New seeds and learning rates for the collapsed-bias mechanism.

Seed 0 is not trained. The source is experiments/06_sgd_vs_adam_collapsed_bias
on main commit 50d908b, and that directory is not modified. This process does
not import the CIFAR follow-up or the WikiText trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = (1, 2, 3, 4)
SGD_LRS = (0.01, 0.05, 0.1, 0.2)
ADAM_LRS = (1e-4, 3e-4, 1e-3, 3e-3)
BIASES = (0.0, -12.0)
LOSSES_ORDER = ("log_loss", "brier")
PUBLISHED = {("sgd", 0.05), ("adam", 0.001)}
STALL_BIAS = 1e-3
RECOVERED_RECALL = 0.5
LOGIT_SPECS = (("sgd", 0.1, 0.0), ("sgd", 0.1, 0.9), ("adam", 1e-2, 0.0), ("adam", 1e-3, 0.0))
HEAD_SPECS = (("sgd", 0.1, 0.0), ("sgd", 0.1, 0.9), ("adam", 1e-3, 0.0), ("adam", 1e-2, 0.0))

# Printed in results.md on 11-06-2026. Not re-executed. Rounded as published.
SEED0_CNN = [
    {"loss": "log_loss", "optimizer": "sgd", "lr": 0.05, "start_bias": 0.0, "accuracy": 0.9717, "collapsed_recall": 0.988, "final_bias": -0.50, "mean_logit0": 0.94},
    {"loss": "brier", "optimizer": "sgd", "lr": 0.05, "start_bias": 0.0, "accuracy": 0.9643, "collapsed_recall": 0.982, "final_bias": -0.37, "mean_logit0": 2.51},
    {"loss": "log_loss", "optimizer": "sgd", "lr": 0.05, "start_bias": -12.0, "accuracy": 0.9717, "collapsed_recall": 0.993, "final_bias": -8.07, "mean_logit0": 1.38},
    {"loss": "brier", "optimizer": "sgd", "lr": 0.05, "start_bias": -12.0, "accuracy": 0.8694, "collapsed_recall": 0.000, "final_bias": -12.00, "mean_logit0": -12.83},
    {"loss": "log_loss", "optimizer": "adam", "lr": 0.001, "start_bias": -12.0, "accuracy": 0.9385, "collapsed_recall": 0.972, "final_bias": -11.99, "mean_logit0": 0.96},
    {"loss": "brier", "optimizer": "adam", "lr": 0.001, "start_bias": -12.0, "accuracy": 0.9567, "collapsed_recall": 0.978, "final_bias": -11.91, "mean_logit0": 4.34},
]


def brier_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    q = logits.softmax(dim=-1)
    q_y = q.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (q.square().sum(dim=-1) + 1.0 - 2.0 * q_y).mean()


def log_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


LOSSES = {"log_loss": log_loss, "brier": brier_loss}


def make_opt(params, name: str, lr: float, momentum: float):
    if name == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
    elif name == "adam":
        opt = torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(name)
    if any(group["weight_decay"] != 0.0 for group in opt.param_groups):
        raise RuntimeError("weight decay must stay at the library default of 0")
    return opt


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


def label_for(recall: float, bias_change: float) -> str:
    if recall == 0.0 and abs(bias_change) < STALL_BIAS:
        return "stalled"
    if recall >= RECOVERED_RECALL:
        return "recovered"
    return "partial"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def load_state(out: Path) -> dict:
    path = out / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"markers": [], "records": []}


def remember(out: Path, state: dict, key: str, row: dict) -> None:
    state["records"].append(row)
    state["markers"].append(key)
    atomic_json(out / "state.json", state)


def load_mnist(data: Path, device: torch.device):
    from torchvision import datasets, transforms

    root = data / "mnist"
    tfm = transforms.ToTensor()
    train_set = datasets.MNIST(str(root), train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(str(root), train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    counts = torch.bincount(test_set.targets, minlength=10)
    print(f"MNIST test class counts {counts.tolist()}", flush=True)
    return train_loader, test_loader, device


@torch.no_grad()
def cnn_metrics(model, loader, device) -> dict:
    model.eval()
    correct = total = hit = collapsed = 0
    logit_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(-1)
        total += int(y.numel())
        correct += int((pred == y).sum())
        mask = y == 0
        collapsed += int(mask.sum())
        hit += int((pred[mask] == 0).sum())
        logit_sum += float(logits[:, 0].sum())
    return {
        "accuracy": correct / total,
        "collapsed_recall": hit / max(collapsed, 1),
        "class0_hit": hit,
        "class0_count": collapsed,
        "mean_logit0": logit_sum / total,
        "bias0": float(model.fc.bias.detach()[0]),
    }


def cnn_runs():
    for loss_name in LOSSES_ORDER:
        for opt_name, lrs in (("sgd", SGD_LRS), ("adam", ADAM_LRS)):
            for lr in lrs:
                for bias in BIASES:
                    for seed in SEEDS:
                        yield loss_name, opt_name, lr, bias, seed


def run_cnn(device, train_loader, test_loader, loss_name, opt_name, lr, bias, seed) -> dict:
    torch.manual_seed(seed)
    model = SmallCNN().to(device)
    with torch.no_grad():
        model.fc.bias.zero_()
        model.fc.bias[0] = bias
    opt = make_opt(model.parameters(), opt_name, lr, 0.9 if opt_name == "sgd" else 0.0)
    fn = LOSSES[loss_name]
    x0, y0 = next(iter(train_loader))
    opt.zero_grad(set_to_none=True)
    fn(model(x0.to(device)), y0.to(device)).backward()
    probe_grad0 = float(model.fc.bias.grad[0])
    model.zero_grad(set_to_none=True)
    t0 = time.time()
    first = None
    abs_bias = 0.0
    n_steps = 0
    trajectory = []
    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            fn(model(x), y).backward()
            grad0 = float(model.fc.bias.grad[0])
            before = [p.detach().clone() for p in model.parameters()]
            bias_before = float(model.fc.bias.detach()[0])
            opt.step()
            sq = 0.0
            for p, old in zip(model.parameters(), before):
                sq += float((p.detach() - old).square().sum())
            step_bias = abs(float(model.fc.bias.detach()[0]) - bias_before)
            if first is None:
                first = {"grad0": grad0, "abs_bias_step": step_bias, "update_l2": sq ** 0.5}
            abs_bias += step_bias
            n_steps += 1
        metrics = cnn_metrics(model, test_loader, device)
        trajectory.append({"epoch": epoch + 1, **metrics})
    final = trajectory[-1]
    delta = final["bias0"] - bias
    row = {
        "level": "full_cnn",
        "executed": True,
        "loss": loss_name,
        "optimizer": opt_name,
        "optimizer_class": "torch.optim.SGD" if opt_name == "sgd" else "torch.optim.Adam",
        "lr": lr,
        "momentum": 0.9 if opt_name == "sgd" else 0.0,
        "weight_decay": 0.0,
        "start_bias": bias,
        "seed": seed,
        "epochs": 3,
        "probe_grad0": probe_grad0,
        "first_step": first,
        "mean_abs_bias_step": abs_bias / max(n_steps, 1),
        "steps": n_steps,
        "seconds": time.time() - t0,
        "trajectory": trajectory,
        "accuracy": final["accuracy"],
        "collapsed_recall": final["collapsed_recall"],
        "class0_hit": final["class0_hit"],
        "class0_count": final["class0_count"],
        "mean_logit0": final["mean_logit0"],
        "bias0": final["bias0"],
        "delta_bias": delta,
        "outcome": label_for(final["collapsed_recall"], delta),
        "published_lr": (opt_name, lr) in PUBLISHED,
    }
    print(
        f"CNN {loss_name} {opt_name} lr {lr} bias {bias} seed {seed} "
        f"recall {row['collapsed_recall']:.4f} bias {row['bias0']:.4f} {row['outcome']}",
        flush=True,
    )
    return row


def run_logits(device, seed: int) -> list[dict]:
    rows = []
    n, k = 1000, 10
    for loss_name in LOSSES_ORDER:
        for opt_name, lr, momentum in LOGIT_SPECS:
            torch.manual_seed(seed)
            y = torch.arange(n, device=device) % k
            theta = torch.zeros(k, device=device, requires_grad=True)
            with torch.no_grad():
                theta.zero_()
                theta[0] = -12.0
            opt = make_opt([theta], opt_name, lr, momentum)
            probe = theta.detach().clone().requires_grad_(True)
            LOSSES[loss_name](probe[None].expand(n, -1), y).backward()
            grad0 = float(probe.grad[0])
            before = float(theta.detach()[0])
            opt.zero_grad(set_to_none=True)
            LOSSES[loss_name](theta[None].expand(n, -1), y).backward()
            first_grad = float(theta.grad[0])
            opt.step()
            first_step = abs(float(theta.detach()[0]) - before)
            for _ in range(1999):
                opt.zero_grad(set_to_none=True)
                LOSSES[loss_name](theta[None].expand(n, -1), y).backward()
                opt.step()
            logit0 = float(theta.detach()[0])
            delta = logit0 + 12.0
            row = {
                "level": "logits",
                "executed": True,
                "loss": loss_name,
                "optimizer": opt_name,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": 0.0,
                "seed": seed,
                "steps": 2000,
                "probe_grad0": grad0,
                "first_step_grad0": first_grad,
                "first_abs_step": first_step,
                "logit0": logit0,
                "delta": delta,
                "outcome": "stalled" if abs(delta) < STALL_BIAS else "moved",
            }
            rows.append(row)
            print(f"LOGIT {loss_name} {opt_name} lr {lr} mom {momentum} seed {seed} delta {delta:.4f}", flush=True)
    return rows


def run_heads(device, train_loader, test_loader, seed: int) -> list[dict]:
    torch.manual_seed(seed)
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
    for loss_name in LOSSES_ORDER:
        for opt_name, lr, momentum in HEAD_SPECS:
            torch.manual_seed(seed)
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
            probe_grad0 = float(head.bias.grad[0])
            head.zero_grad(set_to_none=True)
            t0 = time.time()
            first = None
            for _ in range(8):
                head.train()
                perm = torch.randperm(train_h.shape[0])
                for i in range(0, perm.numel() - 128, 128):
                    batch = perm[i : i + 128]
                    opt.zero_grad(set_to_none=True)
                    fn(head(train_h[batch].to(device)), train_y[batch].to(device)).backward()
                    if first is None:
                        grad0 = float(head.bias.grad[0])
                        before = float(head.bias.detach()[0])
                        opt.step()
                        first = {
                            "grad0": grad0,
                            "abs_bias_step": abs(float(head.bias.detach()[0]) - before),
                        }
                    else:
                        opt.step()
            head.eval()
            with torch.no_grad():
                logits = head(test_h.to(device))
                pred = logits.argmax(-1)
                y = test_y.to(device)
                mask = y == 0
                recall = float((pred[mask] == 0).float().mean())
                bias0 = float(head.bias.detach()[0])
            delta = bias0 + 12.0
            row = {
                "level": "frozen_head",
                "executed": True,
                "loss": loss_name,
                "optimizer": opt_name,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": 0.0,
                "seed": seed,
                "epochs": 8,
                "probe_grad0": probe_grad0,
                "first_step": first,
                "seconds": time.time() - t0,
                "collapsed_recall": recall,
                "bias0": bias0,
                "delta_bias": delta,
                "outcome": label_for(recall, delta),
            }
            rows.append(row)
            print(
                f"HEAD {loss_name} {opt_name} lr {lr} mom {momentum} seed {seed} "
                f"recall {recall:.4f} bias {bias0:.4f}",
                flush=True,
            )
    return rows


def sign_flip(diffs: list[float]) -> dict:
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    observed = sum(diffs) / n
    signs = [1 if d > 0 else -1 if d < 0 else 0 for d in diffs]
    extreme = 0
    total = 1 << n
    for mask in range(total):
        total_signed = 0.0
        for i, value in enumerate(diffs):
            total_signed += value if (mask >> i) & 1 else -value
        if abs(total_signed / n) >= abs(observed) - 1e-15:
            extreme += 1
    return {"n": n, "mean_diff": observed, "p_value": extreme / total, "nonzero_signs": sum(s != 0 for s in signs)}


def render(out: Path, state: dict) -> None:
    rows = state["records"]
    lines = [
        "# Collapsed-bias replication, seeds 1-4",
        "",
        f"Status: {state.get('status', 'RUNNING')}",
        "",
        "Seed 0 was not trained. Its full-CNN rows are the printed values from",
        "`experiments/06_sgd_vs_adam_collapsed_bias/results.md` on 11-06-2026.",
        "",
        "## Seed 0, not re-executed",
        "",
        "| Loss | Optimizer | Start bias | Accuracy | Class-0 recall | Final bias |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in SEED0_CNN:
        lines.append(
            f"| {row['loss']} | {row['optimizer']} | {row['start_bias']} | {row['accuracy']:.4f} | "
            f"{row['collapsed_recall']:.3f} | {row['final_bias']:.2f} |"
        )
    lines += [
        "",
        "## New full-CNN runs",
        "",
        "| Seed | Loss | Optimizer | lr | Start bias | Recall | Hits/count | Final bias | Probe grad | First update L2 | Outcome |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["level"] != "full_cnn":
            continue
        lines.append(
            f"| {row['seed']} | {row['loss']} | {row['optimizer']} | {row['lr']} | {row['start_bias']} | "
            f"{row['collapsed_recall']:.4f} | {row['class0_hit']}/{row['class0_count']} | {row['bias0']:.4f} | "
            f"{row['probe_grad0']:.6g} | {row['first_step']['update_l2']:.6g} | {row['outcome']} |"
        )
    published = [
        r
        for r in rows
        if r["level"] == "full_cnn" and r["published_lr"] and r["start_bias"] == -12.0
    ]
    lines += ["", "## Paired recall at the published learning rates, start bias -12, seeds 1-4", ""]
    for opt_name in ("sgd", "adam"):
        by_seed = {}
        for row in published:
            if row["optimizer"] != opt_name:
                continue
            by_seed.setdefault(row["seed"], {})[row["loss"]] = row["collapsed_recall"]
        diffs = []
        for seed in SEEDS:
            pair = by_seed.get(seed, {})
            if "log_loss" in pair and "brier" in pair:
                diffs.append(pair["log_loss"] - pair["brier"])
        comp = sign_flip(diffs)
        if comp["n"]:
            lines.append(
                f"{opt_name}: mean log-loss recall minus Brier recall {comp['mean_diff']:.4f}, "
                f"sign-flip p {comp['p_value']:.4g}, n {comp['n']}."
            )
    lines.append("")
    (out / "results.md").write_text("\n".join(lines) + "\n")


def write_protocol(out: Path, smoke_seconds: float | None) -> None:
    runs = 128 + 32 + 32
    if smoke_seconds is None:
        watchdog = None
    else:
        watchdog = min(smoke_seconds * 160 * 2 + 900, 6 * 3600)
    payload = {
        "study": "collapsed bias seeds 1-4 and learning-rate grid",
        "source": "experiments/06_sgd_vs_adam_collapsed_bias/train_validate.py at main 50d908b",
        "seed0": "not executed; copied from results.md",
        "seeds": list(SEEDS),
        "optimizers": {
            "sgd": {"class": "torch.optim.SGD", "momentum": 0.9, "weight_decay": 0.0, "learning_rates": list(SGD_LRS)},
            "adam": {
                "class": "torch.optim.Adam",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "learning_rates": list(ADAM_LRS),
            },
        },
        "start_biases": list(BIASES),
        "epochs": 3,
        "architecture": "Conv 1-32, 32-64, 64-128, then Linear 128-10",
        "cnn_runs": 128,
        "logit_runs": 32,
        "frozen_head_runs": 32,
        "stall_abs_bias_change": STALL_BIAS,
        "recovered_recall": RECOVERED_RECALL,
        "smoke_seconds": smoke_seconds,
        "watchdog_seconds": watchdog,
        "planned_runs": runs,
        "no_learning_rate_selection": True,
        "wikitext_unchanged": True,
    }
    atomic_json(out / "protocol.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "train"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("CUDA_VISIBLE_DEVICES did not pin exactly one GPU")
    device = torch.device("cuda")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    forbidden = {
        "GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697",
        "GPU-071c409d-a1bb-9fd7-1239-a271b8ce1889",
    }
    if visible in forbidden:
        raise SystemExit(f"refusing forbidden GPU {visible}")
    args.out.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader, device = load_mnist(args.data, device)
    state = load_state(args.out)
    if args.stage == "smoke":
        key = "cnn-brier-sgd-lr0.05-bias-12-seed1"
        if key not in state["markers"]:
            t0 = time.time()
            row = run_cnn(device, train_loader, test_loader, "brier", "sgd", 0.05, -12.0, 1)
            remember(args.out, state, key, row)
            elapsed = time.time() - t0
        else:
            elapsed = float(state["records"][state["markers"].index(key)].get("seconds", 30.0))
        write_protocol(args.out, elapsed)
        render(args.out, state)
        print(f"SMOKE_SECONDS {elapsed:.3f}", flush=True)
        return
    if not (args.out / "protocol.json").exists():
        raise SystemExit("protocol.json missing; smoke has not recorded the watchdog")
    for loss_name, opt_name, lr, bias, seed in cnn_runs():
        key = f"cnn-{loss_name}-{opt_name}-lr{lr:.8g}-bias{bias:g}-seed{seed}"
        if key in state["markers"]:
            continue
        row = run_cnn(device, train_loader, test_loader, loss_name, opt_name, lr, bias, seed)
        remember(args.out, state, key, row)
        render(args.out, state)
    for seed in SEEDS:
        key = f"logits-seed{seed}"
        if key not in state["markers"]:
            for row in run_logits(device, seed):
                state["records"].append(row)
            state["markers"].append(key)
            atomic_json(args.out / "state.json", state)
            render(args.out, state)
        key = f"head-seed{seed}"
        if key not in state["markers"]:
            for row in run_heads(device, train_loader, test_loader, seed):
                state["records"].append(row)
            state["markers"].append(key)
            atomic_json(args.out / "state.json", state)
            render(args.out, state)
    state["status"] = "DONE"
    atomic_json(args.out / "state.json", state)
    render(args.out, state)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
