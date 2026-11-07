"""Matched Brier score versus cross-entropy training.

Smoke mode checks the losses, one AlexNet, and the language model on one GPU.
Full mode is the comparison fixed in README.md: separate learning-rate grids
with equal trial counts, then multi-seed training at the selected learning
rates. The held-out test split is not constructed until that selection is on disk.

The wall clock for full mode is 12 hours from the first start, including a
resume. One GPU. New files in the job directory stay under 6GB because the
node scratch has little free space.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
import traceback
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from aggregate import write_results
from data_util import (
    add_one_unigram,
    encode_file,
    load_or_make_split,
    pack_sequences,
    prepare_wikitext,
)
from losses import LOSSES, brier_per_example, collapsed_logit_check, probabilities
from models import AlexNetSmallImage, DenseCausalLM, adamw_groups, count_parameters

CNN_LRS = (3e-4, 1e-3, 3e-3, 1e-2)
LM_LRS = (1e-4, 3e-4, 1e-3, 3e-3)
CNN_WD = 5e-4
LM_WD = 0.1
BETAS = (0.9, 0.95)
VISION_BATCH = 128
LM_BATCH = 8
SEQ_LEN = 256
LM_TUNE_TOKENS = 2_000_000
FULL_SECONDS = 12 * 3600
OWN_CAP = 6 * 1024**3
GRAD_CLIP = 1.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def free_bytes(path: Path) -> int:
    stat = os.statvfs(path)
    return int(stat.f_bavail * stat.f_frsize)


def own_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


class TestGate:
    """Per-dataset permission to read the held-out test split."""

    def __init__(self):
        self.allowed: set[str] = set()
        self.opened: set[str] = set()

    def allow(self, name: str) -> None:
        self.allowed.add(name)

    def open(self, name: str) -> None:
        if name not in self.allowed:
            raise RuntimeError(f"test split for {name} opened before selection was saved")
        self.opened.add(name)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * float(step + 1) / float(max(1, warmup))
    progress = min(1.0, (step - warmup) / float(max(1, total - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (0.1 + 0.9 * cosine)


def set_lr(opt, value: float) -> None:
    for group in opt.param_groups:
        group["lr"] = value


class Meter:
    def __init__(self):
        self.n = 0
        self.sums = {
            "cross_entropy": 0.0,
            "brier": 0.0,
            "local_square": 0.0,
            "accuracy": 0.0,
            "true_prob": 0.0,
            "true_logit": 0.0,
            "abs_logit": 0.0,
        }
        self.conf = []
        self.correct = []

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        logits = logits.detach().float()
        q = probabilities(logits)
        q_y = q.gather(-1, y[:, None]).squeeze(-1)
        brier = brier_per_example(logits, y)
        local = (1.0 - q_y).square()
        ce = -q_y.clamp_min(1e-12).log()
        pred = logits.argmax(dim=-1)
        correct = pred.eq(y)
        count = int(y.numel())
        self.n += count
        self.sums["cross_entropy"] += float(ce.sum())
        self.sums["brier"] += float(brier.sum())
        self.sums["local_square"] += float(local.sum())
        self.sums["accuracy"] += float(correct.float().sum())
        self.sums["true_prob"] += float(q_y.sum())
        self.sums["true_logit"] += float(logits.gather(-1, y[:, None]).sum())
        self.sums["abs_logit"] += float(logits.abs().sum())
        self.conf.append(q.max(dim=-1).values.cpu())
        self.correct.append(correct.cpu())

    def metrics(self) -> dict:
        if self.n == 0:
            raise RuntimeError("evaluation saw no examples")
        ce = self.sums["cross_entropy"] / self.n
        conf = torch.cat(self.conf)
        correct = torch.cat(self.correct).float()
        return {
            "accuracy": self.sums["accuracy"] / self.n,
            "cross_entropy": ce,
            "perplexity": math.exp(ce) if ce < 50 else None,
            "brier": self.sums["brier"] / self.n,
            "local_square": self.sums["local_square"] / self.n,
            "brier_minus_local_square": (self.sums["brier"] - self.sums["local_square"]) / self.n,
            "expected_calibration_error": _ece(conf, correct),
            "mean_true_class_probability": self.sums["true_prob"] / self.n,
            "mean_true_class_logit": self.sums["true_logit"] / self.n,
            "mean_abs_logit": self.sums["abs_logit"] / self.n,
            "n": self.n,
        }


def _ece(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    total = conf.numel()
    edges = torch.linspace(0, 1, n_bins + 1)
    total_gap = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf > edges[i]) & (conf <= edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        total_gap += abs(float(correct[mask].mean()) - float(conf[mask].mean())) * (count / total)
    return total_gap


def autocast_for(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def placement() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = {
        "hostname": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_uuid_env": os.environ.get("GPU_UUID", ""),
        "device": str(device),
        "torch": torch.__version__,
        "started": now_iso(),
    }
    if device.type == "cuda":
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        if info["device_count"] != 1:
            raise SystemExit("CUDA_VISIBLE_DEVICES did not pin exactly one GPU")
    return info


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


class Run:
    def __init__(self, args):
        self.out = Path(args.out)
        self.data = Path(args.data)
        self.job = Path(args.job_root)
        self.mode = args.mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "state.json"
        if self.state_path.exists() and self.mode == "full":
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {
                "status": "RUNNING",
                "mode": self.mode,
                "updated": now_iso(),
                "phases": {},
                "records": [],
                "selection": {},
                "notes": [],
                "budget": {},
                "placement": placement(),
            }
        if self.mode == "full":
            self.state.setdefault("deadline", time.time() + FULL_SECONDS)
        else:
            self.state["deadline"] = time.time() + 18 * 60
        self.gate = TestGate()
        for name, chosen in self.state.get("selection", {}).items():
            if chosen:
                self.gate.allow(name)
        self.save()

    def save(self) -> None:
        self.state["updated"] = now_iso()
        self.state["own_bytes"] = own_bytes(self.job)
        self.state["free_bytes"] = free_bytes(self.job)
        write_results(self.out, self.state)
        atomic_json(self.state_path, self.state)

    def note(self, text: str) -> None:
        print(text, flush=True)
        self.state.setdefault("notes", []).append(f"{now_iso()} {text}")
        self.save()

    def time_left(self) -> float:
        return self.state["deadline"] - time.time()

    def over_cap(self) -> bool:
        return own_bytes(self.job) > OWN_CAP or free_bytes(self.job) < 4 * 1024**3

    def done_key(self, key: str) -> bool:
        return self.state["phases"].get(key) == "DONE"


def make_optimizer(model, weight_decay: float):
    return torch.optim.AdamW(adamw_groups(model, weight_decay), lr=1e-3, betas=BETAS)


def vision_sets(name: str, data_root: Path, gate: TestGate, need_test: bool):
    from torchvision import datasets, transforms

    if name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_full = datasets.CIFAR10(str(data_root), train=True, download=True, transform=transform)
        in_ch = 3
        official_test = lambda: datasets.CIFAR10(str(data_root), train=False, download=True, transform=transform)
    elif name == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_full = datasets.MNIST(str(data_root), train=True, download=True, transform=transform)
        in_ch = 1
        official_test = lambda: datasets.MNIST(str(data_root), train=False, download=True, transform=transform)
    else:
        raise ValueError(name)
    split = load_or_make_split(data_root / f"{name}_split_seed123.json", len(train_full))
    train_set = Subset(train_full, split["train_idx"])
    val_set = Subset(train_full, split["val_idx"])
    test_set = None
    if need_test:
        gate.open(name)
        test_set = official_test()
    return in_ch, train_set, val_set, test_set, split


def loader(dataset, seed: int, epoch: int, batch: int, shuffle: bool):
    generator = torch.Generator().manual_seed(seed + 1000 * epoch + (17 if shuffle else 0))
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def train_vision_once(run: Run, dataset: str, loss_name: str, lr: float, seed: int, epochs: int, init_state, phase: str, eval_test: bool) -> dict:
    in_ch, train_set, val_set, test_set, split = vision_sets(
        dataset, run.data, run.gate, need_test=eval_test
    )
    model = AlexNetSmallImage(in_ch).to(run.device)
    model.load_state_dict(init_state)
    seed_everything(seed)
    opt = make_optimizer(model, CNN_WD)
    loss_fn = LOSSES[loss_name]
    steps_per_epoch = math.ceil(len(train_set) / VISION_BATCH)
    total_steps = epochs * steps_per_epoch
    warmup = min(100, max(1, total_steps // 10))
    step = 0
    train_losses = []
    started = time.time()
    invalid = False
    if run.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        if run.time_left() < 120:
            run.note(f"deadline before finishing {dataset} {loss_name} seed {seed}")
            invalid = True
            break
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        for x, y in loader(train_set, seed, epoch, VISION_BATCH, True):
            x = x.to(run.device, non_blocking=True)
            y = y.to(run.device, non_blocking=True)
            set_lr(opt, lr_at(step, total_steps, warmup, lr))
            opt.zero_grad(set_to_none=True)
            with autocast_for(run.device):
                logits = model(x)
                loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                invalid = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_loss += float(loss.detach()) * int(y.shape[0])
            epoch_n += int(y.shape[0])
            step += 1
        if invalid:
            break
        train_losses.append(epoch_loss / max(1, epoch_n))
        print(
            f"{dataset} {loss_name} seed {seed} lr {lr} epoch {epoch} train_loss {train_losses[-1]:.4f}",
            flush=True,
        )
    val_metrics = None if invalid else evaluate_vision(model, val_set, run.device, seed)
    test_metrics = None
    if eval_test and not invalid and test_set is not None:
        test_metrics = evaluate_vision(model, test_set, run.device, seed + 99)
    record = {
        "dataset": dataset,
        "loss": loss_name,
        "seed": seed,
        "learning_rate": lr,
        "phase": phase,
        "epochs": epochs,
        "steps": step,
        "seconds": time.time() - started,
        "invalid": invalid,
        "train_loss_per_epoch": train_losses,
        "n_train": len(split["train_idx"]),
        "n_val": len(split["val_idx"]),
        "weight_decay": CNN_WD,
        "n_params": count_parameters(model),
        "metrics": val_metrics,
        "split": "validation",
    }
    if run.device.type == "cuda":
        record["peak_bytes"] = int(torch.cuda.max_memory_allocated())
    run.state["records"].append(record)
    if test_metrics is not None:
        test_record = dict(record)
        test_record["metrics"] = test_metrics
        test_record["split"] = "test"
        run.state["records"].append(test_record)
    run.save()
    return record


def evaluate_vision(model, dataset, device, seed: int) -> dict:
    model.eval()
    meter = Meter()
    with torch.no_grad():
        for x, y in loader(dataset, seed, 0, VISION_BATCH, False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_for(device):
                logits = model(x)
            meter.update(logits, y)
    return {key: finite(value) if isinstance(value, float) else value for key, value in meter.metrics().items()}


def select_lr(run: Run, dataset: str) -> dict:
    chosen = {}
    for loss_name in ("cross_entropy", "brier"):
        trials = [
            r
            for r in run.state["records"]
            if r["dataset"] == dataset
            and r["phase"] == "tune"
            and r["loss"] == loss_name
            and r["split"] == "validation"
            and not r["invalid"]
            and not r.get("incomplete")
            and r.get("metrics")
        ]
        if not trials:
            run.note(f"no valid tune trial for {dataset} {loss_name}")
            continue
        metric = "cross_entropy" if loss_name == "cross_entropy" else "brier"
        trials.sort(key=lambda r: (r["metrics"][metric], -r["metrics"]["accuracy"], r["learning_rate"]))
        chosen[loss_name] = trials[0]["learning_rate"]
    run.state["selection"][dataset] = chosen
    run.save()
    if chosen:
        run.gate.allow(dataset)
    return chosen


def vision_init(in_ch: int, seed: int, device) -> dict:
    seed_everything(seed)
    model = AlexNetSmallImage(in_ch)
    state = copy.deepcopy(model.state_dict())
    return state


def run_vision_tune(run: Run, dataset: str) -> None:
    key = f"{dataset}_tune"
    if run.done_key(key):
        return
    run.state["phases"][key] = "RUNNING"
    run.save()
    # Fetch the official training set before the timed trial so a download
    # is not counted as an epoch.
    vision_sets(dataset, run.data, run.gate, need_test=False)
    in_ch = 3 if dataset == "cifar10" else 1
    init = vision_init(in_ch, 0, run.device)
    first_seconds = None
    for loss_name in ("cross_entropy", "brier"):
        for lr in CNN_LRS:
            marker = f"{dataset}-tune-{loss_name}-{lr}"
            if marker in run.state.get("completed_markers", []):
                continue
            if run.time_left() < 180:
                run.note(f"skip remaining {dataset} tune, time left {run.time_left():.0f}s")
                break
            started = time.time()
            train_vision_once(run, dataset, loss_name, lr, 0, 1, init, "tune", False)
            run.state.setdefault("completed_markers", []).append(marker)
            run.save()
            if first_seconds is None and dataset == "cifar10":
                first_seconds = time.time() - started
                apply_epoch_rule(run, first_seconds)
    select_lr(run, dataset)
    run.state["phases"][key] = "DONE"
    run.save()


def apply_epoch_rule(run: Run, first_epoch_seconds: float) -> None:
    """Fixed before the run. Uses the first CIFAR-10 tune epoch wall time."""
    t = first_epoch_seconds
    if t <= 60:
        cifar_epochs, mnist_epochs = 6, 2
    elif t <= 180:
        cifar_epochs, mnist_epochs = 3, 2
    else:
        cifar_epochs, mnist_epochs = 1, 1
    seeds = 5
    projected = t * (cifar_epochs + mnist_epochs) * seeds * 2
    if projected > 2.5 * 3600:
        seeds = 3
    run.state["budget"].update(
        {
            "cifar_first_epoch_seconds": t,
            "cifar_final_epochs": cifar_epochs,
            "mnist_final_epochs": mnist_epochs,
            "cnn_seeds": seeds,
            "cnn_projected_seconds": projected,
        }
    )
    run.note(
        f"epoch rule: first CIFAR epoch {t:.1f}s, CIFAR epochs {cifar_epochs}, "
        f"MNIST epochs {mnist_epochs}, CNN seeds {seeds}"
    )


def run_vision_final(run: Run, dataset: str) -> None:
    key = f"{dataset}_final"
    if run.done_key(key):
        return
    if dataset not in run.state["selection"]:
        run.note(f"no selection for {dataset}")
        run.state["phases"][key] = "BLOCKED"
        run.save()
        return
    run.state["phases"][key] = "RUNNING"
    run.save()
    budget = run.state["budget"]
    epochs = budget.get("cifar_final_epochs" if dataset == "cifar10" else "mnist_final_epochs", 2)
    n_seeds = int(budget.get("cnn_seeds", 3))
    in_ch = 3 if dataset == "cifar10" else 1
    for seed in range(n_seeds):
        init = vision_init(in_ch, seed, run.device)
        for loss_name in ("cross_entropy", "brier"):
            lr = run.state["selection"][dataset].get(loss_name)
            if lr is None:
                continue
            marker = f"{dataset}-final-{loss_name}-seed{seed}"
            if marker in run.state.get("completed_markers", []):
                continue
            if run.time_left() < 180:
                run.note(f"deadline reached during {dataset} finals")
                run.state["phases"][key] = "PARTIAL"
                run.save()
                return
            train_vision_once(run, dataset, loss_name, lr, seed, epochs, init, "final", True)
            run.state.setdefault("completed_markers", []).append(marker)
            run.save()
    run.state["phases"][key] = "DONE"
    run.save()


def build_lm(vocab: int, device) -> DenseCausalLM:
    model = DenseCausalLM(vocab=vocab, d=768, n_layer=12, n_head=12, seq_len=SEQ_LEN)
    n_params = count_parameters(model)
    if not (100_000_000 <= n_params <= 300_000_000):
        raise RuntimeError(f"language model has {n_params} parameters, expected 100M to 300M")
    return model.to(device)


def lm_batch(packed: torch.Tensor, perm: torch.Tensor, step: int, batch: int) -> torch.Tensor:
    """Sequential batches of one seed-fixed permutation. Same seed, same tokens."""
    n_seq = perm.shape[0]
    start = (step * batch) % n_seq
    positions = torch.arange(start, start + batch) % n_seq
    return packed[perm[positions]].long()


def evaluate_lm(model, packed: torch.Tensor, device, batch: int) -> dict:
    model.eval()
    meter = Meter()
    with torch.no_grad():
        for start in range(0, packed.shape[0], batch):
            idx = packed[start : start + batch].long().to(device)
            if idx.shape[0] == 0:
                continue
            with autocast_for(device):
                logits = model(idx)
            meter.update(logits[:, :-1].reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
    return {key: finite(value) if isinstance(value, float) else value for key, value in meter.metrics().items()}


def train_lm_once(
    run: Run,
    packed_train: torch.Tensor,
    packed_val: torch.Tensor,
    loss_name: str,
    lr: float,
    seed: int,
    steps: int,
    init_state,
    phase: str,
    eval_test_packed,
    batch: int,
) -> dict:
    # Rebuild from the saved init so both losses share it, then retie the output layer.
    model = build_lm(init_state["tok_emb.weight"].shape[0], run.device)
    model.load_state_dict(init_state)
    model.tie_weights()
    seed_everything(seed)
    opt = make_optimizer(model, LM_WD)
    perm = torch.randperm(
        packed_train.shape[0], generator=torch.Generator().manual_seed(seed)
    )
    loss_fn = LOSSES[loss_name]
    warmup = max(20, min(steps // 100, 500))
    tokens_per_step = batch * (SEQ_LEN - 1)
    started = time.time()
    invalid = False
    train_curve = []
    step = 0
    if run.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    while step < steps:
        if run.time_left() < 90:
            run.note(f"deadline during LM {loss_name} seed {seed} at step {step}")
            break
        if step % 50 == 0 and run.over_cap():
            run.note("storage cap reached; stopping new LM work after this step's bookkeeping")
            invalid = True
            break
        idx = lm_batch(packed_train, perm, step, batch).to(run.device)
        set_lr(opt, lr_at(step, steps, warmup, lr))
        opt.zero_grad(set_to_none=True)
        try:
            with autocast_for(run.device):
                logits = model(idx)
                loss = loss_fn(logits[:, :-1].reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        if not torch.isfinite(loss):
            invalid = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            train_curve.append({"step": step, "train_loss": float(loss.detach())})
            print(f"lm {loss_name} seed {seed} lr {lr} step {step} loss {float(loss):.4f}", flush=True)
        step += 1
    val_metrics = None if invalid else evaluate_lm(model, packed_val, run.device, batch)
    record = {
        "dataset": "wikitext",
        "loss": loss_name,
        "seed": seed,
        "learning_rate": lr,
        "phase": phase,
        "steps": step,
        "planned_steps": steps,
        "tokens": step * tokens_per_step,
        "seconds": time.time() - started,
        "invalid": invalid,
        "incomplete": step < steps,
        "train_curve": train_curve[-40:],
        "weight_decay": LM_WD,
        "n_params": count_parameters(model),
        "batch": batch,
        "seq_len": SEQ_LEN,
        "metrics": val_metrics,
        "split": "validation",
    }
    if run.device.type == "cuda":
        record["peak_bytes"] = int(torch.cuda.max_memory_allocated())
    run.state["records"].append(record)
    if eval_test_packed is not None and not record["invalid"] and not record["incomplete"]:
        run.gate.open("wikitext")
        test_metrics = evaluate_lm(model, eval_test_packed, run.device, batch)
        test_record = dict(record)
        test_record["metrics"] = test_metrics
        test_record["split"] = "test"
        run.state["records"].append(test_record)
    # Metrics are the retained record. A full Adam checkpoint is about 2GB, and
    # this node's free scratch is only about 14GB, so weights are not kept.
    run.save()
    return record


def prepare_lm_tensors(run: Run):
    import tiktoken

    info = prepare_wikitext(run.data / "wikitext", prefer_103=True)
    encoder = tiktoken.get_encoding("gpt2")
    cache = run.data / "wikitext" / f"{info['name']}_gpt2_seq{SEQ_LEN}.pt"
    meta_path = cache.with_suffix(".json")
    if cache.exists() and meta_path.exists():
        tensors = torch.load(cache, weights_only=True)
        meta = json.loads(meta_path.read_text())
        blob = {**tensors, **meta}
    else:
        train_ids = encode_file(Path(info["train"]), encoder)
        valid_ids = encode_file(Path(info["valid"]), encoder)
        # The test file stays unread until learning-rate selection is on disk.
        train_packed = pack_sequences(train_ids, SEQ_LEN)
        valid_packed = pack_sequences(valid_ids, SEQ_LEN)
        if own_bytes(run.job) + train_packed.nbytes + valid_packed.nbytes > OWN_CAP:
            raise RuntimeError("token cache would pass the 6GB job cap")
        torch.save({"train": train_packed, "valid": valid_packed}, cache)
        meta = {
            "vocab": encoder.n_vocab,
            "info": info,
            "n_train_tokens": int(train_ids.numel()),
            "n_valid_tokens": int(valid_ids.numel()),
        }
        meta_path.write_text(json.dumps(meta) + "\n")
        blob = {"train": train_packed, "valid": valid_packed, **meta}
    p = add_one_unigram(blob["train"].view(-1), blob["vocab"])
    val_tokens = blob["valid"][:, 1:].reshape(-1).long()
    p_y = p[val_tokens]
    ce = float((-p_y.clamp_min(1e-12).log()).mean())
    brier = float((p.square().sum() + 1.0 - 2.0 * p_y).mean())
    pred = int(p.argmax())
    run.state["baseline"] = {
        "name": "add-one unigram fit on training tokens",
        "split": "validation",
        "cross_entropy": ce,
        "perplexity": math.exp(ce) if ce < 50 else None,
        "brier": brier,
        "accuracy": float((val_tokens == pred).float().mean()),
        "corpus": blob["info"]["name"],
        "n_train_tokens": blob["n_train_tokens"],
        "n_valid_tokens": blob["n_valid_tokens"],
        "vocab": blob["vocab"],
        "zip_sha256": blob["info"].get("zip_sha256"),
    }
    run.note(
        f"corpus {blob['info']['name']} train_tokens {blob['n_train_tokens']} "
        f"val_unigram_cross_entropy {ce:.4f}"
    )
    return blob


def load_test_packed(run: Run, blob: dict) -> torch.Tensor:
    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    test_ids = encode_file(Path(blob["info"]["test"]), encoder)
    packed = pack_sequences(test_ids, SEQ_LEN)
    run.state["baseline"]["n_test_tokens"] = int(test_ids.numel())
    run.note(f"opened held-out test tokens {int(test_ids.numel())}")
    return packed


def run_lm(run: Run) -> None:
    if run.done_key("lm_final") and run.done_key("lm_tune"):
        return
    if run.time_left() < 600:
        run.note("less than 10 minutes left; not starting the language model")
        run.state["phases"]["lm_tune"] = "SKIPPED"
        run.state["phases"]["lm_final"] = "SKIPPED"
        run.save()
        return
    try:
        blob = prepare_lm_tensors(run)
    except Exception as exc:
        run.note(f"language-model data blocked: {exc}")
        run.state["phases"]["lm_tune"] = "BLOCKED"
        run.state["phases"]["lm_final"] = "BLOCKED"
        run.save()
        return
    batch = LM_BATCH
    tokens_per_step = batch * (SEQ_LEN - 1)
    tune_steps = max(20, math.ceil(LM_TUNE_TOKENS / tokens_per_step))
    if not run.done_key("lm_tune"):
        run.state["phases"]["lm_tune"] = "RUNNING"
        run.save()
        seed_everything(0)
        init_model = build_lm(blob["vocab"], run.device)
        init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
        run.note(f"language model parameters {count_parameters(init_model)}")
        del init_model
        if run.device.type == "cuda":
            torch.cuda.empty_cache()
        step_seconds = []
        for loss_name in ("cross_entropy", "brier"):
            for lr in LM_LRS:
                marker = f"lm-tune-{loss_name}-{lr}"
                if marker in run.state.get("completed_markers", []):
                    continue
                if run.time_left() < 300:
                    break
                try:
                    rec = train_lm_once(
                        run,
                        blob["train"],
                        blob["valid"],
                        loss_name,
                        lr,
                        0,
                        tune_steps,
                        init_state,
                        "tune",
                        None,
                        batch,
                    )
                except torch.cuda.OutOfMemoryError:
                    if batch == LM_BATCH:
                        batch = 4
                        tokens_per_step = batch * (SEQ_LEN - 1)
                        tune_steps = max(20, math.ceil(LM_TUNE_TOKENS / tokens_per_step))
                        run.note("LM batch reduced from 8 to 4 after out-of-memory, same for both losses")
                        rec = train_lm_once(
                            run, blob["train"], blob["valid"], loss_name, lr, 0, tune_steps, init_state, "tune", None, batch
                        )
                    else:
                        raise
                if rec["steps"] > 0:
                    step_seconds.append(rec["seconds"] / rec["steps"])
                if not rec.get("incomplete"):
                    run.state.setdefault("completed_markers", []).append(marker)
                    run.save()
        select_lr(run, "wikitext")
        run.state["budget"]["lm_step_seconds"] = sum(step_seconds) / len(step_seconds) if step_seconds else None
        run.state["budget"]["lm_batch"] = batch
        run.state["phases"]["lm_tune"] = "DONE"
        run.save()
    if run.done_key("lm_final"):
        return
    run.state["phases"]["lm_final"] = "RUNNING"
    step_seconds = run.state["budget"].get("lm_step_seconds") or 0.5
    batch = int(run.state["budget"].get("lm_batch", LM_BATCH))
    tokens_per_step = batch * (SEQ_LEN - 1)
    remaining = max(0.0, run.time_left() - 300)
    # Three seeds if each run can see 20M tokens; otherwise two. Equal steps either way.
    for n_seeds in (3, 2):
        n_runs = n_seeds * 2
        steps = int(remaining / step_seconds / n_runs) if step_seconds > 0 else 0
        tokens = steps * tokens_per_step
        if n_seeds == 3 and tokens >= 20_000_000:
            break
        if n_seeds == 2:
            break
    if steps < 20:
        run.note("not enough time for a language-model final pair")
        run.state["phases"]["lm_final"] = "SKIPPED"
        run.save()
        return
    run.state["budget"].update(
        {
            "lm_final_seeds": n_seeds,
            "lm_final_steps": steps,
            "lm_final_tokens_per_run": tokens,
            "lm_step_seconds": step_seconds,
        }
    )
    run.note(f"LM finals: {n_seeds} seeds, {steps} steps, {tokens} tokens per run, batch {batch}")
    test_packed = None
    for seed in range(n_seeds):
        seed_everything(seed)
        init_model = build_lm(blob["vocab"], run.device)
        init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
        del init_model
        if run.device.type == "cuda":
            torch.cuda.empty_cache()
        for loss_name in ("cross_entropy", "brier"):
            lr = run.state["selection"].get("wikitext", {}).get(loss_name)
            if lr is None:
                continue
            marker = f"lm-final-{loss_name}-seed{seed}"
            if marker in run.state.get("completed_markers", []):
                continue
            if test_packed is None:
                # Selection is already on disk. Reading the test file starts here.
                run.gate.allow("wikitext")
                test_packed = load_test_packed(run, blob)
            train_lm_once(
                run,
                blob["train"],
                blob["valid"],
                loss_name,
                lr,
                seed,
                steps,
                init_state,
                "final",
                test_packed,
                batch,
            )
            run.state.setdefault("completed_markers", []).append(marker)
            run.save()
    run.state["phases"]["lm_final"] = "DONE"
    run.save()


def run_controls(run: Run) -> None:
    if run.done_key("controls"):
        return
    rows = collapsed_logit_check(run.device)
    run.state["controls"] = rows
    run.state["phases"]["controls"] = "DONE"
    run.note(f"collapsed-logit check {json.dumps(rows)}")


def run_full(run: Run) -> None:
    run.state["status"] = "RUNNING"
    run_controls(run)
    run_vision_tune(run, "mnist")
    run_vision_tune(run, "cifar10")
    if "cnn_seeds" not in run.state["budget"]:
        apply_epoch_rule(run, 120.0)
        run.note("CIFAR timing missing; used the 120s branch of the epoch rule")
    run_vision_final(run, "mnist")
    run_vision_final(run, "cifar10")
    run_lm(run)
    unfinished = [k for k, v in run.state["phases"].items() if v not in ("DONE", "SKIPPED", "BLOCKED")]
    run.state["status"] = "DONE" if not unfinished else "PARTIAL"
    run.save()
    print("FULL_DONE", run.state["status"], flush=True)


def run_smoke(run: Run) -> int:
    if os.environ.get("GPU_UUID") and run.device.type != "cuda":
        run.note("SMOKE did not see the pinned GPU")
        print("SMOKE_FAIL", flush=True)
        return 1
    run_controls(run)
    in_ch, train_set, val_set, _, _ = vision_sets("mnist", run.data, run.gate, need_test=False)
    init = vision_init(in_ch, 0, run.device)
    # Short MNIST runs at two learning rates. Test split stays closed.
    for loss_name in ("cross_entropy", "brier"):
        for lr in (1e-3, 3e-3):
            model = AlexNetSmallImage(in_ch).to(run.device)
            model.load_state_dict(init)
            seed_everything(0)
            opt = make_optimizer(model, CNN_WD)
            loss_fn = LOSSES[loss_name]
            losses = []
            for step, (x, y) in enumerate(loader(train_set, 0, 0, 128, True)):
                if step >= 60:
                    break
                x = x.to(run.device)
                y = y.to(run.device)
                set_lr(opt, lr)
                opt.zero_grad(set_to_none=True)
                with autocast_for(run.device):
                    loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    run.note(f"SMOKE nonfinite {loss_name} {lr}")
                    print("SMOKE_FAIL", flush=True)
                    return 1
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            val = evaluate_vision(model, val_set, run.device, 0)
            run.note(f"smoke mnist {loss_name} lr {lr} start {losses[0]:.4f} end {losses[-1]:.4f} val_acc {val['accuracy']:.3f}")
            if loss_name == "cross_entropy" and lr == 1e-3 and not (losses[-1] < losses[0]):
                run.note("SMOKE cross-entropy did not decrease on MNIST")
                print("SMOKE_FAIL", flush=True)
                return 1
    # Real language-model step, tiny token batch, no test corpus.
    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    model = build_lm(encoder.n_vocab, run.device)
    run.note(f"smoke LM parameters {count_parameters(model)}")
    text = ("The full Brier score sums every class. Cross-entropy uses the observed class. " * 80)
    ids = torch.tensor(encoder.encode(text), dtype=torch.long)
    ids = ids[: SEQ_LEN * 4].view(4, SEQ_LEN).to(run.device)
    opt = make_optimizer(model, LM_WD)
    start = None
    for step in range(10):
        opt.zero_grad(set_to_none=True)
        with autocast_for(run.device):
            logits = model(ids)
            loss = LOSSES["cross_entropy"](logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        if start is None:
            start = float(loss.detach())
        if not torch.isfinite(loss):
            print("SMOKE_FAIL", flush=True)
            return 1
        loss.backward()
        opt.step()
    brier = LOSSES["brier"](logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    if not torch.isfinite(brier):
        print("SMOKE_FAIL", flush=True)
        return 1
    if run.gate.opened:
        run.note("SMOKE opened the test split")
        print("SMOKE_FAIL", flush=True)
        return 1
    run.state["status"] = "SMOKE_PASS"
    run.state["phases"]["smoke"] = "DONE"
    run.save()
    print(f"SMOKE_PASS lm_ce {start:.4f} -> {float(loss):.4f} brier {float(brier):.4f}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--job-root", required=True)
    args = parser.parse_args()
    if args.mode == "full" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass
    if free_bytes(Path(args.job_root)) < 8 * 1024**3 and args.mode == "full":
        raise SystemExit("BLOCKED fewer than 8GB free on the job filesystem")
    run = Run(args)
    if args.mode == "smoke":
        code = run_smoke(run)
        raise SystemExit(code)
    run_full(run)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
