"""CIFAR-10 stability follow-up: loss by weight decay.

The rules are the ones in STABILITY_PROTOCOL.md. This file does not read
the WikiText run and does not change its checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

from losses import LOSSES, brier_explicit, brier_loss, collapsed_logit_check, probabilities
from metrics import classification_metrics, paired_seed_comparison
from models import AlexNetSmallImage, adamw_groups, count_parameters
from train import (
    BETAS,
    GRAD_CLIP,
    VISION_BATCH,
    TestGate,
    autocast_for,
    loader,
    lr_at,
    seed_everything,
    set_lr,
    vision_sets,
)

LOSSES_ORDER = ("cross_entropy", "brier")
WEIGHT_DECAYS = (0.0, 5e-4)
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
MAX_EPOCHS = 30
PATIENCE = 3
SEEDS = (0, 1, 2, 3, 4)
STALL_ACCURACY = 0.20
STALL_RELATIVE = 0.01
CONFIDENT = 0.99
MAP_RUNS = len(LOSSES_ORDER) * len(WEIGHT_DECAYS) * len(LEARNING_RATES)
EXTRA_FINALS = len(LOSSES_ORDER) * len(WEIGHT_DECAYS) * (len(SEEDS) - 1)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def require_one_gpu() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(f"expected one visible GPU, found {torch.cuda.device_count()}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def check_brier_gradient() -> None:
    logits = torch.randn(64, 10, requires_grad=True)
    y = torch.randint(0, 10, (64,))
    brier_explicit(logits, y).mean().backward()
    auto = logits.grad.detach().gather(1, y[:, None]).squeeze(1)
    q = probabilities(logits.detach())
    q_y = q.gather(1, y[:, None]).squeeze(1)
    others = q.square().sum(1) - q_y.square()
    formula = 2.0 * (q_y - 1.0) * q_y * (1.0 - q_y) - 2.0 * q_y * others
    # brier_explicit.mean divides by the batch, so the autograd row is the per-example gradient over 64.
    if not torch.allclose(auto * 64, formula, atol=1e-4):
        raise RuntimeError("Brier true-class logit gradient does not match the reference")


def true_class_grads(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = probabilities(logits)
    q_y = q.gather(1, y[:, None]).squeeze(1)
    pred = q.argmax(dim=-1)
    conf = q.max(dim=-1).values
    mask = (conf >= CONFIDENT) & pred.ne(y)
    others = q.square().sum(1) - q_y.square()
    brier_g = 2.0 * (q_y - 1.0) * q_y * (1.0 - q_y) - 2.0 * q_y * others
    return mask, q_y - 1.0, brier_g


def evaluate(model, dataset, device) -> dict:
    model.eval()
    chunks = []
    labels = []
    masks = []
    ce_grads = []
    brier_grads = []
    with torch.no_grad():
        for x, y in loader(dataset, 0, 0, VISION_BATCH, False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_for(device):
                logits = model(x)
            logits = logits.float()
            mask, ce_g, brier_g = true_class_grads(logits, y)
            chunks.append(logits.cpu())
            labels.append(y.cpu())
            masks.append(mask.cpu())
            ce_grads.append(ce_g.cpu())
            brier_grads.append(brier_g.cpu())
    metrics = classification_metrics(torch.cat(chunks), torch.cat(labels))
    mask = torch.cat(masks)
    count = int(mask.sum())
    metrics["confident_wrong_count"] = count
    if count == 0:
        metrics["confident_wrong_cross_entropy_grad"] = None
        metrics["confident_wrong_brier_grad"] = None
    else:
        metrics["confident_wrong_cross_entropy_grad"] = float(torch.cat(ce_grads)[mask].mean())
        metrics["confident_wrong_brier_grad"] = float(torch.cat(brier_grads)[mask].mean())
    return metrics


def objective_value(metrics: dict, loss_name: str) -> float:
    return float(metrics["cross_entropy" if loss_name == "cross_entropy" else "brier"])


def classify(history: list[dict], diverged: bool) -> str:
    if diverged or not history:
        return "divergence"
    best = min(history, key=lambda row: row["val_objective"])
    base = history[0]["val_objective"]
    relative = (base - best["val_objective"]) / max(abs(base), 1e-8)
    if best["accuracy"] < STALL_ACCURACY or relative < STALL_RELATIVE:
        return "stall"
    return "success"


def run_one(out: Path, device, train_set, val_set, init: dict, loss_name: str, wd: float, lr: float, seed: int, phase: str) -> dict:
    model = AlexNetSmallImage(3).to(device)
    model.load_state_dict(init)
    seed_everything(seed)
    opt = torch.optim.AdamW(adamw_groups(model, wd), lr=lr, betas=BETAS)
    steps_per_epoch = math.ceil(len(train_set) / VISION_BATCH)
    horizon = MAX_EPOCHS * steps_per_epoch
    warmup = min(100, max(1, horizon // 10))
    step = 0
    history = []
    diverged = False
    best_objective = float("inf")
    best_epoch = None
    best_state = None
    stale = 0
    started = time.time()
    params = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(MAX_EPOCHS):
        model.train()
        loss_sum = 0.0
        seen = 0
        grad_norms = []
        update_norms = []
        for x, y in loader(train_set, seed, epoch, VISION_BATCH, True):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            set_lr(opt, lr_at(step, horizon, warmup, lr))
            opt.zero_grad(set_to_none=True)
            with autocast_for(device):
                logits = model(x)
                loss = LOSSES[loss_name](logits, y)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            grade = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            if not torch.isfinite(grade):
                diverged = True
                break
            before = [p.detach().clone() for p in params]
            opt.step()
            with torch.no_grad():
                delta = torch.zeros((), device=device)
                for p, old in zip(params, before):
                    delta = delta + (p - old).square().sum()
                update_norms.append(float(torch.sqrt(delta)))
            grad_norms.append(float(grade))
            loss_sum += float(loss.detach()) * int(y.shape[0])
            seen += int(y.shape[0])
            step += 1
        if diverged:
            break
        metrics = evaluate(model, val_set, device)
        value = objective_value(metrics, loss_name)
        median = float(torch.tensor(grad_norms).median()) if grad_norms else float("nan")
        spikes = float(sum(g > 10.0 * median for g in grad_norms) / len(grad_norms)) if grad_norms else float("nan")
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, seen),
            "val_objective": value,
            "accuracy": metrics["accuracy"],
            "cross_entropy": metrics["cross_entropy"],
            "brier": metrics["brier"],
            "expected_calibration_error": metrics["expected_calibration_error"],
            "grad_norm_mean": sum(grad_norms) / len(grad_norms),
            "grad_norm_max": max(grad_norms),
            "grad_spike_fraction": spikes,
            "update_norm_mean": sum(update_norms) / len(update_norms),
            "update_norm_max": max(update_norms),
            "confident_wrong_count": metrics["confident_wrong_count"],
            "confident_wrong_cross_entropy_grad": metrics["confident_wrong_cross_entropy_grad"],
            "confident_wrong_brier_grad": metrics["confident_wrong_brier_grad"],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"{phase} {loss_name} wd {wd} seed {seed} lr {lr} epoch {epoch} "
            f"train {row['train_loss']:.4f} val {value:.4f} acc {metrics['accuracy']:.4f}",
            flush=True,
        )
        if value < best_objective:
            best_objective = value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    outcome = classify(history, diverged)
    record = {
        "dataset": "cifar10",
        "followup": "stability",
        "loss": loss_name,
        "weight_decay": wd,
        "learning_rate": lr,
        "seed": seed,
        "phase": phase,
        "outcome": outcome,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "steps": step,
        "seconds": time.time() - started,
        "n_params": count_parameters(model),
        "history": history,
        "diverged": diverged,
    }
    if best_state is not None and outcome != "divergence":
        path = out / "ckpt" / f"{phase}-{loss_name}-wd{wd}-seed{seed}-lr{lr}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, path)
        record["checkpoint"] = str(path)
    print(f"OUTCOME {outcome} {loss_name} wd {wd} seed {seed} lr {lr}", flush=True)
    return record


def select(records: list[dict]) -> dict:
    chosen = {}
    for loss_name in LOSSES_ORDER:
        for wd in WEIGHT_DECAYS:
            key = f"{loss_name}|{wd}"
            group = [
                r
                for r in records
                if r["phase"] == "tune" and r["loss"] == loss_name and r["weight_decay"] == wd and r["seed"] == 0
            ]
            successes = [r for r in group if r["outcome"] == "success" and r["history"]]
            pool = successes or [r for r in group if r["outcome"] != "divergence" and r["history"]]
            if not pool:
                chosen[key] = {"learning_rate": None, "label": "all learning rates diverged"}
                continue
            def best_row(record: dict) -> dict:
                return min(record["history"], key=lambda row: row["val_objective"])

            pool.sort(key=lambda r: (best_row(r)["val_objective"], -best_row(r)["accuracy"], r["learning_rate"]))
            chosen[key] = {
                "learning_rate": pool[0]["learning_rate"],
                "label": "success" if successes else "no successful learning rate",
                "outcome": pool[0]["outcome"],
            }
    return chosen


def score_test(device, test_set, record: dict) -> dict | None:
    path = record.get("checkpoint")
    if not path or not Path(path).exists():
        return None
    model = AlexNetSmallImage(3).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    return evaluate(model, test_set, device)


def render(out: Path, state: dict) -> None:
    lines = [
        "# CIFAR-10 stability follow-up",
        "",
        f"**Status:** {state.get('status')}",
        "",
        f"**Last updated:** {state.get('updated')}",
        "",
        "This is a follow-up. The 6-epoch test scores were already observed and were not used to choose these learning rates, the patience, or the 30-epoch cap.",
        "",
        "A lower validation loss is compared only inside one training loss. Cross-entropy and the Brier score are different scales.",
        "",
        "## Selection",
        "",
        "```",
        json.dumps(state.get("selection", {}), indent=2),
        "```",
        "",
        "## Map, seed 0",
        "",
        "| loss | weight decay | learning rate | outcome | best epoch | best validation objective | best accuracy | seconds |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in state["records"]:
        if r["phase"] != "tune":
            continue
        if r["history"]:
            best = min(r["history"], key=lambda row: row["val_objective"])
            best_v = f"{best['val_objective']:.6g}"
            best_a = f"{best['accuracy']:.6g}"
            best_e = best["epoch"]
        else:
            best_v = best_a = ""
            best_e = ""
        lines.append(
            f"| {r['loss']} | {r['weight_decay']} | {r['learning_rate']} | {r['outcome']} | {best_e} | {best_v} | {best_a} | {r['seconds']:.1f} |"
        )
    lines += ["", "## Held-out test at the selected learning rates", ""]
    finals = [r for r in state["records"] if r["phase"] == "final" and r.get("test")]
    if not finals:
        lines.append("No held-out rows yet.")
    else:
        lines.append("| loss | weight decay | seed | learning rate | accuracy | cross-entropy | Brier score | ECE | outcome |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for r in finals:
            m = r["test"]
            lines.append(
                f"| {r['loss']} | {r['weight_decay']} | {r['seed']} | {r['learning_rate']} | "
                f"{m['accuracy']:.6g} | {m['cross_entropy']:.6g} | {m['brier']:.6g} | "
                f"{m['expected_calibration_error']:.6g} | {r['outcome']} |"
            )
        lines.append("")
        lines.append("Paired differences use the training seed. With 5 seeds the smallest nonzero sign-flip p-value is 0.0625.")
        lines.append("")
        by = {}
        for r in finals:
            by.setdefault((r["loss"], r["weight_decay"]), {})[r["seed"]] = r["test"]
        lines.append("| comparison | metric | mean difference | 95% interval | p-value | n |")
        lines.append("| --- | --- | ---: | --- | ---: | ---: |")
        for loss_name in LOSSES_ORDER:
            left = by.get((loss_name, 5e-4), {})
            right = by.get((loss_name, 0.0), {})
            seeds = sorted(set(left) & set(right))
            for metric in ("accuracy", "cross_entropy", "brier", "expected_calibration_error"):
                if not seeds:
                    continue
                comp = paired_seed_comparison([left[s][metric] for s in seeds], [right[s][metric] for s in seeds])
                lo, hi = comp["ci95"]
                lines.append(
                    f"| {loss_name}, weight decay 5e-4 minus 0 | {metric} | {comp['mean_diff']:.6g} | "
                    f"[{lo:.6g}, {hi:.6g}] | {comp['p_value']:.4g} | {comp['n']} |"
                )
        for wd in WEIGHT_DECAYS:
            left = by.get(("cross_entropy", wd), {})
            right = by.get(("brier", wd), {})
            seeds = sorted(set(left) & set(right))
            for metric in ("accuracy", "cross_entropy", "brier", "expected_calibration_error"):
                if not seeds:
                    continue
                comp = paired_seed_comparison([left[s][metric] for s in seeds], [right[s][metric] for s in seeds])
                lo, hi = comp["ci95"]
                lines.append(
                    f"| weight decay {wd}, cross-entropy minus Brier | {metric} | {comp['mean_diff']:.6g} | "
                    f"[{lo:.6g}, {hi:.6g}] | {comp['p_value']:.4g} | {comp['n']} |"
                )
    lines += [
        "",
        "**TLDR-end:** [ultimate-utils: brier vs cross-entropy] CIFAR-10 stability follow-up. "
        f"Status {state.get('status')}. Four conditions, validation selection, held-out test after selection.",
        "",
    ]
    (out / "results.md").write_text("\n".join(lines))


def load_state(out: Path) -> dict:
    path = out / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"status": "RUNNING", "records": [], "markers": [], "selection": {}}


def save_state(out: Path, state: dict) -> None:
    state["updated"] = now_iso()
    render(out, state)
    atomic_json(out / "state.json", state)


def smoke(device, train_set) -> dict:
    check_brier_gradient()
    collapsed = collapsed_logit_check(device)
    x, y = next(iter(loader(train_set, 0, 0, VISION_BATCH, True)))
    x = x.to(device)
    y = y.to(device)
    times = []
    for loss_name in LOSSES_ORDER:
        for wd in WEIGHT_DECAYS:
            model = AlexNetSmallImage(3).to(device)
            opt = torch.optim.AdamW(adamw_groups(model, wd), lr=1e-3, betas=BETAS)
            for step in range(30):
                started = time.time()
                opt.zero_grad(set_to_none=True)
                with autocast_for(device):
                    loss = LOSSES[loss_name](model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"smoke loss not finite for {loss_name} wd {wd}")
                loss.backward()
                grade = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                before = [p.detach().clone() for p in model.parameters()]
                opt.step()
                delta = torch.zeros((), device=device)
                for p, old in zip(model.parameters(), before):
                    delta = delta + (p - old).square().sum()
                if not torch.isfinite(grade) or not torch.isfinite(delta):
                    raise RuntimeError("smoke gradient or update is not finite")
                if step >= 5:
                    times.append(time.time() - started)
            del model
            torch.cuda.empty_cache()
    step_seconds = sum(times) / len(times)
    steps_per_epoch = math.ceil(len(train_set) / VISION_BATCH)
    epoch_seconds = step_seconds * steps_per_epoch
    runs = MAP_RUNS + EXTRA_FINALS
    watchdog = epoch_seconds * 2.0 * MAX_EPOCHS * runs + 3600
    print(
        f"SMOKE_PASS step_seconds {step_seconds:.4f} epoch_seconds {epoch_seconds:.1f} "
        f"watchdog_hours {watchdog / 3600:.2f}",
        flush=True,
    )
    return {
        "step_seconds": step_seconds,
        "steps_per_epoch": steps_per_epoch,
        "epoch_seconds": epoch_seconds,
        "watchdog_seconds": watchdog,
        "collapsed_bias_gradient": {name: row["bias_gradient"] for name, row in collapsed.items()},
    }


def write_protocol(out: Path, smoke_info: dict) -> dict:
    payload = {
        "study": "cifar10 stability follow-up",
        "losses": list(LOSSES_ORDER),
        "weight_decays": list(WEIGHT_DECAYS),
        "learning_rates": list(LEARNING_RATES),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seeds": list(SEEDS),
        "stall_accuracy": STALL_ACCURACY,
        "stall_relative_improvement": STALL_RELATIVE,
        "confident_threshold": CONFIDENT,
        "schedule": "cosine to 0.1 times the peak over a 30-epoch horizon, warmup min(100, horizon/10)",
        "batch": VISION_BATCH,
        "optimizer": "AdamW",
        "betas": list(BETAS),
        "grad_clip": GRAD_CLIP,
        "precision": "bfloat16 autocast, loss float32",
        **smoke_info,
    }
    atomic_json(out / "protocol.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "train", "all"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = require_one_gpu()
    data = Path(args.data)
    gate = TestGate()
    _in_ch, train_set, val_set, _test, _split = vision_sets("cifar10", data, gate, need_test=False)
    if len(train_set) != 45000 or len(val_set) != 5000:
        raise SystemExit(f"unexpected split sizes {len(train_set)} {len(val_set)}")
    if args.stage in ("smoke", "all") and not (out / "protocol.json").exists():
        info = smoke(device, train_set)
        write_protocol(out, info)
    if args.stage == "smoke":
        return
    protocol_path = out / "protocol.json"
    if not protocol_path.exists():
        raise SystemExit("protocol.json is missing; refusing to invent the stability rules")
    state = load_state(out)
    state["protocol"] = json.loads(protocol_path.read_text())
    state["status"] = "RUNNING"
    inits = {}
    for seed in SEEDS:
        path = out / "inits" / f"seed{seed}.pt"
        if path.exists():
            inits[seed] = torch.load(path, weights_only=True)
        else:
            seed_everything(seed)
            model = AlexNetSmallImage(3)
            inits[seed] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(inits[seed], path)
    for loss_name in LOSSES_ORDER:
        for wd in WEIGHT_DECAYS:
            for lr in LEARNING_RATES:
                key = f"tune-{loss_name}-wd{wd}-seed0-lr{lr}"
                if key in state["markers"]:
                    continue
                record = run_one(out, device, train_set, val_set, inits[0], loss_name, wd, lr, 0, "tune")
                state["records"].append(record)
                state["markers"].append(key)
                save_state(out, state)
    if not state.get("selection"):
        state["selection"] = select(state["records"])
        save_state(out, state)
        print(f"SELECTION {json.dumps(state['selection'])}", flush=True)
        for record in state["records"]:
            if record["phase"] != "tune" or not record.get("checkpoint"):
                continue
            chosen_lr = state["selection"][f"{record['loss']}|{record['weight_decay']}"]["learning_rate"]
            if chosen_lr is None or record["learning_rate"] != chosen_lr:
                Path(record["checkpoint"]).unlink(missing_ok=True)
                record["checkpoint"] = None
        save_state(out, state)
    for loss_name in LOSSES_ORDER:
        for wd in WEIGHT_DECAYS:
            choice = state["selection"][f"{loss_name}|{wd}"]
            lr = choice["learning_rate"]
            if lr is None:
                continue
            for seed in SEEDS:
                key = f"final-{loss_name}-wd{wd}-seed{seed}-lr{lr}"
                if key in state["markers"]:
                    continue
                if seed == 0:
                    tune_key = f"tune-{loss_name}-wd{wd}-seed0-lr{lr}"
                    source = next(r for r in state["records"] if f"tune-{r['loss']}-wd{r['weight_decay']}-seed0-lr{r['learning_rate']}" == tune_key)
                    record = dict(source)
                    record["phase"] = "final"
                    record["reused_tune"] = True
                else:
                    record = run_one(out, device, train_set, val_set, inits[seed], loss_name, wd, lr, seed, "final")
                state["records"].append(record)
                state["markers"].append(key)
                save_state(out, state)
    gate.allow("cifar10")
    _in_ch, _train, _val, test_set, _split = vision_sets("cifar10", data, gate, need_test=True)
    for record in state["records"]:
        if record["phase"] != "final" or record.get("test"):
            continue
        record["test"] = score_test(device, test_set, record)
    missing = [
        r
        for r in state["records"]
        if r["phase"] == "final" and r["outcome"] != "divergence" and not r.get("test")
    ]
    state["status"] = "PARTIAL" if missing else "DONE"
    save_state(out, state)
    print(f"STABILITY_DONE {state['status']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
