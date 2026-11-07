"""Fixed-budget Brier score versus cross-entropy pretraining.

The AlexNet results under full/ are not touched. This process trains only the
123,551,232-parameter decoder-only Transformer. The token count is written to
protocol.json before any final step. A restart keeps that count.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

import torch

from data_util import (
    add_one_unigram,
    encode_file,
    pack_sequences,
    parquet_lines_to_text,
    prepare_wikitext103_hf,
)
from lm_budget import TUNE_TOKENS, choose_final_tokens
from losses import LOSSES, brier_explicit, brier_loss, collapsed_logit_check, probabilities
from metrics import paired_seed_comparison
from models import DenseCausalLM, adamw_groups, count_parameters

SEQ_LEN = 256
LM_WD = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
SMOKE_STEPS = 40
EXPECTED_PARAMS = 123_551_232


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * float(step + 1) / float(max(1, warmup))
    progress = min(1.0, (step - warmup) / float(max(1, total - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (0.1 + 0.9 * cosine)


def require_one_gpu() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("CUDA_VISIBLE_DEVICES did not pin exactly one GPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def check_objectives(device: torch.device) -> dict:
    logits = torch.tensor([[0.2, -0.4, 1.1, 0.0], [2.0, 2.0, 2.0, 2.0]], device=device)
    y = torch.tensor([2, 0], device=device)
    fast = brier_loss(logits, y)
    explicit = brier_explicit(logits, y)
    if not torch.allclose(fast, explicit, atol=1e-6):
        raise RuntimeError(f"Brier identity failed: {float(fast)} vs {float(explicit)}")
    rows = collapsed_logit_check(device)
    return {"brier_identity_ok": True, "collapsed": rows}


def build_model(vocab: int, device: torch.device) -> DenseCausalLM:
    model = DenseCausalLM(vocab=vocab, d=768, n_layer=12, n_head=12, seq_len=SEQ_LEN).to(device)
    n_params = count_parameters(model)
    if n_params != EXPECTED_PARAMS:
        raise RuntimeError(f"parameter count {n_params}, expected {EXPECTED_PARAMS}")
    model.tie_weights()
    return model


def make_opt(model: DenseCausalLM) -> torch.optim.Optimizer:
    return torch.optim.AdamW(adamw_groups(model, LM_WD), lr=1e-4, betas=BETAS)


def batch_at(packed: torch.Tensor, perm: torch.Tensor, step: int, batch: int) -> torch.Tensor:
    n_seq = perm.shape[0]
    start = (step * batch) % n_seq
    positions = torch.arange(start, start + batch) % n_seq
    return packed[perm[positions]].long()


def forward_loss(model, idx: torch.Tensor, loss_name: str) -> torch.Tensor:
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(idx)
    return LOSSES[loss_name](logits[:, :-1].reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1)), logits


def evaluate(model, packed: torch.Tensor, batch: int, device: torch.device) -> dict:
    model.eval()
    n = 0
    ce_sum = brier_sum = local_sum = acc_sum = 0.0
    confs = []
    corrects = []
    with torch.no_grad():
        for start in range(0, packed.shape[0], batch):
            idx = packed[start : start + batch].long().to(device)
            if idx.shape[0] == 0:
                continue
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(idx)
            pred_logits = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
            y = idx[:, 1:].reshape(-1)
            q = probabilities(pred_logits)
            q_y = q.gather(-1, y[:, None]).squeeze(-1)
            count = int(y.numel())
            n += count
            ce_sum += float((-q_y.clamp_min(1e-12).log()).sum())
            brier_sum += float((q.square().sum(-1) + 1.0 - 2.0 * q_y).sum())
            local_sum += float((1.0 - q_y).square().sum())
            correct = pred_logits.argmax(-1).eq(y)
            acc_sum += float(correct.float().sum())
            confs.append(q.max(-1).values.cpu())
            corrects.append(correct.cpu())
    if n == 0:
        raise RuntimeError("evaluation saw no tokens")
    ce = ce_sum / n
    conf = torch.cat(confs)
    correct = torch.cat(corrects).float()
    ece, bins = _ece_bins(conf, correct)
    saturated = float((conf >= 0.99).float().mean())
    return {
        "n": n,
        "accuracy": acc_sum / n,
        "cross_entropy": ce,
        "perplexity": math.exp(ce) if ce < 50 else None,
        "brier": brier_sum / n,
        "local_square": local_sum / n,
        "expected_calibration_error": ece,
        "calibration_bins": bins,
        "mean_max_probability": float(conf.mean()),
        "fraction_max_probability_at_least_0.99": saturated,
    }


def _ece(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    return _ece_bins(conf, correct, n_bins)[0]


def _ece_bins(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> tuple[float, list[dict]]:
    """Top-label expected calibration error, 15 equal-width bins.

    Bin 0 is closed on both ends. Later bins are open on the left and closed
    on the right, so a confidence exactly on an interior edge stays in the
    lower bin. Empty bins are stored and then skipped in the sum.
    """
    total = conf.numel()
    edges = torch.linspace(0, 1, n_bins + 1)
    total_gap = 0.0
    bins = []
    for i in range(n_bins):
        if i == 0:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf > edges[i]) & (conf <= edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            acc = conf_mean = gap = None
        else:
            acc = float(correct[mask].mean())
            conf_mean = float(conf[mask].mean())
            gap = abs(acc - conf_mean)
            total_gap += gap * (count / total)
        bins.append(
            {
                "bin": i,
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "count": count,
                "accuracy": acc,
                "mean_confidence": conf_mean,
                "gap": gap,
                "weight": (count / total) if total else 0.0,
            }
        )
    return total_gap, bins


def load_or_encode(data_root: Path, out: Path) -> dict:
    info = prepare_wikitext103_hf(data_root)
    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    if encoder.n_vocab != 50257:
        raise RuntimeError(f"gpt2 vocabulary is {encoder.n_vocab}")
    cache = out / "tokens_train_valid.pt"
    meta_path = out / "tokens_meta.json"
    if cache.exists() and meta_path.exists():
        tensors = torch.load(cache, weights_only=True)
        meta = json.loads(meta_path.read_text())
    else:
        train_text, train_rows = parquet_lines_to_text(info["files"]["train"])
        valid_text, valid_rows = parquet_lines_to_text(info["files"]["validation"])
        if train_rows != info["expected_rows"]["train"] or valid_rows != info["expected_rows"]["validation"]:
            raise RuntimeError(f"row counts train {train_rows} valid {valid_rows}")
        train_ids = torch.tensor(encoder.encode(train_text), dtype=torch.int32)
        valid_ids = torch.tensor(encoder.encode(valid_text), dtype=torch.int32)
        del train_text, valid_text
        tensors = {
            "train": pack_sequences(train_ids, SEQ_LEN),
            "valid": pack_sequences(valid_ids, SEQ_LEN),
        }
        meta = {
            "vocab": encoder.n_vocab,
            "tokenizer": "tiktoken gpt2",
            "info": {k: info[k] for k in ("name", "source", "config", "license", "paper", "bytes")},
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "n_train_tokens": int(train_ids.numel()),
            "n_valid_tokens": int(valid_ids.numel()),
            "test_decoded": False,
        }
        torch.save(tensors, cache)
        atomic_json(meta_path, meta)
    return {"tensors": tensors, "meta": meta, "info": info}


def encode_test(data_root: Path, out: Path, info: dict) -> torch.Tensor:
    import tiktoken

    test_path = out / "tokens_test.pt"
    if test_path.exists():
        return torch.load(test_path, weights_only=True)["test"]
    encoder = tiktoken.get_encoding("gpt2")
    text, rows = parquet_lines_to_text(info["files"]["test"])
    if rows != info["expected_rows"]["test"]:
        raise RuntimeError(f"test rows {rows}")
    ids = torch.tensor(encoder.encode(text), dtype=torch.int32)
    packed = pack_sequences(ids, SEQ_LEN)
    torch.save({"test": packed}, test_path)
    meta_path = out / "tokens_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["test_decoded"] = True
    meta["test_rows"] = rows
    meta["n_test_tokens"] = int(ids.numel())
    atomic_json(meta_path, meta)
    print(f"decoded held-out test rows {rows} tokens {int(ids.numel())}", flush=True)
    return packed


def smoke(device, tensors, batch: int) -> dict:
    vocab = 50257
    perm = torch.randperm(tensors["train"].shape[0], generator=torch.Generator().manual_seed(0))
    rates = {}
    for loss_name in ("cross_entropy", "brier"):
        model = build_model(vocab, device)
        opt = make_opt(model)
        # One shared initialization for the timing pair.
        if loss_name == "cross_entropy":
            init = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(init)
            model.tie_weights()
        times = []
        for step in range(SMOKE_STEPS):
            idx = batch_at(tensors["train"], perm, step, batch).to(device)
            opt.zero_grad(set_to_none=True)
            started = time.perf_counter()
            loss, _logits = forward_loss(model, idx, loss_name)
            if not torch.isfinite(loss):
                raise RuntimeError(f"smoke loss was non-finite for {loss_name}")
            loss.backward()
            grade = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if not torch.isfinite(grade):
                raise RuntimeError(f"smoke gradient was non-finite for {loss_name}")
            opt.step()
            if step > 0:
                times.append(time.perf_counter() - started)
            print(f"smoke {loss_name} step {step} loss {float(loss):.4f} grad {float(grade):.4f}", flush=True)
        tokens = batch * (SEQ_LEN - 1)
        sec = sum(times) / len(times)
        rates[loss_name] = {"seconds_per_step": sec, "tokens_per_sec": tokens / sec, "grad_norm": float(grade)}
        del model, opt
        torch.cuda.empty_cache()
    slower = min(rates["cross_entropy"]["tokens_per_sec"], rates["brier"]["tokens_per_sec"])
    return {"per_loss": rates, "slower_tokens_per_sec": slower, "batch": batch}


def epoch_tokens(n_seq: int, batch: int) -> int:
    steps = n_seq // batch
    return steps * batch * (SEQ_LEN - 1)


def write_protocol(out: Path, decision: dict, smoke_info: dict, meta: dict, batch: int) -> dict:
    tokens_per_step = batch * (SEQ_LEN - 1)
    final_steps = decision["final_tokens_per_run"] // tokens_per_step
    tune_steps = TUNE_TOKENS // tokens_per_step
    payload = dict(decision)
    payload.update(
        {
            "batch": batch,
            "seq_len": SEQ_LEN,
            "tokens_per_step": tokens_per_step,
            "final_steps": final_steps,
            "tune_steps": tune_steps,
            "final_tokens_realized": final_steps * tokens_per_step,
            "tune_tokens_realized": tune_steps * tokens_per_step,
            "corpus": meta["info"]["name"] if "info" in meta else meta.get("info"),
            "n_train_tokens": meta["n_train_tokens"],
            "n_valid_tokens": meta["n_valid_tokens"],
            "tokenizer": meta["tokenizer"],
            "parameters": EXPECTED_PARAMS,
            "smoke": smoke_info,
            "written": now_iso(),
        }
    )
    # The stored corpus field above is messy if meta shape varies. Set it cleanly.
    payload["corpus"] = meta.get("info", meta).get("name", "Salesforce/wikitext wikitext-103-raw-v1")
    atomic_json(out / "protocol.json", payload)
    print(
        f"PROTOCOL final_tokens {payload['final_tokens_realized']} steps {final_steps} "
        f"tune_steps {tune_steps} projected_hours {payload['projected_seconds']/3600:.2f} "
        f"watchdog_hours {payload['watchdog_seconds']/3600:.2f} rule {payload['rule']}",
        flush=True,
    )
    return payload


def ckpt_path(out: Path) -> Path:
    return out / "ckpt" / "current.pt"


def _save_ckpt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    path.with_suffix(".key").write_text(payload["key"] + "\n")


def _drop_ckpt(out: Path, key: str) -> None:
    path = ckpt_path(out)
    label = path.with_suffix(".key")
    if label.exists() and label.read_text().strip() == key and path.exists():
        path.unlink()
        label.unlink()


def train_one(out, device, packed_train, packed_val, packed_test, loss_name, lr, seed, steps, batch, phase) -> dict:
    key = f"{phase}-{loss_name}-seed{seed}-lr{lr}"
    path = ckpt_path(out)
    torch.manual_seed(seed)
    model = build_model(50257, device)
    resumed = False
    if path.exists():
        blob = torch.load(path, weights_only=False)
        if blob.get("key") == key:
            model.load_state_dict(blob["model"])
            model.tie_weights()
            opt = make_opt(model)
            opt.load_state_dict(blob["opt"])
            start_step = int(blob["step"])
            curve = blob.get("curve", [])
            resumed = True
        else:
            path.unlink()
            key_path = path.with_suffix(".key")
            if key_path.exists():
                key_path.unlink()
    if not resumed:
        shared = out / "ckpt" / f"init_seed{seed}.pt"
        shared.parent.mkdir(parents=True, exist_ok=True)
        if shared.exists():
            model.load_state_dict(torch.load(shared, weights_only=True))
            model.tie_weights()
        else:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, shared)
        opt = make_opt(model)
        curve = []
        start_step = 0
    perm = torch.randperm(packed_train.shape[0], generator=torch.Generator().manual_seed(seed))
    warmup = max(20, min(steps // 20, 500))
    invalid = False
    oom = False
    reason = ""
    model.train()
    step = start_step
    while step < steps:
        idx = batch_at(packed_train, perm, step, batch).to(device)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps, warmup, lr)
        opt.zero_grad(set_to_none=True)
        try:
            loss, _logits = forward_loss(model, idx, loss_name)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            invalid = True
            oom = True
            reason = "out of memory"
            break
        if not torch.isfinite(loss):
            invalid = True
            reason = "non-finite loss"
            break
        loss.backward()
        grade = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not torch.isfinite(grade):
            invalid = True
            reason = "non-finite gradient"
            break
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            row = {"step": step, "train_loss": float(loss.detach()), "grad_norm": float(grade)}
            curve.append(row)
            print(f"{phase} {loss_name} seed {seed} lr {lr} step {step} loss {float(loss):.4f} grad {float(grade):.4f}", flush=True)
        if step % 200 == 0:
            _save_ckpt(
                path,
                {"key": key, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "opt": opt.state_dict(), "step": step + 1, "curve": curve[-400:]},
            )
        step += 1
    if not invalid and step >= steps:
        _save_ckpt(
            path,
            {
                "key": key,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "opt": opt.state_dict(),
                "step": step,
                "curve": curve[-400:],
            },
        )
    val = None if invalid else evaluate(model, packed_val, batch, device)
    test = None
    if packed_test is not None and not invalid and step >= steps:
        test = evaluate(model, packed_test, batch, device)
    record = {
        "key": key,
        "dataset": "wikitext103",
        "loss": loss_name,
        "seed": seed,
        "learning_rate": lr,
        "phase": phase,
        "steps": step,
        "planned_steps": steps,
        "tokens": step * batch * (SEQ_LEN - 1),
        "invalid": invalid,
        "oom": oom,
        "reason": reason,
        "incomplete": step < steps,
        "curve": curve[-400:],
        "metrics": val,
        "split": "validation",
        "parameters": EXPECTED_PARAMS,
        "batch": batch,
    }
    test_record = None
    if test is not None:
        test_record = dict(record)
        test_record["metrics"] = test
        test_record["split"] = "test"
        test_record["curve"] = []
    return record, test_record


def select_lr(records: list[dict]) -> dict:
    chosen = {}
    for loss_name in ("cross_entropy", "brier"):
        trials = [
            r
            for r in records
            if r["phase"] == "tune"
            and r["loss"] == loss_name
            and r["split"] == "validation"
            and not r["invalid"]
            and not r["incomplete"]
            and r.get("metrics")
        ]
        if not trials:
            raise RuntimeError(f"no valid tune trial for {loss_name}")
        metric = "cross_entropy" if loss_name == "cross_entropy" else "brier"
        trials.sort(key=lambda r: (r["metrics"][metric], -r["metrics"]["accuracy"], r["learning_rate"]))
        chosen[loss_name] = trials[0]["learning_rate"]
    return chosen


def render(out: Path, state: dict) -> None:
    lines = [
        "# WikiText-103 raw: Brier score versus cross-entropy",
        "",
        f"**Status:** {state.get('status')}",
        "",
        f"**Last updated:** {state.get('updated')}",
        "",
        "The AlexNet tables are in the earlier `full/` run. This file is the language-model run only.",
        "",
        "## Protocol",
        "",
        "```",
        json.dumps(state.get("protocol", {}), indent=2)[:4000],
        "```",
        "",
        "## Validation and test",
        "",
    ]
    for split in ("validation", "test"):
        rows = [r for r in state["records"] if r["split"] == split and r["phase"] == "final"]
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| loss | seed | learning rate | steps | tokens | accuracy | cross-entropy | perplexity | Brier | ECE | invalid |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for r in rows:
            m = r.get("metrics") or {}
            lines.append(
                "| {loss} | {seed} | {lr} | {steps} | {tokens} | {acc} | {ce} | {ppl} | {br} | {ece} | {inv} |".format(
                    loss=r["loss"],
                    seed=r["seed"],
                    lr=r["learning_rate"],
                    steps=r["steps"],
                    tokens=r["tokens"],
                    acc=_fmt(m.get("accuracy")),
                    ce=_fmt(m.get("cross_entropy")),
                    ppl=_fmt(m.get("perplexity")),
                    br=_fmt(m.get("brier")),
                    ece=_fmt(m.get("expected_calibration_error")),
                    inv=r["invalid"],
                )
            )
        lines.append("")
    finals = [r for r in state["records"] if r["phase"] == "final" and r["split"] == "test" and not r["invalid"]]
    by = {"cross_entropy": {}, "brier": {}}
    for r in finals:
        by[r["loss"]][r["seed"]] = r["metrics"]
    seeds = sorted(set(by["cross_entropy"]) & set(by["brier"]))
    if seeds:
        lines.append("Paired difference is cross-entropy-trained minus Brier-trained. The interval resamples seeds. With 3 seeds the sign-flip p-value cannot resolve 0.05.")
        lines.append("")
        lines.append("| metric | mean difference | 95% interval | p-value | n |")
        lines.append("| --- | ---: | --- | ---: | ---: |")
        for metric in ("accuracy", "cross_entropy", "perplexity", "brier", "expected_calibration_error"):
            a = [by["cross_entropy"][s][metric] for s in seeds]
            b = [by["brier"][s][metric] for s in seeds]
            comp = paired_seed_comparison(a, b)
            lo, hi = comp["ci95"]
            lines.append(
                f"| {metric} | {comp['mean_diff']:.6g} | [{lo:.6g}, {hi:.6g}] | {comp['p_value']:.4g} | {comp['n']} |"
            )
        lines.append("")
    n_invalid = sum(1 for r in state["records"] if r["invalid"])
    lines.append(f"Invalid runs: {n_invalid}.")
    lines.append("")
    lines.append(
        "ECE is top-label expected calibration error: 15 equal-width bins of the "
        "maximum softmax probability, first bin closed on both ends, later bins "
        "open on the left and closed on the right, empty bins omitted from the sum. "
        "It is not classwise and not equal-count. A lower ECE is not, by itself, "
        "better accuracy, cross-entropy, or full Brier score. Bin rows, when "
        "evaluation ran, are under metrics.calibration_bins in state.json."
    )
    lines.append("")
    lines.append(
        "**TLDR-end:** [ultimate-utils: brier vs cross-entropy] Language-model status "
        f"{state.get('status')}. WikiText-103 raw, 123551232-parameter Transformer, matched token budget."
    )
    lines.append("")
    (out / "results.md").write_text("\n".join(lines))


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def load_state(out: Path) -> dict:
    path = out / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"status": "RUNNING", "records": [], "markers": [], "selection": {}}


def save_state(out: Path, state: dict) -> None:
    state["updated"] = now_iso()
    render(out, state)
    atomic_json(out / "state.json", state)


def run_prepare(out: Path, data_root: Path, device) -> None:
    if (out / "protocol.json").exists():
        print("PROTOCOL already stored", flush=True)
        return
    checks = check_objectives(device)
    print(f"objective checks {json.dumps(checks['collapsed'])}", flush=True)
    loaded = load_or_encode(data_root, out)
    tensors = loaded["tensors"]
    meta = loaded["meta"]
    batch = 8
    try:
        smoke_info = smoke(device, tensors, batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        batch = 4
        print("smoke batch reduced from 8 to 4 for both losses", flush=True)
        smoke_info = smoke(device, tensors, batch)
    n_seq = tensors["train"].shape[0]
    decision = choose_final_tokens(epoch_tokens(n_seq, batch), smoke_info["slower_tokens_per_sec"])
    decision["objective_checks"] = {"brier_identity_ok": True}
    write_protocol(out, decision, smoke_info, meta, batch)
    p = add_one_unigram(tensors["train"].view(-1), 50257)
    val_tokens = tensors["valid"][:, 1:].reshape(-1).long()
    ce = float((-p[val_tokens].clamp_min(1e-12).log()).mean())
    atomic_json(out / "unigram.json", {"validation_cross_entropy": ce, "validation_perplexity": math.exp(ce)})
    print(f"unigram validation cross-entropy {ce:.4f}", flush=True)


def run_train(out: Path, data_root: Path, device) -> None:
    protocol_path = out / "protocol.json"
    if not protocol_path.exists():
        raise SystemExit("protocol.json is missing; refusing to invent a token budget")
    protocol = json.loads(protocol_path.read_text())
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    loaded = load_or_encode(data_root, out)
    tensors = loaded["tensors"]
    info = loaded["info"]
    state = load_state(out)
    state["protocol"] = protocol
    state["status"] = "RUNNING"
    batch = int(protocol["batch"])
    tune_steps = int(protocol["tune_steps"])
    final_steps = int(protocol["final_steps"])
    # Shared tune initialization, seed 0.
    shared = out / "ckpt" / "init_seed0.pt"
    if not shared.exists():
        torch.manual_seed(0)
        model = build_model(50257, device)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, shared)
        del model
        torch.cuda.empty_cache()
    for loss_name in ("cross_entropy", "brier"):
        for lr in protocol["learning_rates"]:
            key = f"tune-{loss_name}-seed0-lr{lr}"
            if key in state["markers"]:
                continue
            record, _test = train_one(
                out, device, tensors["train"], tensors["valid"], None, loss_name, lr, 0, tune_steps, batch, "tune"
            )
            state["records"].append(record)
            state["markers"].append(key)
            save_state(out, state)
            _drop_ckpt(out, key)
    if "selection" not in state or not state["selection"]:
        state["selection"] = select_lr(state["records"])
        save_state(out, state)
        print(f"SELECTION {state['selection']}", flush=True)
    test_packed = encode_test(data_root, out, info)
    for seed in range(int(protocol["seeds"])):
        shared = out / "ckpt" / f"init_seed{seed}.pt"
        if not shared.exists():
            torch.manual_seed(seed)
            model = build_model(50257, device)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, shared)
            del model
            torch.cuda.empty_cache()
        for loss_name in ("cross_entropy", "brier"):
            lr = state["selection"][loss_name]
            key = f"final-{loss_name}-seed{seed}-lr{lr}"
            if key in state["markers"]:
                continue
            record, test_record = train_one(
                out, device, tensors["train"], tensors["valid"], test_packed, loss_name, lr, seed, final_steps, batch, "final"
            )
            state["records"].append(record)
            if test_record is not None:
                state["records"].append(test_record)
            state["markers"].append(key)
            save_state(out, state)
            _drop_ckpt(out, key)
    unfinished = [
        r for r in state["records"] if r["phase"] == "final" and r["split"] == "validation" and r["incomplete"]
    ]
    state["status"] = "PARTIAL" if unfinished else "DONE"
    save_state(out, state)
    print(f"LM_DONE {state['status']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "train"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = require_one_gpu()
    if args.stage == "prepare":
        run_prepare(out, Path(args.data), device)
    else:
        run_train(out, Path(args.data), device)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
