"""FineWeb-Edu replication of the WikiText-103 token budget.

The final budget is 118,032,360 tokens, the realized WikiText count. This
process does not read the WikiText directory and does not run on that disk.
Held-out metrics are computed only after the learning-rate selection is stored.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

from data_util import pack_sequences
from fineweb_data import collect_tokens
from lm_budget import LEARNING_RATES, N_SEEDS, TUNE_TOKENS, program_seconds
from train_lm_fixed import (
    EXPECTED_PARAMS,
    SEQ_LEN,
    atomic_json,
    build_model,
    check_objectives,
    load_state,
    now_iso,
    require_one_gpu,
    _drop_ckpt,
    select_lr,
    smoke,
    train_one,
)

FINAL_TOKENS = 118_032_360
HELD_TOKENS = 2_000_000
BATCH = 8


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render(out: Path, state: dict) -> None:
    lines = [
        "# FineWeb-Edu: Brier score versus cross-entropy",
        "",
        f"**Status:** {state.get('status')}",
        "",
        f"**Last updated:** {state.get('updated')}",
        "",
        "Matched to the WikiText-103 run: 118,032,360 final tokens, AdamW, weight decay 0.1, normal initialization.",
        "The held-out split is scored after learning-rate selection.",
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
        for row in rows:
            metrics = row.get("metrics") or {}
            lines.append(
                "| {loss} | {seed} | {lr} | {steps} | {tokens} | {acc} | {ce} | {ppl} | {br} | {ece} | {inv} |".format(
                    loss=row["loss"],
                    seed=row["seed"],
                    lr=row["learning_rate"],
                    steps=row["steps"],
                    tokens=row["tokens"],
                    acc=_fmt(metrics.get("accuracy")),
                    ce=_fmt(metrics.get("cross_entropy")),
                    ppl=_fmt(metrics.get("perplexity")),
                    br=_fmt(metrics.get("brier")),
                    ece=_fmt(metrics.get("expected_calibration_error")),
                    inv=row["invalid"],
                )
            )
        lines.append("")
    lines.append(f"Invalid runs: {sum(1 for row in state['records'] if row['invalid'])}.")
    lines.append("")
    (out / "results.md").write_text("\n".join(lines))


def save_state(out: Path, state: dict) -> None:
    state["updated"] = now_iso()
    render(out, state)
    atomic_json(out / "state.json", state)


def _materialize(collected: dict) -> dict:
    tensors = {}
    counts = {}
    for name, buf in collected["buffers"].items():
        ids = torch.tensor(buf, dtype=torch.int32)
        packed = pack_sequences(ids, SEQ_LEN)
        tensors[name] = packed
        counts[name] = {"raw": int(ids.numel()), "packed": int(packed.numel()), "sequences": int(packed.shape[0])}
    return {"tensors": tensors, "token_counts": counts}


def load_or_prepare(data_root: Path, out: Path) -> dict:
    cache = out / "tokens_train_valid.pt"
    meta_path = out / "tokens_meta.json"
    if cache.exists() and meta_path.exists() and (out / "tokens_test.pt").exists():
        tensors = torch.load(cache, weights_only=True)
        test = torch.load(out / "tokens_test.pt", weights_only=True)
        tensors["test"] = test["test"]
        return {"tensors": tensors, "meta": json.loads(meta_path.read_text())}
    steps = FINAL_TOKENS // (BATCH * (SEQ_LEN - 1))
    if steps * BATCH * (SEQ_LEN - 1) != FINAL_TOKENS:
        raise RuntimeError("final token count is not divisible by the batch")
    min_train = steps * BATCH * SEQ_LEN
    collected = collect_tokens(data_root, min_train, HELD_TOKENS, SEQ_LEN)
    made = _materialize(collected)
    meta = {
        "name": "HuggingFaceFW/fineweb-edu sample-10BT",
        "revision": collected["revision"],
        "license": "odc-by",
        "tokenizer": collected["tokenizer"],
        "vocab": collected["vocab"],
        "id_sha256": collected["id_sha256"],
        "counts": collected["counts"],
        "shards": collected["shards"],
        "token_counts": made["token_counts"],
        "targets": collected["targets"],
        "test_stored_during_split": True,
        "test_evaluated_after_selection": True,
    }
    torch.save({"train": made["tensors"]["train"], "valid": made["tensors"]["valid"]}, cache)
    torch.save({"test": made["tensors"]["test"]}, out / "tokens_test.pt")
    atomic_json(meta_path, meta)
    print(f"ID_SHA256 {meta['id_sha256']} tokens {json.dumps(meta['token_counts'])}", flush=True)
    return {"tensors": made["tensors"], "meta": meta}


def write_protocol(out: Path, smoke_info: dict, meta: dict) -> dict:
    tokens_per_step = BATCH * (SEQ_LEN - 1)
    final_steps = FINAL_TOKENS // tokens_per_step
    tune_steps = TUNE_TOKENS // tokens_per_step
    rate = smoke_info["slower_tokens_per_sec"]
    projected = program_seconds(FINAL_TOKENS, rate)
    payload = {
        "study": "fineweb-edu matched token replication",
        "corpus": meta["name"],
        "revision": meta["revision"],
        "license": meta["license"],
        "id_sha256": meta["id_sha256"],
        "final_tokens_per_run": FINAL_TOKENS,
        "tune_tokens_per_trial": TUNE_TOKENS,
        "learning_rates": list(LEARNING_RATES),
        "seeds": N_SEEDS,
        "batch": BATCH,
        "seq_len": SEQ_LEN,
        "tokens_per_step": tokens_per_step,
        "final_steps": final_steps,
        "tune_steps": tune_steps,
        "parameters": EXPECTED_PARAMS,
        "optimizer": "torch.optim.AdamW",
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "weight_decay_excluded": "ndim < 2",
        "initialization": "embedding normal std 0.02, no output bias, no collapsed coordinate",
        "tokens_per_sec": rate,
        "projected_seconds": projected,
        "watchdog_seconds": projected * 1.5 + 3600,
        "rule": "same 118032360 final tokens as the WikiText-103 run",
        "token_counts": meta["token_counts"],
        "smoke": smoke_info,
        "written": now_iso(),
    }
    atomic_json(out / "protocol.json", payload)
    print(
        f"PROTOCOL final_steps {final_steps} tune_steps {tune_steps} "
        f"watchdog_hours {payload['watchdog_seconds']/3600:.2f}",
        flush=True,
    )
    return payload


def run_prepare(out: Path, data_root: Path, device) -> None:
    if (out / "protocol.json").exists():
        print("PROTOCOL already stored", flush=True)
        return
    checks = check_objectives(device)
    print(f"objective checks {json.dumps(checks['collapsed'])}", flush=True)
    loaded = load_or_prepare(data_root, out)
    try:
        smoke_info = smoke(device, loaded["tensors"], BATCH)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit("smoke did not fit at batch 8; refusing a different token budget")
    write_protocol(out, smoke_info, loaded["meta"])


def run_train(out: Path, data_root: Path, device) -> None:
    protocol_path = out / "protocol.json"
    if not protocol_path.exists():
        raise SystemExit("protocol.json is missing")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("final_tokens_per_run") != FINAL_TOKENS:
        raise SystemExit("protocol token budget is not the WikiText count")
    loaded = load_or_prepare(data_root, out)
    tensors = loaded["tensors"]
    state = load_state(out)
    state["protocol"] = protocol
    state["status"] = "RUNNING"
    batch = int(protocol["batch"])
    for loss_name in ("cross_entropy", "brier"):
        for lr in protocol["learning_rates"]:
            key = f"tune-{loss_name}-seed0-lr{lr}"
            if key in state["markers"]:
                continue
            record, _test = train_one(
                out, device, tensors["train"], tensors["valid"], None,
                loss_name, lr, 0, int(protocol["tune_steps"]), batch, "tune",
            )
            state["records"].append(record)
            state["markers"].append(key)
            save_state(out, state)
            _drop_ckpt(out, key)
    if not state.get("selection"):
        state["selection"] = select_lr(state["records"])
        save_state(out, state)
        print(f"SELECTION {state['selection']}", flush=True)
    for seed in range(int(protocol["seeds"])):
        shared = out / "ckpt" / f"init_seed{seed}.pt"
        if not shared.exists():
            torch.manual_seed(seed)
            model = build_model(50257, device)
            shared.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, shared)
            del model
            torch.cuda.empty_cache()
        for loss_name in ("cross_entropy", "brier"):
            lr = state["selection"][loss_name]
            key = f"final-{loss_name}-seed{seed}-lr{lr}"
            if key in state["markers"]:
                continue
            record, test_record = train_one(
                out, device, tensors["train"], tensors["valid"], tensors["test"],
                loss_name, lr, seed, int(protocol["final_steps"]), batch, "final",
            )
            state["records"].append(record)
            if test_record is not None:
                state["records"].append(test_record)
            state["markers"].append(key)
            save_state(out, state)
            _drop_ckpt(out, key)
    unfinished = [
        row for row in state["records"]
        if row["phase"] == "final" and row["split"] == "validation" and row["incomplete"]
    ]
    state["status"] = "PARTIAL" if unfinished else "DONE"
    save_state(out, state)
    print(f"FINEWEB_DONE {state['status']}", flush=True)


def main() -> None:
    import argparse

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
