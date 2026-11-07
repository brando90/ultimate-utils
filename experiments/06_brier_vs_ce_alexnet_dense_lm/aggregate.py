"""Build the live results markdown from the run state.

The paired comparison is cross-entropy-trained minus Brier-trained, on the
same seed. The resampling unit is the training seed.
"""

from __future__ import annotations

import json
from pathlib import Path

from metrics import paired_seed_comparison

METRICS = (
    "accuracy",
    "cross_entropy",
    "perplexity",
    "brier",
    "local_square",
    "expected_calibration_error",
)


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def comparison_table(records: list[dict], dataset: str) -> tuple[list[str], dict]:
    finals = [
        r
        for r in records
        if r.get("dataset") == dataset
        and r.get("phase") == "final"
        and r.get("split") == "test"
        and not r.get("invalid")
    ]
    lines = [
        f"### {dataset} held-out test",
        "",
        "| loss | seed | learning rate | accuracy | cross-entropy | perplexity | Brier score | local square | ECE | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_loss = {"cross_entropy": {}, "brier": {}}
    for row in sorted(finals, key=lambda r: (r["loss"], r["seed"])):
        m = row["metrics"]
        lines.append(
            "| {loss} | {seed} | {lr:.6g} | {acc} | {ce} | {ppl} | {br} | {loc} | {ece} | {n} |".format(
                loss=row["loss"],
                seed=row["seed"],
                lr=row["learning_rate"],
                acc=_fmt(m["accuracy"]),
                ce=_fmt(m["cross_entropy"]),
                ppl=_fmt(m["perplexity"]),
                br=_fmt(m["brier"]),
                loc=_fmt(m["local_square"]),
                ece=_fmt(m["expected_calibration_error"]),
                n=m["n"],
            )
        )
        by_loss[row["loss"]][row["seed"]] = m
    seeds = sorted(set(by_loss["cross_entropy"]) & set(by_loss["brier"]))
    stats = {}
    if seeds:
        lines.append("")
        lines.append(
            "Paired difference is the cross-entropy-trained run minus the Brier-trained run. "
            "The 95% interval resamples training seeds. p-val is a two-sided sign-flip test."
        )
        lines.append("")
        lines.append("| metric | mean difference | 95% interval | p-val | n seeds |")
        lines.append("| --- | ---: | --- | ---: | ---: |")
        for metric in METRICS:
            a = [by_loss["cross_entropy"][s][metric] for s in seeds]
            b = [by_loss["brier"][s][metric] for s in seeds]
            if any(v is None or v == float("inf") for v in a + b):
                continue
            comp = paired_seed_comparison(a, b)
            stats[metric] = comp
            lo, hi = comp["ci95"]
            lines.append(
                f"| {metric} | {comp['mean_diff']:.6g} | [{lo:.6g}, {hi:.6g}] | {comp['p_value']:.4g} | {comp['n']} |"
            )
    else:
        lines.append("")
        lines.append("No paired held-out test rows yet.")
    lines.append("")
    return lines, stats


def render_results(state: dict) -> str:
    records = state.get("records", [])
    lines = [
        "# Brier score versus cross-entropy on AlexNet and a dense language model",
        "",
        f"**Status:** {state.get('status', 'UNKNOWN')}",
        "",
        f"**Last updated:** {state.get('updated', '')}",
        "",
        f"**Doc link:** https://github.com/brando90/ultimate-utils/blob/main/experiments/06_brier_vs_ce_alexnet_dense_lm/results.md",
        "",
        "The link resolves after this branch is on `main`. Definitions and the procedure fixed before the run are in `README.md` in this folder.",
        "",
        "## Phases",
        "",
        "| phase | state |",
        "| --- | --- |",
    ]
    for name, phase_state in state.get("phases", {}).items():
        lines.append(f"| {name} | {phase_state} |")
    lines.append("")
    if state.get("placement"):
        lines.append("## Placement")
        lines.append("")
        for key, value in state["placement"].items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    if state.get("notes"):
        lines.append("## Notes")
        lines.append("")
        for note in state["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("## Held-out test")
    lines.append("")
    lines.append(
        "Learning rates were selected on the validation split. These rows are the official test split, scored after that selection."
    )
    lines.append("")
    all_stats = {}
    for dataset in ("mnist", "cifar10", "wikitext"):
        table, stats = comparison_table(records, dataset)
        lines.extend(table)
        all_stats[dataset] = stats
    if state.get("baseline"):
        lines.append("## Language-model baseline")
        lines.append("")
        lines.append("Add-one unigram fit on the training tokens only, scored on the validation tokens.")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(state["baseline"], indent=2, default=str)[:4000])
        lines.append("```")
        lines.append("")
    phases = state.get("phases") or {}
    lm_blocked = phases.get("lm_tune") == "BLOCKED" or phases.get("lm_final") == "BLOCKED"
    if lm_blocked:
        lines.append(
            "The language-model phase did not train. The legacy WikiText zip downloads were not zip files. "
            "The tables above are the AlexNet comparison only."
        )
    else:
        lines.append(
            "This is one architecture, one corpus, and the token count written in the phase notes. "
            "It is not a result about a frontier language model."
        )
    lines.append("")
    status = state.get("status", "")
    if lm_blocked:
        closing = (
            "AlexNet on MNIST and CIFAR-10 finished. The dense language model did not train."
        )
    else:
        closing = (
            "AlexNet on MNIST and CIFAR-10, and one dense causal language model, "
            "trained with the full Brier score and with cross-entropy under matched data, initialization, and step budgets."
        )
    lines.append(
        f"**TLDR-end:** [ultimate-utils: brier vs cross-entropy] Status {status}. {closing}"
    )
    lines.append("")
    state["comparisons"] = all_stats
    return "\n".join(lines) + "\n"


def write_results(out_dir: Path, state: dict) -> None:
    text = render_results(state)
    path = out_dir / "results.md"
    path.write_text(text)
    # Keep a copy at the experiment root when the out dir is the experiment results folder.
    summary = {
        "status": state.get("status"),
        "updated": state.get("updated"),
        "phases": state.get("phases"),
        "placement": state.get("placement"),
        "selection": state.get("selection"),
        "notes": state.get("notes"),
        "comparisons": state.get("comparisons"),
        "records": state.get("records"),
        "baseline": state.get("baseline"),
        "budget": state.get("budget"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
