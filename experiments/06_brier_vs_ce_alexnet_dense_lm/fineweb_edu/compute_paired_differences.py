"""Analyze saved FineWeb-Edu metrics; never load a model or run evaluation.

Run with --write to regenerate paired_differences.md, --check to verify it,
or no option to print full-precision derived statistics as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
STATE_SHA256 = "eec2e1453221c15e9d3442753b9e7328a8b110bbd4e9e24b0ad759a8c133891a"
METRICS = ("accuracy", "cross_entropy", "perplexity", "brier", "expected_calibration_error")
LABELS = ("Accuracy", "Cross-entropy", "Perplexity", "Full Brier score", "Top-label expected calibration error")
LOSSES = ("cross_entropy", "brier")
REPORT_URL = "https://github.com/brando90/ultimate-utils/blob/cursor/brier-ce-pilot-076e/experiments/06_brier_vs_ce_alexnet_dense_lm/fineweb_edu/paired_differences.md"


def compare(a: list[float], b: list[float]) -> dict:
    """Exact n=3 seed bootstrap and sign flips, matching metrics.py's estimand."""
    assert len(a) == len(b) == 3, "This exact-enumeration report is fixed to three seeds."
    differences = [x - y for x, y in zip(a, b)]
    mean = statistics.mean(differences)
    bootstrap = sorted(statistics.mean(sample) for sample in itertools.product(differences, repeat=3))
    # Inverse empirical CDF, retaining the probability mass of each resample.
    ci95 = [bootstrap[math.ceil(p * len(bootstrap)) - 1] for p in (0.025, 0.975)]
    count = sum(
        abs(statistics.mean(d * s for d, s in zip(differences, signs))) + 1e-15 >= abs(mean)
        for signs in itertools.product((-1, 1), repeat=3)
    )
    assert ci95 == [min(differences), max(differences)]
    return {
        "cross_entropy_values": a,
        "brier_values": b,
        "cross_entropy_mean": statistics.mean(a),
        "brier_mean": statistics.mean(b),
        "cross_entropy_sample_sd": statistics.stdev(a),
        "brier_sample_sd": statistics.stdev(b),
        "paired_differences": differences,
        "mean_difference": mean,
        "paired_difference_sample_sd": statistics.stdev(differences),
        "ci95": ci95,
        "p_value": count / 8,
        "sign_flip_count": count,
        "sign_flip_total": 8,
        "bootstrap_ordered_resamples": 27,
        "n_pairs": 3,
    }


def analyze() -> dict:
    sources = {name: (HERE / name).read_bytes() for name in ("state.json", "protocol.json", "tokens_meta.json", "results.md")}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    assert hashes["state.json"] == STATE_SHA256, "The frozen input changed; investigate rather than silently reanalyze."
    state = json.loads(sources["state.json"])
    protocol = json.loads(sources["protocol.json"])
    metadata = json.loads(sources["tokens_meta.json"])
    assert state["status"] == "DONE"
    assert state["protocol"] == protocol
    assert protocol["corpus"] == metadata["name"] == "HuggingFaceFW/fineweb-edu sample-10BT"
    assert protocol["revision"] == metadata["revision"]
    assert protocol["id_sha256"] == metadata["id_sha256"]
    assert state["selection"] == {loss: 0.0003 for loss in LOSSES}
    records = state["records"]
    assert len(records) == 20
    assert all(not row.get(flag, False) for row in records for flag in ("invalid", "incomplete", "oom"))
    assert len({row["key"] for row in records if row["phase"] == "tune"}) == 8
    assert len({row["key"] for row in records if row["phase"] == "final"}) == 6
    seeds = list(range(protocol["seeds"]))
    assert seeds == [0, 1, 2]
    result = {"source_commit": "a1fd6973bff8d9da770284050970714d94bb266c", "source_sha256": hashes, "seeds": seeds, "splits": {}}
    lookup = {}
    for split in ("test", "validation"):
        rows = [r for r in records if r["phase"] == "final" and r["split"] == split]
        by_key = {(r["loss"], r["seed"]): r for r in rows}
        assert len(rows) == len(by_key) == 6
        assert set(by_key) == set(itertools.product(LOSSES, seeds))
        denominators = {r["metrics"]["n"] for r in rows}
        assert denominators == {1992315}
        for row in rows:
            assert row["steps"] == row["planned_steps"] == protocol["final_steps"] == 57859
            assert row["tokens"] == protocol["final_tokens_per_run"] == 118032360
            assert row["learning_rate"] == state["selection"][row["loss"]]
            assert all(math.isfinite(row["metrics"][metric]) for metric in METRICS)
            lookup[(split, row["loss"], row["seed"])] = row
        result["splits"][split] = {
            "scored_targets_per_run": 1992315,
            "metrics": {
                metric: compare(*[[by_key[(loss, seed)]["metrics"][metric] for seed in seeds] for loss in LOSSES])
                for metric in METRICS
            },
        }
    # The public table is rounded to six significant digits. Check every final row.
    split, checked = None, 0
    for line in sources["results.md"].decode().splitlines():
        if line in ("### validation", "### test"):
            split = line.removeprefix("### ")
        if line.startswith(("| cross_entropy |", "| brier |")):
            cells = [c.strip() for c in line.strip("|").split("|")]
            row = lookup[(split, cells[0], int(cells[1]))]
            assert float(cells[2]) == row["learning_rate"]
            assert int(cells[3]) == row["steps"] and int(cells[4]) == row["tokens"]
            assert cells[10] == "False"
            for metric, printed in zip(METRICS, cells[5:10]):
                assert format(row["metrics"][metric], ".6g") == printed, (split, metric, printed)
            checked += 1
    assert checked == 12
    result["checks"] = {"final_table_rows_match_to_printed_precision": checked, "invalid": 0, "incomplete": 0, "out_of_memory": 0}
    return result


def render(result: dict) -> str:
    lines = [
        "# FineWeb-Edu: paired differences from the finished runs",
        "", f"**Doc link:** <{REPORT_URL}>", "",
        "**TLDR:** At the matched training budget, cross-entropy training had higher held-out accuracy and lower cross-entropy, perplexity, and full Brier score in all three seed pairs; Brier training had lower top-label expected calibration error. Every exact two-sided sign-flip p-value is 0.25, the minimum attainable with three nonzero pairs, so these results do not establish a general superiority claim.",
        "", "Analysis date: 11-08-2026. This is post-processing of saved metrics only; no training or evaluation was rerun. The original `results.md`, `state.json`, `protocol.json`, and `tokens_meta.json` are unchanged.",
        "", "## Scope and method", "",
        "- Experiment: `experiments/06_brier_vs_ce_alexnet_dense_lm/fineweb_edu`. The 123,551,232-parameter Transformer trained for 57,859 steps and 118,032,360 tokens per final run. Both objectives selected learning rate 0.0003. There are three paired training seeds, 0–2, and 1,992,315 scored targets per run on each split. The target count is not the number of independent replications.",
        "- Difference is cross-entropy-trained minus Brier-trained, paired by seed. Positive accuracy favors cross-entropy training; negative cross-entropy, perplexity, or full Brier score favors cross-entropy training; positive expected calibration error favors Brier training. Accuracy and expected calibration error are fractions, not percentage points. Cross-entropy uses natural logarithms; perplexity is dimensionless. Full Brier score is the unnormalized sum of squared probability errors across classes, with range 0–2.",
        "- The 95% interval is the percentile bootstrap of the paired mean difference, resampling the three training seeds with replacement. This report enumerates all 3³ = 27 equally likely ordered samples and uses inverse empirical cumulative-distribution-function quantiles at 0.025 and 0.975. Each extreme has probability 1/27 > 0.025, so these endpoints equal the minimum and maximum observed paired difference. This computes the exact bootstrap distribution; the older tables in `../metrics.py` use 10,000 pseudorandom draws with PyTorch seed 0. The target resampling distribution is the same, but this report does not claim to reuse that random-number stream.",
        "- The p-value enumerates all 2³ = 8 sign assignments and counts absolute mean differences at least as large as observed (with the existing code's 1e-15 comparison tolerance). The null is that the seed-paired difference distribution is symmetric about zero, so its signs are exchangeable. Every reported test has count 2/8 = 0.25. P-values are unadjusted across five correlated metrics; validation is secondary and was used for selection. No new acceptance threshold is introduced.",
        "- These intervals describe empirical training-seed variability conditional on this architecture, corpus shard, splits, tuning grid, and token budget. With only three seeds the bootstrap is extremely discrete and cannot support reliable population-coverage or broad generalization claims; it does not include corpus, document, architecture, or tuning-procedure uncertainty. An interval excluding zero and a coarse p-value of 0.25 are not contradictory: the small-sample bootstrap interval is not the inversion of the sign-flip test. Failure to reject does not establish equality.",
        "- Top-label expected calibration error uses 15 equal-width bins of maximum softmax probability; it is neither classwise calibration nor full-distribution probability quality. Perplexity means below average each seed's perplexity; they are not the exponential of mean cross-entropy.",
        "",
    ]
    for split in ("test", "validation"):
        title = "Held-out test" if split == "test" else "Validation (secondary; used for selection)"
        lines.extend([f"## {title}", "", "All rows: 3 paired seeds; 1,992,315 scored targets per run.", "", "| Metric | Cross-entropy-trained mean | Brier-trained mean | Paired mean difference [95% seed-bootstrap interval] | Two-sided p-value |", "| --- | ---: | ---: | --- | ---: |"])
        for metric, label in zip(METRICS, LABELS):
            s = result["splits"][split]["metrics"][metric]
            lo, hi = s["ci95"]
            lines.append(f"| {label} | {s['cross_entropy_mean']:.9g} | {s['brier_mean']:.9g} | {s['mean_difference']:+.9g} [{lo:+.9g}, {hi:+.9g}] | {s['p_value']:.2f} |")
        lines.extend(["", "| Metric | Seed 0 difference | Seed 1 difference | Seed 2 difference |", "| --- | ---: | ---: | ---: |"])
        for metric, label in zip(METRICS, LABELS):
            s = result["splits"][split]["metrics"][metric]
            lines.append(f"| {label} | " + " | ".join(format(d, "+.17g") for d in s["paired_differences"]) + " |")
        lines.extend(["", "Sample standard deviations across the same three seeds (denominator n−1):", "", "| Metric | Cross-entropy-trained | Brier-trained | Paired difference |", "| --- | ---: | ---: | ---: |"])
        for metric, label in zip(METRICS, LABELS):
            s = result["splits"][split]["metrics"][metric]
            lines.append(f"| {label} | {s['cross_entropy_sample_sd']:.9g} | {s['brier_sample_sd']:.9g} | {s['paired_difference_sample_sd']:.9g} |")
        lines.append("")
    lines.extend([
        "## Provenance and checks", "",
        "Source revision: `a1fd6973bff8d9da770284050970714d94bb266c`. Protocol and token metadata identify `HuggingFaceFW/fineweb-edu sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The stored per-record `dataset: wikitext103` is a hardcoded label from the reused trainer; selection here uses the FineWeb file and its matching protocol, not that label.", "",
        "Checks: 20 records = 8 tuning records + validation/test records for 6 final runs; all final seed/loss/split combinations are unique and complete. There are 0 invalid, 0 incomplete, and 0 out-of-memory records. All 12 final rows match the untouched `results.md` at its printed precision; `state.protocol` equals `protocol.json`, whose revision and document-ID hash match `tokens_meta.json`.", "",
        "SHA-256 file digests:", "",
    ])
    for name, digest in result["source_sha256"].items():
        lines.append(f"- `{name}`: `{digest}`")
    lines.extend(["", "Reproduce from the designated worktree:", "", "```bash", "python3 experiments/06_brier_vs_ce_alexnet_dense_lm/fineweb_edu/compute_paired_differences.py --check", "# Omit --check to print full-precision source values and derived statistics as JSON.", "# Use --write to regenerate this derived report only.", "```", "", "**TLDR-end:** [ultimate-utils: FineWeb-Edu paired differences] `experiments/06_brier_vs_ce_alexnet_dense_lm/fineweb_edu` favors cross-entropy training on held-out accuracy and the two proper scoring rules at this budget, while Brier training has lower top-label calibration error; three seeds limit statistical resolution and scope.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    report = render(result)
    path = HERE / "paired_differences.md"
    if args.write:
        path.write_text(report)
        print(f"Wrote {path}")
    elif args.check:
        assert path.read_text() == report, "Derived report differs from computed output."
        print("PASS: derived report reproduces exactly; 12 final rows checked; frozen state hash verified.")
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
