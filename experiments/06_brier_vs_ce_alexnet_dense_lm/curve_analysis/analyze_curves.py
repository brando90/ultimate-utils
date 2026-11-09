"""Plot and summarize saved training-loss curves; never train or evaluate a model.

Usage from the worktree:
  python3 experiments/06_brier_vs_ce_alexnet_dense_lm/curve_analysis/analyze_curves.py
  python3 experiments/06_brier_vs_ce_alexnet_dense_lm/curve_analysis/analyze_curves.py --check
  python3 experiments/06_brier_vs_ce_alexnet_dense_lm/curve_analysis/analyze_curves.py --available-only

The default writes only this directory's derived JSON, PNG, and SVG outputs.
--check recomputes the numbers and compares the JSON without writing anything.
--available-only explicitly permits a partial report, visibly marking missing data.
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
EXPERIMENT = HERE.parent
OUTPUT_JSON = HERE / "saved_curve_statistics.json"
INPUTS = {
    "wikitext103": ("WikiText-103", EXPERIMENT / "lm_fixed/state.json"),
    "fineweb_edu": ("FineWeb-Edu", EXPERIMENT / "fineweb_edu/state.json"),
}
LOSSES = ("cross_entropy", "brier")
SEEDS = (0, 1, 2)
BLOCK_WIDTH = 2000
WINDOWS = {
    "60_to_80_percent": (0.60, 0.80),
    "80_to_90_percent": (0.80, 0.90),
    "90_to_100_percent": (0.90, 1.00),
    "90_to_95_percent": (0.90, 0.95),
    "95_to_100_percent": (0.95, 1.00),
    "80_to_100_percent": (0.80, 1.00),
}
COMPARISONS = {
    "last_10_percent_vs_previous_10_percent": ("80_to_90_percent", "90_to_100_percent"),
    "last_5_percent_vs_previous_5_percent": ("90_to_95_percent", "95_to_100_percent"),
    "last_20_percent_vs_previous_20_percent": ("60_to_80_percent", "80_to_100_percent"),
}


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def exact_seed_summary(values: list[float]) -> dict:
    """Three-seed percentile bootstrap and exact two-sided sign-flip test."""
    assert len(values) == 3
    observed = mean(values)
    resamples = sorted(mean(list(row)) for row in itertools.product(values, repeat=3))
    interval = [resamples[math.ceil(p * len(resamples)) - 1] for p in (0.025, 0.975)]
    extreme = sum(
        abs(mean([v * s for v, s in zip(values, signs)])) + 1e-15 >= abs(observed)
        for signs in itertools.product((-1, 1), repeat=3)
    )
    return {
        "n_training_seeds": 3,
        "seed_values": values,
        "mean": observed,
        "sample_standard_deviation": statistics.stdev(values),
        "percentile_bootstrap_95_interval": interval,
        "bootstrap_ordered_resamples": 27,
        "bootstrap_quantile_definition": "inverse empirical cumulative distribution at 0.025 and 0.975",
        "two_sided_sign_flip_p_value": extreme / 8,
        "sign_flip_extreme_assignments": extreme,
        "sign_flip_total_assignments": 8,
        "sign_flip_null": "seed-level differences are symmetric about zero; signs are exchangeable",
        "seeds_with_negative_change": sum(v < 0 for v in values),
    }


def window_summary(curve: list[dict], low: float, high: float) -> dict:
    points = [p for p in curve if low <= p["step"] < high]
    assert len(points) >= 2, (low, high, len(points))
    x = [p["step"] for p in points]
    y = [p["train_loss"] for p in points]
    x_mean, y_mean = mean(x), mean(y)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sum((a - x_mean) ** 2 for a in x)
    return {
        "lower_step_inclusive": low,
        "upper_step_exclusive": high,
        "first_logged_step": min(x),
        "last_logged_step": max(x),
        "n_logged_minibatches": len(points),
        "mean_training_objective": y_mean,
        "sample_standard_deviation_across_logged_minibatches": statistics.stdev(y),
        "ordinary_least_squares_slope_per_1000_steps": slope * 1000,
    }


def block_means(curve: list[dict]) -> list[dict]:
    blocks: dict[int, list[dict]] = {}
    for point in curve:
        blocks.setdefault(point["step"] // BLOCK_WIDTH, []).append(point)
    return [
        {
            "block_start_step": block * BLOCK_WIDTH,
            "block_end_step_exclusive": (block + 1) * BLOCK_WIDTH,
            "first_logged_step": points[0]["step"],
            "last_logged_step": points[-1]["step"],
            "mean_logged_step": mean([p["step"] for p in points]),
            "n_logged_minibatches": len(points),
            "mean_training_objective": mean([p["train_loss"] for p in points]),
        }
        for block, points in sorted(blocks.items())
    ]


def load_and_analyze(available_only: bool = False) -> dict:
    missing_keys = [key for key, (_, path) in INPUTS.items() if not path.is_file()]
    missing = [{"corpus": key, "label": INPUTS[key][0], "source_path_relative_to_experiment": str(INPUTS[key][1].relative_to(EXPERIMENT))} for key in missing_keys]
    if missing and not available_only:
        raise SystemExit("Required saved input missing; no analysis output written: " + ", ".join(item["source_path_relative_to_experiment"] for item in missing))
    if len(missing) == len(INPUTS):
        raise SystemExit("No saved inputs are available; no analysis output written.")
    trainer_paths = [EXPERIMENT / "train_lm_fixed.py", EXPERIMENT / "train_fineweb.py", EXPERIMENT / "losses.py"]
    output = {
        "analysis_version": 1,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_status": "PARTIAL_INPUTS_MISSING" if missing else "COMPLETE",
        "missing_inputs": missing,
        "analysis_scope": "Exploratory post-processing of saved training minibatch losses only; no new training or evaluation.",
        "primary_comparison": "last_10_percent_vs_previous_10_percent",
        "comparison_direction": "later-window mean minus earlier-window mean; negative means the logged own-objective average fell",
        "window_definition": "Fractions of planned_steps; include logged steps at or above the lower bound and strictly below the upper bound.",
        "statistical_unit": "training seed, n=3 for each corpus and training objective",
        "multiplicity": "Unadjusted, exploratory comparisons; primary and sensitivity windows are all reported, with no selection of the best-looking window.",
        "limitations": [
            "Logged points measure different minibatches, not a fixed training or held-out evaluation set.",
            "Within-curve minibatches are not independent experiment replications; slope standard errors and minibatch-based p-values are not reported.",
            "Three-seed bootstrap intervals are discrete empirical summaries, not reliable coverage guarantees; they exclude data, architecture, schedule, and tuning uncertainty.",
            "The bootstrap interval is not an inversion of the sign-flip test; the smallest attainable two-sided sign-flip p-value is 0.25 for three nonzero differences.",
            "Cross-entropy and Brier training objectives have different scales; their raw magnitudes, slopes, and percentage changes must not be compared as learning-speed measures.",
            "No intermediate held-out metrics are present, so these curves cannot identify equal-quality arrival times, a converged minimum, or an asymptotic ordering.",
        ],
        "trainer_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in trainer_paths},
        "logging_semantics": {
            "curve_fields": ["step", "train_loss", "grad_norm"],
            "loss_measurement": "Mean selected training objective over one minibatch: eight sequences, each with 255 next-token targets, total 2040 targets.",
            "timing": "Loss computed before the update at zero-based step j; the pre-update value is appended after opt.step().",
            "logging_frequency": "Every 100 steps, plus the final zero-based step.",
            "retention": "Only the last 400 points are retained in final validation records; test records intentionally have empty curves.",
            "schedule": "500-step linear warmup to the selected peak 0.0003, then cosine decay toward 0.1 times that peak (0.00003).",
            "held_out_metrics": "Validation/test metrics are evaluated at the end of each run only; tuning endpoints are separate runs with shorter schedules.",
            "source_locations": ["train_lm_fixed.py:51", "train_lm_fixed.py:98", "train_lm_fixed.py:384", "train_lm_fixed.py:391", "train_lm_fixed.py:415", "train_lm_fixed.py:436", "train_lm_fixed.py:454", "train_fineweb.py:218"],
        },
        "plot_smoothing": {
            "method": "Nonoverlapping 2000-step bins anchored at step 0; mean of logged losses within each bin, plotted at mean logged step; then an equally weighted mean across three seeds.",
            "block_width_steps": BLOCK_WIDTH,
            "uncertainty_band": "none; individual seed traces show seed variability",
            "separate_y_scales": True,
        },
        "corpora": {},
    }
    for corpus_key, (label, path) in INPUTS.items():
        if corpus_key in missing_keys:
            continue
        raw = path.read_bytes()
        state = json.loads(raw)
        assert state["status"] == "DONE", (label, state["status"])
        protocol = state["protocol"]
        assert ("fineweb" if corpus_key == "fineweb_edu" else "wikitext") in protocol["corpus"].lower(), (label, protocol["corpus"])
        assert protocol["final_steps"] == 57859
        assert protocol["batch"] == 8 and protocol["seq_len"] == 256
        assert protocol["parameters"] == 123551232 and protocol["seeds"] == 3
        assert state["selection"] == {loss: 0.0003 for loss in LOSSES}
        rows = [r for r in state["records"] if r["phase"] == "final" and r["split"] == "validation"]
        assert len(rows) == 6
        lookup = {(r["loss"], r["seed"]): r for r in rows}
        assert set(lookup) == set(itertools.product(LOSSES, SEEDS))
        corpus = {
            "label": label,
            "source_path_relative_to_experiment": str(path.relative_to(EXPERIMENT)),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_status": state["status"],
            "source_updated": state.get("updated"),
            "protocol": protocol,
            "selected_learning_rates": state["selection"],
            "losses": {},
        }
        for loss in LOSSES:
            seed_rows = []
            for seed in SEEDS:
                row = lookup[(loss, seed)]
                assert not any(row.get(flag, False) for flag in ("invalid", "incomplete", "oom"))
                steps = row["planned_steps"]
                assert row["steps"] == steps == protocol["final_steps"]
                assert row["tokens"] == 118032360 and row["batch"] == 8
                assert row["learning_rate"] == state["selection"][loss]
                curve = row["curve"]
                assert len(curve) == 400
                assert [p["step"] for p in curve] == list(range(18000, 57859, 100)) + [57858]
                assert all(set(p) == {"step", "train_loss", "grad_norm"} for p in curve)
                assert all(math.isfinite(p[field]) for p in curve for field in ("train_loss", "grad_norm"))
                windows = {name: window_summary(curve, low * steps, high * steps) for name, (low, high) in WINDOWS.items()}
                comparisons = {}
                for name, (earlier, later) in COMPARISONS.items():
                    a = windows[earlier]["mean_training_objective"]
                    b = windows[later]["mean_training_objective"]
                    comparisons[name] = {"earlier_window": earlier, "later_window": later, "earlier_mean": a, "later_mean": b, "later_minus_earlier": b - a, "relative_change_percent": 100 * (b - a) / a}
                seed_rows.append({
                    "seed": seed, "record_key": row["key"], "planned_steps": steps,
                    "n_retained_points": len(curve), "first_retained_step": curve[0]["step"], "last_retained_step": curve[-1]["step"],
                    "first_retained_loss": curve[0]["train_loss"], "last_retained_loss": curve[-1]["train_loss"],
                    "earliest_retained_10_percent_of_total_training_window": window_summary(curve, curve[0]["step"], curve[0]["step"] + 0.1 * steps),
                    "windows": windows, "comparisons": comparisons, "plot_block_means": block_means(curve),
                })
            comparison_summaries = {}
            for name in COMPARISONS:
                values = [r["comparisons"][name]["later_minus_earlier"] for r in seed_rows]
                summary = exact_seed_summary(values)
                summary["earlier_mean_across_seeds"] = mean([r["comparisons"][name]["earlier_mean"] for r in seed_rows])
                summary["later_mean_across_seeds"] = mean([r["comparisons"][name]["later_mean"] for r in seed_rows])
                comparison_summaries[name] = summary
            slope_summaries = {}
            for name in ("90_to_100_percent", "80_to_100_percent"):
                values = [r["windows"][name]["ordinary_least_squares_slope_per_1000_steps"] for r in seed_rows]
                slope_summaries[name] = {"seed_slopes_per_1000_steps": values, "mean_slope_per_1000_steps": mean(values), "sample_standard_deviation_across_seeds": statistics.stdev(values), "interpretation": "descriptive only; no independent-minibatch inference"}
            corpus["losses"][loss] = {"seeds": seed_rows, "comparison_summaries": comparison_summaries, "slope_summaries": slope_summaries}
        output["corpora"][corpus_key] = corpus
    return output


def plot(result: dict, late_zoom: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "svg.fonttype": "none", "svg.hashsalt": "brier-ce-saved-curves-v1"})
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.4), sharex=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    earlier_color, later_color = "#bcd9f0", "#f5d49a"
    for i, (corpus_key, (label, _)) in enumerate(INPUTS.items()):
        corpus = result["corpora"].get(corpus_key)
        if corpus is None:
            for j, loss in enumerate(LOSSES):
                ax = axes[i, j]
                objective = "Cross-entropy training" if loss == "cross_entropy" else "Brier training"
                ax.set_title(f"{label} · {objective}", loc="left", pad=10)
                ax.set_facecolor("#f4f5f7")
                ax.text(0.5, 0.54, f"Saved {label} curves unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=14, color="#4a4a4a", fontweight="bold")
                ax.text(0.5, 0.40, "Retrieval pending", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#666666")
                ax.tick_params(axis="both", left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            continue
        total = corpus["protocol"]["final_steps"]
        for j, loss in enumerate(LOSSES):
            ax = axes[i, j]
            data = corpus["losses"][loss]
            values_for_limits = []
            for seed_data, color in zip(data["seeds"], colors):
                blocks = seed_data["plot_block_means"]
                x = [b["mean_logged_step"] for b in blocks]
                y = [b["mean_training_objective"] for b in blocks]
                ax.plot(x, y, color=color, linewidth=1.35, alpha=0.80, marker="o", markersize=3.0)
                values_for_limits.extend(value for step, value in zip(x, y) if not late_zoom or step >= 0.8 * total)
            all_blocks = [r["plot_block_means"] for r in data["seeds"]]
            x = [b["mean_logged_step"] for b in all_blocks[0]]
            y = [mean([b[k]["mean_training_objective"] for b in all_blocks]) for k in range(len(x))]
            ax.plot(x, y, color="#1b1b1b", linewidth=2.7, marker="o", markersize=3.8, zorder=5)
            ax.axvspan(0.8 * total, 0.9 * total, color=earlier_color, alpha=0.55, zorder=0)
            ax.axvspan(0.9 * total, total, color=later_color, alpha=0.55, zorder=0)
            ax.axvline(0.9 * total, color="#777777", linewidth=0.8, linestyle=":", zorder=1)
            objective = "Cross-entropy training" if loss == "cross_entropy" else "Brier training"
            ax.set_title(f"{corpus['label']} · {objective}", loc="left", pad=10)
            ax.set_ylabel("Mean cross-entropy (nats / target)" if loss == "cross_entropy" else "Mean full Brier objective")
            ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
            ax.set_xlim(0.8 * total if late_zoom else 18000, total)
            padding = max(max(values_for_limits) - min(values_for_limits), 0.001) * 0.12
            ax.set_ylim(min(values_for_limits) - padding, max(values_for_limits) + padding)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlabel("Training step")
    subtitle = "Final 20% of training" if late_zoom else "All retained training data (steps 18,000–57,858)"
    if result["missing_inputs"]:
        missing_labels = ", ".join(item["label"] for item in result["missing_inputs"])
        subtitle += f" · {missing_labels} retrieval pending"
    fig.suptitle(f"Saved language-model training curves\n{subtitle}", x=0.08, y=0.985, ha="left", fontsize=18, fontweight="bold")
    handles = [Line2D([0], [0], color=c, linewidth=1.6, marker="o", markersize=3, label=f"Seed {seed}") for seed, c in zip(SEEDS, colors)]
    handles += [Line2D([0], [0], color="#1b1b1b", linewidth=2.7, label="Mean of 3 seeds"), Patch(facecolor=earlier_color, label="80–90% window"), Patch(facecolor=later_color, label="90–100% window")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.925), ncol=6, frameon=False, fontsize=10)
    caption = (
        "Each point: own-objective mean of logged minibatches in a nonoverlapping 2,000-step bin; black: equal mean across 3 seeds.\n"
        "Each logged loss is one 2,040-target minibatch, computed before its optimizer update; curves contain no intermediate held-out metrics.\n"
        "Panels use separate y-scales. Cross-entropy and Brier objective magnitudes/slopes are not comparable as learning speeds.\n"
        "Shading marks the late windows selected for this analysis; statistics use unsmoothed logged losses. Saved records only; no new runs."
    )
    fig.text(0.08, 0.025, caption, ha="left", va="bottom", fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.19, top=0.855, hspace=0.38, wspace=0.28)
    stem = "saved_training_curves_late_zoom" if late_zoom else "saved_training_curves"
    fig.savefig(HERE / f"{stem}.png", dpi=180, metadata={"Software": "saved-curve analysis; no training or evaluation"})
    fig.savefig(HERE / f"{stem}.svg", metadata={"Date": None, "Description": caption})
    svg_path = HERE / f"{stem}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the numerical JSON without writing or plotting.")
    parser.add_argument("--available-only", action="store_true", help="Explicitly allow a partial report and visibly mark unavailable corpus panels.")
    args = parser.parse_args()
    result = load_and_analyze(available_only=args.available_only)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.check:
        assert OUTPUT_JSON.read_text() == serialized, "Derived numerical JSON does not reproduce exactly."
        count = len(result["corpora"]) * 6
        print(f"PASS: numerical JSON reproduces exactly; all {count} available final curves validated; status {result['analysis_status']}; no files written.")
        return
    OUTPUT_JSON.write_text(serialized)
    plot(result, late_zoom=False)
    plot(result, late_zoom=True)
    print(f"Saved numerical JSON and full/late-zoom PNG and SVG plots; status {result['analysis_status']}.")
    for corpus in result["corpora"].values():
        for loss in LOSSES:
            s = corpus["losses"][loss]["comparison_summaries"][result["primary_comparison"]]
            print(f"{corpus['label']} {loss}: late-minus-previous={s['mean']:+.9g}; 95% seed-bootstrap={s['percentile_bootstrap_95_interval']}; p={s['two_sided_sign_flip_p_value']}; n=3")


if __name__ == "__main__":
    main()
