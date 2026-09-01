"""
Validity-Stability scatter: proxy-vs-target item-level joint plot.

Every element earns its place by being interpretable -- a reader looks at
everything on the canvas and asks what it means, so nothing is decoration:

  - Central panel: one marker per DISTINCT (target, proxy) position, sized by
    how many items share it (the count encoding a tie-heavy lattice needs
    instead of a smoothed density), with a vertical error bar = that
    position's mean repeat std (proxy self-consistency).
  - Marginal histograms (top/right), discrete-aware (tie-snapped) -- the m0
    diagnostic -- with an optional 1D KDE line over each for shape legibility
    (`show_marginal_kde`, on by default). A 1D density of one variable is a
    normal, safe smoothing choice; nothing about it can be mistaken for
    evidence of a relationship between two variables.
  - The correlation (validity) with a bootstrap CI, plus Kendall tau-b.

No 2D density wash in the central panel by default: a joint KDE can encode
dependence, but at the intended small sample size on a tie-heavy lattice it
smooths discrete support into an apparent continuum and can obscure exact
counts. The count-sized markers preserve the observed pairing directly.
`show_density_shading=True` remains available for exploratory rendering, but
defaults off and is not recommended for the publication figure.

Deliberately NOT drawing a y=x reference line by default: proxy and target
are frequently on uncalibrated scales (e.g. a judge's rubric vs a human's
rubric), and the validity score is correlation-based -- invariant to a level
offset by construction -- so y=x would imply a calibration target the score
does not claim. Pass `show_diagonal=True` when proxy and target really are on
the same calibrated scale.

See uutils.stats_uu.validity_stability for the score this figure visualizes.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from uutils.stats_uu.validity_stability import (
    _resolve_repeat_item_ids,
    ValidityStabilityResult,
    compute_validity_stability,
    bootstrap_correlation_ci,
    compute_kendall_tau_b,
    compute_calibration_gap,
)

try:
    from scipy.stats import gaussian_kde

    _HAVE_SCIPY_KDE = True
except ImportError:  # pragma: no cover - scipy is a declared dependency
    _HAVE_SCIPY_KDE = False

# Chart chrome (Claude Code dataviz-skill reference palette).
_POINT_COLOR = "#2a78d6"  # categorical slot 1
_HIST_COLOR = "#9ec5f4"  # lighter step of the same hue, for the marginals
_ERRORBAR_COLOR = "#184f95"  # darker step of the same hue
_DIAGONAL_COLOR = "#c3c2b7"  # baseline / axis
_GRID_COLOR = "#e1e0d9"  # hairline gridline
_TEXT_MUTED = "#898781"
_TEXT_PRIMARY = "#0b0b0b"
_SURFACE = "#fcfcfb"
_SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]


def _sequential_blue_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("vs_seq_blue", _SEQ_BLUE, N=256)


def _discrete_bin_edges(
    data: np.ndarray, pad_frac: float = 0.5, min_gap_fallback: float = 0.05, max_bins: int = 25
) -> np.ndarray:
    """
    Bin edges at midpoints between sorted unique values, so ties never split
    across bins -- for genuinely discrete/tie-heavy data (e.g. human labels
    averaged from a handful of integer ratings). One-bin-per-unique-value only
    makes sense while there are few enough values for that to be a histogram
    rather than a list: past `max_bins` distinct values (e.g. a proxy whose
    scores are closer to continuous), fall back to a regular grid over the
    observed range -- still count-based, just not snapped to individual values.
    """
    uniq = np.unique(np.round(data, 6))
    if uniq.size == 0:
        return np.array([0.0, 1.0])
    if uniq.size == 1:
        return np.array([uniq[0] - min_gap_fallback, uniq[0] + min_gap_fallback])
    if uniq.size <= max_bins:
        gaps = np.diff(uniq)
        min_gap = float(gaps.min())
        return np.concatenate((
            [uniq[0] - min_gap * pad_frac],
            uniq[:-1] + gaps / 2.0,
            [uniq[-1] + min_gap * pad_frac],
        ))
    lo, hi = float(uniq[0]), float(uniq[-1])
    pad = (hi - lo) * 0.02 if hi > lo else min_gap_fallback
    return np.linspace(lo - pad, hi + pad, max_bins + 1)


def _group_by_position(x: np.ndarray, y: np.ndarray, per_item_std: Optional[np.ndarray]):
    """Dedup (x, y) pairs for the scatter: count per position, mean repeat-std per position."""
    groups: dict = defaultdict(list)
    for i in range(x.shape[0]):
        key = (round(float(x[i]), 6), round(float(y[i]), 6))
        groups[key].append(per_item_std[i] if per_item_std is not None else None)
    xs = np.array([k[0] for k in groups.keys()])
    ys = np.array([k[1] for k in groups.keys()])
    counts = np.array([len(v) for v in groups.values()])
    if per_item_std is None:
        return xs, ys, counts, None
    errs = np.array([
        float(np.nanmean([s for s in v if s is not None and np.isfinite(s)]))
        if any(s is not None and np.isfinite(s) for s in v) else np.nan
        for v in groups.values()
    ])
    return xs, ys, counts, errs


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8, alpha=0.9, zorder=0)
    ax.tick_params(colors=_TEXT_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)


def _add_density_wash(ax: plt.Axes, x: np.ndarray, y: np.ndarray, seed: int, alpha: float = 0.4) -> None:
    """Decorative-only 2D density texture behind the honest markers -- see module docstring."""
    if not _HAVE_SCIPY_KDE or x.shape[0] < 5:
        return
    try:
        rng = np.random.default_rng(seed)
        jx = x + rng.normal(scale=1e-6, size=x.shape)
        jy = y + rng.normal(scale=1e-6, size=y.shape)
        kde = gaussian_kde(np.vstack([jx, jy]))
        pad = 0.06
        gx, gy = np.mgrid[-pad : 1 + pad : 120j, -pad : 1 + pad : 120j]
        density = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        ax.contourf(gx, gy, density, levels=12, cmap=_sequential_blue_cmap(), alpha=alpha, zorder=1)
    except Exception:
        pass  # decorative only -- a degenerate point cloud just skips it


def _marginal_hist(
    ax: plt.Axes, data: np.ndarray, orientation: str, seed: int, show_kde: bool
) -> None:
    """Discrete-aware histogram (the real m0 diagnostic) with an optional decorative 1D KDE line."""
    edges = _discrete_bin_edges(data)
    ax.hist(
        data, bins=edges, density=True, orientation=orientation, color=_HIST_COLOR,
        edgecolor="white", linewidth=0.5, zorder=2,
    )
    n_uniq = np.unique(np.round(data, 6)).size
    if show_kde and _HAVE_SCIPY_KDE and n_uniq >= 3:
        try:
            rng = np.random.default_rng(seed)
            jittered = data + rng.normal(scale=1e-6, size=data.shape)
            kde = gaussian_kde(jittered)
            grid = np.linspace(-0.05, 1.05, 200)
            density = kde(grid)
            if orientation == "vertical":
                ax.plot(grid, density, color=_ERRORBAR_COLOR, linewidth=1.4, alpha=0.9, zorder=3)
            else:
                ax.plot(density, grid, color=_ERRORBAR_COLOR, linewidth=1.4, alpha=0.9, zorder=3)
        except Exception:
            pass  # decorative only


def _repeat_std_vector(
    repeats: Optional[Mapping[object, Sequence[float]]],
    n_items: int,
    item_ids: Optional[Sequence[object]],
) -> Optional[np.ndarray]:
    if repeats is None:
        return None
    resolved_ids = _resolve_repeat_item_ids(repeats, n_items, item_ids)
    return np.array([
        float(np.std(np.asarray(repeats[item], dtype=float)))
        for item in resolved_ids
    ])


def _format_info_lines(
    result: ValidityStabilityResult,
    proxy: np.ndarray,
    target: np.ndarray,
    *,
    symbol: str,
    corr_method: str,
    show_calibration_gap: bool,
    n_boot: int,
    seed: int,
) -> list[str]:
    ci_lo, ci_hi = bootstrap_correlation_ci(
        proxy, target, corr_method=corr_method, n_boot=n_boot, seed=seed
    )
    tau_b = compute_kendall_tau_b(proxy, target)
    corr_symbol = r"\rho" if corr_method == "spearman" else "r"
    lines = [
        result.formula_str(symbol=symbol),
        rf"${corr_symbol} = {result.validity_raw:.2f}$  [{ci_lo:.2f}, {ci_hi:.2f}] 95% CI",
        rf"$\tau_b = {tau_b:.2f}$",
    ]
    if show_calibration_gap:
        lines.append(rf"Calibration gap = {compute_calibration_gap(proxy, target):.2f}")
    return lines


def _draw_central_panel(
    ax: plt.Axes,
    target: np.ndarray,
    proxy: np.ndarray,
    per_item_std: Optional[np.ndarray],
    *,
    show_density_shading: bool,
    show_diagonal: bool,
    seed: int,
    scatter_label: Optional[str] = None,
    label_diagonal: bool = False,
) -> Optional[np.ndarray]:
    xs_u, ys_u, counts, errs = _group_by_position(target, proxy, per_item_std)
    _style_axes(ax)
    if show_density_shading:
        _add_density_wash(ax, target, proxy, seed)
    if errs is not None:
        ax.errorbar(
            xs_u, ys_u, yerr=errs, fmt="none", ecolor=_ERRORBAR_COLOR,
            elinewidth=1.1, capsize=2.5, alpha=0.8, zorder=2,
        )
    if show_diagonal:
        ax.plot(
            [-0.05, 1.05], [-0.05, 1.05], "--", color=_DIAGONAL_COLOR,
            linewidth=1.1, alpha=0.9, zorder=1,
            label="y = x" if label_diagonal else None,
        )
    ax.scatter(
        xs_u, ys_u, s=30 + 25 * counts, color=_POINT_COLOR, edgecolor="white",
        linewidth=0.6, alpha=0.9, zorder=3, label=scatter_label,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    return errs


def _render_single_panel(
    ax: plt.Axes,
    target: np.ndarray,
    proxy: np.ndarray,
    per_item_std: Optional[np.ndarray],
    *,
    info_lines: Sequence[str],
    title: str,
    x_label: str,
    y_label: str,
    show_density_shading: bool,
    show_diagonal: bool,
    legend_loc: str,
    seed: int,
    out: Optional[Union[str, Path]],
) -> plt.Figure:
    fig = ax.figure
    _draw_central_panel(
        ax, target, proxy, per_item_std,
        show_density_shading=show_density_shading, show_diagonal=show_diagonal,
        seed=seed, scatter_label="\n".join(info_lines), label_diagonal=True,
    )
    ax.set_xlabel(x_label, color=_TEXT_PRIMARY)
    ax.set_ylabel(y_label, color=_TEXT_PRIMARY)
    ax.set_title(title, color=_TEXT_PRIMARY, fontsize=11, fontweight="bold", pad=8)
    legend = ax.legend(
        loc=legend_loc, frameon=True, framealpha=0.95, facecolor="white",
        edgecolor="#cfcfcf", fontsize=8, borderpad=0.5,
    )
    legend.set_zorder(5)
    if out is not None:
        _save(fig, out)
    return fig


def _render_joint_panel(
    target: np.ndarray,
    proxy: np.ndarray,
    per_item_std: Optional[np.ndarray],
    *,
    info_lines: Sequence[str],
    title: str,
    x_label: str,
    y_label: str,
    show_density_shading: bool,
    show_diagonal: bool,
    show_marginal_kde: bool,
    seed: int,
    out: Optional[Union[str, Path]],
) -> plt.Figure:
    fig = plt.figure(figsize=(7.4, 7.4))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1.4), height_ratios=(1.4, 4),
        left=0.11, right=0.97, bottom=0.09, top=0.93, wspace=0.06, hspace=0.06,
    )
    ax_c = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_c)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_c)
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    errs = _draw_central_panel(
        ax_c, target, proxy, per_item_std,
        show_density_shading=show_density_shading,
        show_diagonal=show_diagonal, seed=seed,
    )
    ax_c.set_xlabel(x_label, color=_TEXT_PRIMARY, fontsize=9.5)
    ax_c.set_ylabel(y_label, color=_TEXT_PRIMARY, fontsize=9.5)
    if errs is not None:
        y_label_core = y_label.split("(")[0].strip().lower()
        ax_c.text(
            0.02, 0.02,
            f"dot size = # tied items  ·  error bar = repeat SD of the {y_label_core}",
            transform=ax_c.transAxes, fontsize=6.8, color=_TEXT_MUTED,
            va="bottom", ha="left",
        )

    _style_axes(ax_top)
    _marginal_hist(ax_top, target, "vertical", seed, show_marginal_kde)
    ax_top.tick_params(labelbottom=False, labelleft=False)
    _style_axes(ax_right)
    _marginal_hist(ax_right, proxy, "horizontal", seed, show_marginal_kde)
    ax_right.tick_params(labelleft=False, labelbottom=False)
    ax_info.text(
        0.0, 1.0, "\n".join(info_lines), transform=ax_info.transAxes, fontsize=9.5,
        color=_TEXT_PRIMARY, va="top", ha="left", linespacing=1.9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cfcfcf", alpha=0.95),
    )
    fig.suptitle(title, color=_TEXT_PRIMARY, fontsize=11.5, fontweight="bold", y=0.98)
    if out is not None:
        _save(fig, out)
    return fig


def plot_validity_stability_scatter(
    proxy_scores: Sequence[float],
    target_scores: Sequence[float],
    repeat_scores_by_item: Optional[Mapping[object, Sequence[float]]] = None,
    *,
    item_ids: Optional[Sequence[object]] = None,
    std: Optional[float] = None,
    corr_method: str = "spearman",
    symbol: str = "VS",
    label: str = "",
    x_label: str = "Target score",
    y_label: str = "Proxy score",
    show_diagonal: bool = False,
    show_density_shading: bool = False,
    show_marginal_kde: bool = True,
    show_calibration_gap: bool = False,
    n_boot: int = 2000,
    ax: Optional[plt.Axes] = None,
    legend_loc: str = "best",
    seed: int = 0,
    out: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Render an item-level joint plot with count markers, repeats, and marginals.

    Integer repeat keys align positionally; named keys require ordered
    `item_ids`. `seed` controls bootstrap/KDE randomness, and saved PDF
    timestamps are stripped for byte determinism. Passing `ax` selects a
    simplified single-panel composition mode.
    """
    result = compute_validity_stability(
        proxy_scores,
        target_scores,
        repeat_scores_by_item,
        item_ids=item_ids,
        std=std,
        corr_method=corr_method,
    )
    x = np.asarray(target_scores, dtype=float)
    y = np.asarray(proxy_scores, dtype=float)
    per_item_std = _repeat_std_vector(repeat_scores_by_item, x.shape[0], item_ids)
    info_lines = _format_info_lines(
        result, y, x, symbol=symbol, corr_method=corr_method,
        show_calibration_gap=show_calibration_gap, n_boot=n_boot, seed=seed,
    )
    title = f"{label} — item-level alignment (n={result.n_items})" if label else (
        f"Item-level alignment (n={result.n_items})"
    )
    if ax is not None:
        return _render_single_panel(
            ax, x, y, per_item_std, info_lines=info_lines, title=title,
            x_label=x_label, y_label=y_label,
            show_density_shading=show_density_shading, show_diagonal=show_diagonal,
            legend_loc=legend_loc, seed=seed, out=out,
        )
    return _render_joint_panel(
        x, y, per_item_std, info_lines=info_lines, title=title,
        x_label=x_label, y_label=y_label,
        show_density_shading=show_density_shading, show_diagonal=show_diagonal,
        show_marginal_kde=show_marginal_kde, seed=seed, out=out,
    )


def _save(fig: plt.Figure, out: Union[str, Path]) -> None:
    # `out` is a path *stem* (no extension) -- append, don't use with_suffix():
    # with_suffix() replaces everything after the LAST dot in the stem, which
    # mangles any stem containing a dot (e.g. a model name like "gpt-5.3-codex").
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight": a long `label` (e.g. a full judge_id) can make the
    # title wider than the figure canvas; savefig crops to the canvas by
    # default instead of expanding it, silently clipping the overflow.
    fig.savefig(f"{out_path}.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(
        f"{out_path}.pdf",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        metadata={"Creator": "ultimate-utils", "CreationDate": None, "ModDate": None},
    )


def _discrete_bin_edges_test() -> None:
    """Regression test: near-continuous data (many distinct values close together)
    must fall back to a regular grid, not one bin per value (which renders as a
    solid smear -- this happened for real with a hash-based 'random' judge whose
    72 items are all slightly different floats)."""
    rng = np.random.default_rng(0)
    near_continuous = rng.uniform(0, 1, size=72)
    edges = _discrete_bin_edges(near_continuous)
    assert len(edges) - 1 <= 25, f"expected a capped bin count, got {len(edges) - 1} bins"

    discrete = np.array([0.25, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    edges_d = _discrete_bin_edges(discrete)
    assert len(edges_d) - 1 == len(discrete), f"expected one bin per unique value, got {len(edges_d) - 1}"
    print("_discrete_bin_edges_test passed")


def plot_validity_stability_scatter_test() -> None:
    """Headless smoke test: builds a full joint-plot figure (density wash + marginal KDEs) from synthetic data."""
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    target = rng.uniform(0, 1, size=40)
    proxy = np.clip(target + rng.normal(0, 0.1, size=40), 0, 1)
    repeats = {
        i: [
            v,
            float(np.clip(v + rng.normal(0, 0.02), 0, 1)),
            float(np.clip(v + rng.normal(0, 0.02), 0, 1)),
        ]
        for i, v in enumerate(proxy)
    }
    fig = plot_validity_stability_scatter(proxy, target, repeats, label="synthetic demo")
    assert fig is not None
    assert len(fig.axes) == 4  # central + top marginal + right marginal + info
    assert any("error bar = repeat SD" in text.get_text() for text in fig.axes[0].texts)
    plt.close(fig)
    print("plot_validity_stability_scatter_test passed")


def plot_validity_stability_scatter_ax_mode_test() -> None:
    """ax= composition still works (simplified single-panel path)."""
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(1)
    target = rng.uniform(0, 1, size=30)
    proxy = np.clip(target + rng.normal(0, 0.1, size=30), 0, 1)
    fig, ax = plt.subplots()
    result_fig = plot_validity_stability_scatter(proxy, target, std=0.05, ax=ax)
    assert result_fig is fig
    assert len(fig.axes) == 1
    plt.close(fig)
    print("plot_validity_stability_scatter_ax_mode_test passed")


def plot_validity_stability_scatter_explicit_std_has_no_errorbar_caption_test() -> None:
    """An aggregate explicit std informs VS but cannot support item-level error bars."""
    import matplotlib

    matplotlib.use("Agg")
    target = np.linspace(0.1, 0.9, 12)
    proxy = target.copy()
    fig = plot_validity_stability_scatter(proxy, target, std=0.05)
    assert not any("error bar" in text.get_text() for text in fig.axes[0].texts)
    plt.close(fig)
    print("plot_validity_stability_scatter_explicit_std_has_no_errorbar_caption_test passed")


def plot_validity_stability_scatter_saved_bytes_are_deterministic_test() -> None:
    """Regression test for volatile PDF CreationDate/ModDate metadata."""
    import hashlib
    import tempfile

    target = np.linspace(0.1, 0.9, 12)
    proxy = target.copy()
    with tempfile.TemporaryDirectory() as tmpdir:
        hashes = []
        for suffix in ("a", "b"):
            out = Path(tmpdir) / suffix
            fig = plot_validity_stability_scatter(proxy, target, std=0.05, out=out, seed=7)
            plt.close(fig)
            hashes.append(
                (
                    hashlib.sha256(Path(f"{out}.png").read_bytes()).hexdigest(),
                    hashlib.sha256(Path(f"{out}.pdf").read_bytes()).hexdigest(),
                )
            )
        assert hashes[0] == hashes[1], hashes
    print("plot_validity_stability_scatter_saved_bytes_are_deterministic_test passed")


if __name__ == "__main__":
    _discrete_bin_edges_test()
    plot_validity_stability_scatter_test()
    plot_validity_stability_scatter_ax_mode_test()
    plot_validity_stability_scatter_explicit_std_has_no_errorbar_caption_test()
    plot_validity_stability_scatter_saved_bytes_are_deterministic_test()
    print("Done, success! \a")
