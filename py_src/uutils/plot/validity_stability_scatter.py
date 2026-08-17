"""
Validity-Stability scatter: proxy-vs-target per-item plot with a light 2D
density wash and the VS-score formula (plugged in with real numbers) as the
legend -- a drop-in replacement for "scatter plot + bare correlation in the
title" whenever the proxy's own repeat-stability matters to whether you'd
trust it, merged into a single, uncrowded panel rather than separate marginal
histogram subplots.

See uutils.stats_uu.validity_stability for the score this figure visualizes
and for why the stability term must come from repeated measurements of the
proxy, not from the spread of its scores across items.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from uutils.stats_uu.validity_stability import compute_validity_stability

try:
    from scipy.stats import gaussian_kde

    _HAVE_SCIPY_KDE = True
except ImportError:  # pragma: no cover - scipy is a declared dependency
    _HAVE_SCIPY_KDE = False


# Validated sequential-blue ramp + chart chrome (Claude Code dataviz-skill
# reference palette: colorblind-checked, see that skill's references/palette.md).
_SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]
_POINT_COLOR = "#2a78d6"  # categorical slot 1
_DIAGONAL_COLOR = "#c3c2b7"  # baseline / axis
_GRID_COLOR = "#e1e0d9"  # hairline gridline
_TEXT_MUTED = "#898781"
_TEXT_PRIMARY = "#0b0b0b"
_SURFACE = "#fcfcfb"


def _sequential_blue_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("vs_seq_blue", _SEQ_BLUE, N=256)


def plot_validity_stability_scatter(
    proxy_scores: Sequence[float],
    target_scores: Sequence[float],
    repeat_scores_by_item: Optional[Mapping[object, Sequence[float]]] = None,
    *,
    std: Optional[float] = None,
    corr_method: str = "spearman",
    symbol: str = "VS",
    label: str = "",
    x_label: str = "Target score",
    y_label: str = "Proxy score",
    show_density: bool = True,
    show_diagonal: bool = True,
    ax: Optional[plt.Axes] = None,
    legend_loc: str = "best",
    seed: int = 0,
    out: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    One panel: item-level (target, proxy) score scatter with a light 2D
    density wash and the Validity-Stability formula -- plugged in with this
    proxy's actual numbers -- as the legend.

    Deterministic: `seed` only perturbs the KDE fit by float noise (1e-6) to
    avoid a singular covariance matrix on near-duplicate points; identical
    inputs + seed produce byte-identical figures. Pass `ax` to compose this
    panel into a larger figure; otherwise a new figure is created. Pass
    `symbol` to relabel the legend for a domain-specific instantiation (e.g.
    cert-judge's TI_2') without changing the underlying math.
    """
    result = compute_validity_stability(
        proxy_scores, target_scores, repeat_scores_by_item, std=std, corr_method=corr_method
    )
    x = np.asarray(target_scores, dtype=float)
    y = np.asarray(proxy_scores, dtype=float)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
    else:
        fig = ax.figure

    ax.set_facecolor(_SURFACE)

    if show_density and _HAVE_SCIPY_KDE and result.n_items >= 5:
        try:
            rng = np.random.default_rng(seed)
            jx = x + rng.normal(scale=1e-6, size=x.shape)
            jy = y + rng.normal(scale=1e-6, size=y.shape)
            kde = gaussian_kde(np.vstack([jx, jy]))
            pad = 0.06
            gx, gy = np.mgrid[-pad : 1 + pad : 120j, -pad : 1 + pad : 120j]
            density = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            ax.contourf(gx, gy, density, levels=12, cmap=_sequential_blue_cmap(), alpha=0.55, zorder=1)
        except Exception:
            pass  # cosmetic-only background wash; a degenerate point cloud just skips it

    if show_diagonal:
        ax.plot(
            [-0.05, 1.05], [-0.05, 1.05], "--", color=_DIAGONAL_COLOR, linewidth=1.1,
            alpha=0.9, zorder=2, label="perfect agreement (y = x)",
        )

    ax.scatter(
        x, y, s=60, color=_POINT_COLOR, edgecolor="white", linewidth=0.6, alpha=0.9,
        zorder=3, label=result.formula_str(symbol=symbol),
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(x_label, color=_TEXT_PRIMARY)
    ax.set_ylabel(y_label, color=_TEXT_PRIMARY)
    title = f"{label} — item-level alignment (n={result.n_items})" if label else (
        f"Item-level alignment (n={result.n_items})"
    )
    ax.set_title(title, color=_TEXT_PRIMARY, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8, alpha=0.9, zorder=0)
    ax.tick_params(colors=_TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)

    legend = ax.legend(
        loc=legend_loc, frameon=True, framealpha=0.95, facecolor="white",
        edgecolor="#cfcfcf", fontsize=9, borderpad=0.5,
    )
    legend.set_zorder(5)

    if owns_fig:
        fig.tight_layout()

    if out is not None:
        # `out` is a path *stem* (no extension) -- append, don't use with_suffix():
        # with_suffix() replaces everything after the LAST dot in the stem, which
        # mangles any stem containing a dot (e.g. a model name like "gpt-5.3-codex").
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # bbox_inches="tight": a long `label` (e.g. a full judge_id) can make the
        # title wider than the figure canvas; savefig crops to the canvas by
        # default instead of expanding it, silently clipping the overflow.
        fig.savefig(f"{out_path}.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        fig.savefig(f"{out_path}.pdf", facecolor=fig.get_facecolor(), bbox_inches="tight")

    return fig


def plot_validity_stability_scatter_test() -> None:
    """Headless smoke test: builds a figure from synthetic data on the Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    target = rng.uniform(0, 1, size=40)
    proxy = np.clip(target + rng.normal(0, 0.1, size=40), 0, 1)
    repeats = {i: [v, v + rng.normal(0, 0.02), v + rng.normal(0, 0.02)] for i, v in enumerate(proxy)}
    fig = plot_validity_stability_scatter(proxy, target, repeats, label="synthetic demo")
    assert fig is not None
    plt.close(fig)
    print("plot_validity_stability_scatter_test passed")


if __name__ == "__main__":
    plot_validity_stability_scatter_test()
    print("Done, success! \a")
