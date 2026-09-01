"""Fail-closed tests for the public validity/stability score and plot APIs."""

from __future__ import annotations

import hashlib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uutils.plot.validity_stability_scatter import plot_validity_stability_scatter
from uutils.stats_uu.validity_stability import (
    _resolve_repeat_item_ids,
    bootstrap_correlation_ci,
    compute_validity,
    compute_validity_stability,
)


PROXY = [0.1, 0.3, 0.5, 0.7, 0.9]
TARGET = list(PROXY)


@pytest.mark.parametrize("std", [-0.1, 1.1, np.nan, np.inf])
def test_explicit_instability_must_be_finite_and_bounded(std):
    with pytest.raises(ValueError, match="std must be finite"):
        compute_validity_stability(PROXY, TARGET, std=std)


def test_stability_sources_are_mutually_exclusive():
    repeats = {i: [score, score] for i, score in enumerate(PROXY)}
    with pytest.raises(ValueError, match="exactly one stability source"):
        compute_validity_stability(PROXY, TARGET, repeats, std=0.1)


def test_repeat_mapping_is_aligned_by_identity_not_insertion_order():
    reversed_positional = {
        4: [0.5, 0.9],
        3: [0.55, 0.85],
        2: [0.4, 0.6],
        1: [0.25, 0.35],
        0: [0.1, 0.1],
    }
    assert _resolve_repeat_item_ids(reversed_positional, len(PROXY), None) == [0, 1, 2, 3, 4]

    named = {f"item-{i}": values for i, values in reversed_positional.items()}
    ordered_ids = [f"item-{i}" for i in range(len(PROXY))]
    assert _resolve_repeat_item_ids(named, len(PROXY), ordered_ids) == ordered_ids
    with pytest.raises(ValueError, match="require `item_ids`"):
        compute_validity_stability(PROXY, TARGET, named)
    result = compute_validity_stability(PROXY, TARGET, named, item_ids=ordered_ids)
    assert result.n_items == len(PROXY)


@pytest.mark.parametrize(
    "repeats",
    [
        {0: [0.1, 0.1]},
        {0: [0.1], 1: [0.3, 0.3], 2: [0.5, 0.5], 3: [0.7, 0.7], 4: [0.9, 0.9]},
        {0: [0.1, np.nan], 1: [0.3, 0.3], 2: [0.5, 0.5], 3: [0.7, 0.7], 4: [0.9, 0.9]},
    ],
)
def test_repeat_stability_requires_complete_finite_coverage(repeats):
    with pytest.raises(ValueError):
        compute_validity_stability(PROXY, TARGET, repeats)


def test_vs_scores_must_be_in_unit_interval():
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        compute_validity_stability([0.1, 0.3, 1.2, 0.7, 0.9], TARGET, std=0.1)
    with pytest.raises(ValueError, match="finite"):
        compute_validity_stability([0.1, 0.3, np.nan, 0.7, 0.9], TARGET, std=0.1)


def test_maximally_dispersed_bounded_repeats_are_a_half_stability_penalty():
    repeats = {i: [0.0, 1.0] for i in range(len(PROXY))}
    result = compute_validity_stability(PROXY, TARGET, repeats)
    assert result.stability == pytest.approx(0.5)
    assert result.vs_score == pytest.approx(np.sqrt(0.5))


def test_constant_vectors_fail_closed_to_zero_validity():
    result = compute_validity([0.2] * 5, [0.8] * 5)
    assert result.validity_raw == 0.0
    assert result.validity == 0.0


@pytest.mark.parametrize("kwargs", [{"corr_method": "kendall"}, {"n_boot": 0}, {"n_boot": True}])
def test_bootstrap_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        bootstrap_correlation_ci(PROXY, TARGET, **kwargs)


def test_explicit_std_plot_does_not_claim_item_error_bars():
    fig = plot_validity_stability_scatter(PROXY, TARGET, std=0.1, n_boot=20)
    assert not any("error bar" in text.get_text() for text in fig.axes[0].texts)
    plt.close(fig)


def test_saved_pdf_and_png_are_byte_deterministic(tmp_path):
    hashes = []
    for stem in (tmp_path / "a", tmp_path / "b"):
        fig = plot_validity_stability_scatter(PROXY, TARGET, std=0.1, n_boot=20, seed=7, out=stem)
        plt.close(fig)
        hashes.append(
            (
                hashlib.sha256(stem.with_suffix(".png").read_bytes()).hexdigest(),
                hashlib.sha256(stem.with_suffix(".pdf").read_bytes()).hexdigest(),
            )
        )
    assert hashes[0] == hashes[1]
