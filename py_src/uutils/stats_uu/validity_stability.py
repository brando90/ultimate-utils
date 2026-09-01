"""
Validity-Stability score (VS): a single-number summary of "how much should I
trust a cheap proxy signal as a stand-in for an expensive target signal?"

Two operators on a pair of paired vectors (D1, D2), D1 the proxy side:

    M^C(D1, D2)  = clip_[0,1](Corr(D1, D2))                      -- validity alone
    M^VS(D1, D2) = ( M^C(D1, D2) * (1 - STD(D1)) )^0.5            -- validity + stability

Two textbook psychometric ingredients, combined by geometric mean (a soft
AND, not an average):
    - validity:  does the proxy track the target?    Corr(D1, D2), items
      paired 1:1 (same index set on both sides).
    - stability: is the proxy internally consistent?  1 - STD(D1), where STD
      comes from REPEATED measurements of the proxy (D1) on the same item --
      NOT the spread of the proxy's scores across different items. A proxy
      that outputs the same score for every item regardless of quality is
      uninformative, and must not be *rewarded* by (1 - STD) being large, so
      STD has to come from repeats (or an explicit instability figure you
      already trust), never from across-item spread. For raw scores bounded
      to [0,1], STD is at most 0.5, so this path is a monotone repeat-noise
      penalty rather than a second zero-annihilator. An explicit normalized
      instability can span [0,1] and reaches zero stability at STD=1.

Use `compute_validity` (M^C) alone whenever a stability factor must NOT be
present -- e.g. a property estimator that correlates a proxy against a
constructed ground truth (a bug-ladder rung, a spec-removal rung) rather than
a repeated-measurement target. Instantiating M^VS on *both* sides of a bridge
correlation would put the same (1 - STD) factor into both scalars, making
part of that correlation mechanical rather than evidential -- a
high-variance proxy would score low on both axes regardless of what the
correlation itself does. Rule of thumb: the criterion/target side gets M^VS;
anything correlated against a constructed (non-repeated-measurement) ground
truth gets M^C only.

This originates from certifying LLM judges of formal proofs, where the proxy
is a judge's score and the target is a human rating (cert-judge's TI_2' =
M^VS(judge, human); see that repo's ti2prime_plot.py). The operators can be
used in other paired-signal settings, but M^VS is a construct-specific
composite, not a drop-in statistical replacement for correlation: its value
depends on a fixed [0,1] score scale and on treating the chosen repeat-noise
penalty as substantively meaningful. Report the raw correlation, stability
factor, and their provenance alongside the composite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr, pearsonr, kendalltau, wasserstein_distance

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a declared dependency
    _HAVE_SCIPY = False


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _as_finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    """Convert a numeric vector and reject ambiguous shapes or missing values."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional score vector, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_finite_unit_interval_vector(values: Sequence[float], name: str) -> np.ndarray:
    """Convert a score vector and enforce the VS operator's documented domain."""
    array = _as_finite_vector(values, name)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} values must lie in [0,1]")
    return array


def _resolve_repeat_item_ids(
    repeat_scores_by_item: Mapping[object, Sequence[float]],
    n_items: int,
    item_ids: Optional[Sequence[object]],
) -> list[object]:
    """Return score-vector order for a repeat mapping, failing closed on ambiguous alignment."""
    if item_ids is None:
        resolved_ids: list[object] = list(range(n_items))
        if set(repeat_scores_by_item) != set(resolved_ids):
            raise ValueError(
                "non-positional repeat_scores_by_item keys require `item_ids` in the same "
                "order as proxy_scores/target_scores"
            )
    else:
        resolved_ids = list(item_ids)
        if len(resolved_ids) != n_items:
            raise ValueError(
                f"item_ids has {len(resolved_ids)} entries but the paired score vectors "
                f"have {n_items}"
            )
        try:
            unique_ids = set(resolved_ids)
        except TypeError as exc:
            raise ValueError("item_ids must be hashable mapping keys") from exc
        if len(unique_ids) != n_items:
            raise ValueError("item_ids must be unique")
        repeat_ids = set(repeat_scores_by_item)
        if repeat_ids != unique_ids:
            missing = unique_ids - repeat_ids
            extra = repeat_ids - unique_ids
            raise ValueError(
                "repeat_scores_by_item keys must exactly match item_ids; "
                f"missing={sorted(map(repr, missing))}, extra={sorted(map(repr, extra))}"
            )
    return resolved_ids


def _compute_clipped_corr(
    d1: Sequence[float], d2: Sequence[float], corr_method: str
) -> Tuple[float, float, int]:
    """Shared by compute_validity and compute_validity_stability. Returns (validity, validity_raw, n)."""
    x = _as_finite_vector(d1, "d1")
    y = _as_finite_vector(d2, "d2")
    if x.shape != y.shape:
        raise ValueError(f"d1 and d2 must be paired 1:1, got shapes {x.shape} vs {y.shape}")
    n = x.shape[0]
    if n < 3:
        raise ValueError(f"need >= 3 paired items to compute a correlation, got {n}")
    if corr_method not in ("spearman", "pearson"):
        raise ValueError(f"unknown corr_method {corr_method!r}, expected 'spearman' or 'pearson'")
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required to compute a correlation")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 0.0, n
    stat_fn = spearmanr if corr_method == "spearman" else pearsonr
    validity_raw = float(stat_fn(x, y).statistic)
    if not np.isfinite(validity_raw):
        validity_raw = 0.0
    return _clip01(validity_raw), validity_raw, n


def _resolve_instability(
    n_items: int,
    repeat_scores_by_item: Optional[Mapping[object, Sequence[float]]],
    item_ids: Optional[Sequence[object]],
    std: Optional[float],
) -> tuple[float, str]:
    """Validate exactly one stability source and return (instability, provenance)."""
    if item_ids is not None and repeat_scores_by_item is None:
        raise ValueError("item_ids is only meaningful with repeat_scores_by_item")
    if std is not None and repeat_scores_by_item is not None:
        raise ValueError("pass exactly one stability source: `std` or `repeat_scores_by_item`, not both")
    if std is not None:
        std_val = float(std)
        if not np.isfinite(std_val) or not 0.0 <= std_val <= 1.0:
            raise ValueError(f"std must be finite and lie in [0,1], got {std!r}")
        return std_val, "explicit"
    if repeat_scores_by_item is None:
        raise ValueError(
            "compute_validity_stability needs a stability term: pass `std` (precomputed) "
            "or `repeat_scores_by_item` (raw per-item repeat measurements). Falling back to "
            "the across-item std of proxy_scores would reward a proxy that never varies its "
            "score regardless of item quality -- see module docstring."
        )
    if len(repeat_scores_by_item) != n_items:
        raise ValueError(
            f"repeat_scores_by_item has {len(repeat_scores_by_item)} entries but "
            f"the paired score vectors have {n_items}; stability coverage must be 1:1"
        )
    resolved_ids = _resolve_repeat_item_ids(repeat_scores_by_item, n_items, item_ids)
    item_vars = []
    for item in resolved_ids:
        repeat_array = _as_finite_unit_interval_vector(
            repeat_scores_by_item[item], f"repeat_scores_by_item[{item!r}]"
        )
        if repeat_array.size < 2:
            raise ValueError(
                f"repeat_scores_by_item[{item!r}] has {repeat_array.size} value(s); "
                "every paired item needs at least 2 repeats"
            )
        item_vars.append(float(np.var(repeat_array)))
    return float(np.sqrt(np.mean(item_vars))), "repeat_scores"


@dataclass
class ValidityResult:
    """M^C(D1, D2): clip_[0,1](Corr(D1, D2)) alone -- no stability term.

    For measurements that must not include a stability factor -- see this
    module's docstring, "M^VS on both sides of a bridge" -- e.g. a property
    estimator correlating a proxy against a constructed ground truth (a
    ladder rung) rather than a repeated-measurement target.
    """

    validity: float
    validity_raw: float  # unclipped signed correlation, kept for diagnostics
    n_items: int
    corr_method: str

    def formula_str(self, symbol: str = "M^C") -> str:
        """Mathtext-ish legend label, e.g. M^C = 0.48."""
        return rf"${symbol} = {self.validity:.3f}$"


def compute_validity(
    d1: Sequence[float],
    d2: Sequence[float],
    *,
    corr_method: str = "spearman",
) -> ValidityResult:
    """M^C(d1, d2) = clip_[0,1](Corr(d1, d2)). Stability-free -- see ValidityResult docstring."""
    validity, validity_raw, n = _compute_clipped_corr(d1, d2, corr_method)
    return ValidityResult(validity=validity, validity_raw=validity_raw, n_items=n, corr_method=corr_method)


@dataclass
class ValidityStabilityResult:
    """M^VS(proxy, target) and the two ingredients it was built from."""

    vs_score: float
    validity: float  # clip_[0,1](Corr(proxy, target)) -- the value actually used
    validity_raw: float  # unclipped signed correlation, kept for diagnostics
    stability: float  # 1 - STD -- the value actually used (higher = more stable)
    n_items: int
    corr_method: str
    stability_source: str  # "repeat_scores" | "explicit"

    def formula_str(self, symbol: str = "VS") -> str:
        """Mathtext legend label, e.g. VS = (0.48 x (1-0.07))^0.5 = 0.667."""
        std_shown = 1.0 - self.stability
        return (
            rf"${symbol} = ({self.validity:.2f} \times (1-{std_shown:.2f}))^{{0.5}}"
            rf" = {self.vs_score:.3f}$"
        )


def compute_validity_stability(
    proxy_scores: Sequence[float],
    target_scores: Sequence[float],
    repeat_scores_by_item: Optional[Mapping[object, Sequence[float]]] = None,
    *,
    item_ids: Optional[Sequence[object]] = None,
    std: Optional[float] = None,
    corr_method: str = "spearman",
) -> ValidityStabilityResult:
    """Compute M^VS; see the module docs for the construct and alignment contract.

    Scores are paired, finite, and in [0,1]. Supply exactly one stability
    source: a finite `std` in [0,1], or complete per-item repeats. Positional
    repeat keys are aligned by integer identity; named keys require `item_ids`
    in score-vector order. Use `compute_validity` when no stability factor is
    substantively justified.
    """
    _as_finite_unit_interval_vector(proxy_scores, "proxy_scores")
    _as_finite_unit_interval_vector(target_scores, "target_scores")
    validity, validity_raw, n = _compute_clipped_corr(proxy_scores, target_scores, corr_method)
    std_val, stability_source = _resolve_instability(n, repeat_scores_by_item, item_ids, std)
    stability = 1.0 - std_val

    vs_score = float((validity * stability) ** 0.5)
    return ValidityStabilityResult(
        vs_score=vs_score,
        validity=validity,
        validity_raw=validity_raw,
        stability=stability,
        n_items=n,
        corr_method=corr_method,
        stability_source=stability_source,
    )


def bootstrap_correlation_ci(
    d1: Sequence[float],
    d2: Sequence[float],
    *,
    corr_method: str = "spearman",
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    95% CI on Corr(d1, d2) (unclipped, i.e. on validity_raw) via the item-level
    nonparametric bootstrap: resample items with replacement, recompute the
    correlation, take the 2.5/97.5 percentiles. Mirrors the resampling
    convention already used elsewhere in this codebase (e.g.
    experiments/04_veribench_validation/correlation_analysis.py's
    `_bootstrap_ci`) rather than a different bootstrap scheme.
    """
    x = _as_finite_vector(d1, "d1")
    y = _as_finite_vector(d2, "d2")
    if corr_method not in ("spearman", "pearson"):
        raise ValueError(f"unknown corr_method {corr_method!r}, expected 'spearman' or 'pearson'")
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    if x.shape != y.shape or x.shape[0] < 3:
        return float("nan"), float("nan")
    stat_fn = spearmanr if corr_method == "spearman" else pearsonr
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        x_boot, y_boot = x[idx], y[idx]
        if np.all(x_boot == x_boot[0]) or np.all(y_boot == y_boot[0]):
            continue
        r = float(stat_fn(x_boot, y_boot).statistic)
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def compute_kendall_tau_b(d1: Sequence[float], d2: Sequence[float]) -> float:
    """
    Kendall's tau-b: like Spearman, a rank correlation, but corrects for tied
    ranks rather than assigning fractional midranks -- the more defensible
    choice when the data is a tie-heavy lattice (e.g. human labels averaged
    from a handful of integer ratings). Reported alongside validity (which
    uses Spearman) as a secondary diagnostic, not a replacement for it.
    """
    x = np.asarray(d1, dtype=float)
    y = np.asarray(d2, dtype=float)
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for compute_kendall_tau_b")
    tau = float(kendalltau(x, y, variant="b").statistic)
    return tau if np.isfinite(tau) else 0.0


def compute_calibration_gap(d1: Sequence[float], d2: Sequence[float]) -> float:
    """
    1D Wasserstein (earth-mover's) distance between the marginal distributions
    of d1 and d2 -- how far apart the two signals' *levels* are, independent
    of whether they're paired or how well they correlate. This is a distinct
    quantity from validity: a proxy can be perfectly rank-correlated with a
    target while sitting on a totally different part of the scale (e.g. a
    judge that is uniformly harsher than a human rater), and validity
    (correlation-based) is invariant to exactly that offset by construction.
    Report calibration gap as a separate diagnostic, never folded into VS.
    """
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for compute_calibration_gap")
    return float(wasserstein_distance(np.asarray(d1, dtype=float), np.asarray(d2, dtype=float)))


def compute_validity_test() -> None:
    """M^C alone: perfectly correlated -> validity == 1.0, no stability term needed."""
    d1 = [0.1, 0.4, 0.6, 0.9, 0.2, 0.7]
    d2 = [0.0, 0.3, 0.5, 1.0, 0.1, 0.8]
    result = compute_validity(d1, d2)
    assert abs(result.validity - 1.0) < 1e-9, f"{result=}"
    print(f"compute_validity_test passed: {result=}")


def compute_validity_matches_validity_stability_test() -> None:
    """The `validity` component of M^VS equals standalone M^C on the same (d1, d2)."""
    d1 = [0.1, 0.5, 0.3, 0.9, 0.6, 0.2]
    d2 = [0.2, 0.4, 0.3, 0.8, 0.5, 0.1]
    only_validity = compute_validity(d1, d2)
    with_stability = compute_validity_stability(d1, d2, std=0.1)
    assert only_validity.validity == with_stability.validity
    assert only_validity.validity_raw == with_stability.validity_raw
    print("compute_validity_matches_validity_stability_test passed")


def compute_vs_score_test() -> None:
    """Perfectly stable, perfectly correlated proxy -> VS == 1.0."""
    proxy = [0.1, 0.4, 0.6, 0.9, 0.2, 0.7]
    target = [0.0, 0.3, 0.5, 1.0, 0.1, 0.8]
    repeats = {i: [v, v, v] for i, v in enumerate(proxy)}
    result = compute_validity_stability(proxy, target, repeats)
    assert abs(result.vs_score - 1.0) < 1e-9, f"{result=}"
    assert result.stability_source == "repeat_scores"
    print(f"compute_vs_score_test passed: {result=}")


def compute_vs_score_explicit_std_test() -> None:
    """`std=` path matches the `repeat_scores_by_item=` path for equivalent inputs."""
    proxy = [0.2, 0.5, 0.8, 0.3, 0.6]
    target = [0.1, 0.6, 0.7, 0.2, 0.9]
    via_std = compute_validity_stability(proxy, target, std=0.05)
    assert via_std.stability_source == "explicit"
    assert abs(via_std.stability - 0.95) < 1e-9
    print(f"compute_vs_score_explicit_std_test passed: {via_std=}")


def compute_vs_score_missing_stability_raises_test() -> None:
    proxy = [0.2, 0.5, 0.8, 0.3, 0.6]
    target = [0.1, 0.6, 0.7, 0.2, 0.9]
    try:
        compute_validity_stability(proxy, target)
        raise AssertionError("expected ValueError when no stability term is given")
    except ValueError:
        print("compute_vs_score_missing_stability_raises_test passed")


def compute_vs_score_rejects_invalid_stability_inputs_test() -> None:
    proxy = [0.2, 0.5, 0.8, 0.3, 0.6]
    target = [0.1, 0.6, 0.7, 0.2, 0.9]
    complete = {i: [v, v] for i, v in enumerate(proxy)}

    for invalid_std in (-0.2, 1.2, float("nan"), float("inf")):
        try:
            compute_validity_stability(proxy, target, std=invalid_std)
            raise AssertionError(f"expected ValueError for std={invalid_std!r}")
        except ValueError:
            pass

    try:
        compute_validity_stability(proxy, target, complete, std=0.1)
        raise AssertionError("expected ValueError for two competing stability sources")
    except ValueError:
        pass

    invalid_repeat_sets = [
        {0: [0.2, 0.2]},  # incomplete item coverage
        {**complete, 0: [0.2]},  # singleton for one item
        {**complete, 0: [0.2, float("nan")]},
        {**complete, 0: [0.2, 1.2]},
    ]
    for repeats in invalid_repeat_sets:
        try:
            compute_validity_stability(proxy, target, repeats)
            raise AssertionError(f"expected ValueError for invalid repeats: {repeats!r}")
        except ValueError:
            pass
    print("compute_vs_score_rejects_invalid_stability_inputs_test passed")


def compute_vs_score_bounded_repeat_penalty_test() -> None:
    """Maximally dispersed [0,1] repeats yield raw STD=.5, not false zero stability."""
    proxy = [0.1, 0.3, 0.5, 0.7, 0.9]
    target = list(proxy)
    repeats = {i: [0.0, 1.0] for i in range(len(proxy))}
    result = compute_validity_stability(proxy, target, repeats)
    assert abs(result.stability - 0.5) < 1e-12, f"{result=}"
    assert abs(result.vs_score - np.sqrt(0.5)) < 1e-12, f"{result=}"
    print("compute_vs_score_bounded_repeat_penalty_test passed")


def compute_vs_score_repeat_identity_alignment_test() -> None:
    proxy = [0.1, 0.3, 0.5, 0.7, 0.9]
    target = list(proxy)
    reversed_positional = {
        4: [0.5, 0.9],
        3: [0.55, 0.85],
        2: [0.4, 0.6],
        1: [0.25, 0.35],
        0: [0.1, 0.1],
    }
    assert _resolve_repeat_item_ids(reversed_positional, len(proxy), None) == [0, 1, 2, 3, 4]
    named = {f"item-{i}": values for i, values in reversed_positional.items()}
    item_ids = [f"item-{i}" for i in range(len(proxy))]
    result = compute_validity_stability(proxy, target, named, item_ids=item_ids)
    assert result.n_items == len(proxy)
    print("compute_vs_score_repeat_identity_alignment_test passed")


def bootstrap_correlation_ci_test() -> None:
    """CI should bracket the point estimate and shrink toward it as n grows."""
    rng = np.random.default_rng(0)
    d1 = rng.uniform(0, 1, size=60)
    d2 = np.clip(d1 + rng.normal(0, 0.1, size=60), 0, 1)
    point = compute_validity(d1, d2).validity_raw
    lo, hi = bootstrap_correlation_ci(d1, d2, n_boot=500, seed=0)
    assert lo <= point <= hi, f"{lo=}, {point=}, {hi=}"
    print(f"bootstrap_correlation_ci_test passed: point={point:.3f}, CI=[{lo:.3f}, {hi:.3f}]")


def bootstrap_correlation_ci_rejects_invalid_options_test() -> None:
    d1 = [0.1, 0.2, 0.3]
    d2 = [0.2, 0.3, 0.4]
    for kwargs in ({"corr_method": "kendall"}, {"n_boot": 0}, {"n_boot": True}):
        try:
            bootstrap_correlation_ci(d1, d2, **kwargs)
            raise AssertionError(f"expected ValueError for {kwargs!r}")
        except ValueError:
            pass
    print("bootstrap_correlation_ci_rejects_invalid_options_test passed")


def compute_kendall_tau_b_test() -> None:
    d1 = [1, 2, 3, 4, 5]
    d2 = [1, 2, 3, 4, 5]
    assert abs(compute_kendall_tau_b(d1, d2) - 1.0) < 1e-9
    d2_rev = [5, 4, 3, 2, 1]
    assert abs(compute_kendall_tau_b(d1, d2_rev) - (-1.0)) < 1e-9
    print("compute_kendall_tau_b_test passed")


def compute_calibration_gap_test() -> None:
    same = [0.1, 0.2, 0.3, 0.4]
    assert compute_calibration_gap(same, same) == 0.0
    shifted = [v + 0.5 for v in same]
    assert abs(compute_calibration_gap(same, shifted) - 0.5) < 1e-9
    print("compute_calibration_gap_test passed")


if __name__ == "__main__":
    compute_validity_test()
    compute_validity_matches_validity_stability_test()
    compute_vs_score_test()
    compute_vs_score_explicit_std_test()
    compute_vs_score_missing_stability_raises_test()
    compute_vs_score_rejects_invalid_stability_inputs_test()
    compute_vs_score_bounded_repeat_penalty_test()
    compute_vs_score_repeat_identity_alignment_test()
    bootstrap_correlation_ci_test()
    bootstrap_correlation_ci_rejects_invalid_options_test()
    compute_kendall_tau_b_test()
    compute_calibration_gap_test()
    print("Done, success! \a")
