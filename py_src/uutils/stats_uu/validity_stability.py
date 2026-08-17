"""
Validity-Stability score (VS): a single-number summary of "how much should I
trust a cheap proxy signal as a stand-in for an expensive target signal?"

Two operators on a pair of paired vectors (D1, D2), D1 the proxy side:

    M^C(D1, D2)  = clip_[0,1](Corr(D1, D2))                      -- validity alone
    M^VS(D1, D2) = ( M^C(D1, D2) * (1 - STD(D1)) )^0.5            -- validity + stability

Two textbook psychometric ingredients, combined by geometric mean so either
one being ~0 collapses the score to ~0 (a soft AND, not an average):
    - validity:  does the proxy track the target?    Corr(D1, D2), items
      paired 1:1 (same index set on both sides).
    - stability: is the proxy internally consistent?  1 - STD(D1), where STD
      comes from REPEATED measurements of the proxy (D1) on the same item --
      NOT the spread of the proxy's scores across different items. A proxy
      that outputs the same score for every item regardless of quality is
      uninformative, and must not be *rewarded* by (1 - STD) being large, so
      STD has to come from repeats (or an explicit instability figure you
      already trust), never from across-item spread.

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
M^VS(judge, human); see that repo's ti2prime_plot.py) -- but both operators
are generic: any time you have (a) a cheap/automatic signal paired with an
expensive/gold signal on the same items, this applies, and M^VS further
applies whenever you also have (b) repeated measurements of the cheap
signal. M^VS is effectively a drop-in replacement for reporting a bare
correlation coefficient whenever the cheap signal's own repeat-noise matters
to whether you'd trust it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr, pearsonr

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a declared dependency
    _HAVE_SCIPY = False


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _compute_clipped_corr(
    d1: Sequence[float], d2: Sequence[float], corr_method: str
) -> Tuple[float, float, int]:
    """Shared by compute_validity and compute_validity_stability. Returns (validity, validity_raw, n)."""
    x = np.asarray(d1, dtype=float)
    y = np.asarray(d2, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"d1 and d2 must be paired 1:1, got shapes {x.shape} vs {y.shape}")
    n = x.shape[0]
    if n < 3:
        raise ValueError(f"need >= 3 paired items to compute a correlation, got {n}")
    if corr_method not in ("spearman", "pearson"):
        raise ValueError(f"unknown corr_method {corr_method!r}, expected 'spearman' or 'pearson'")
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required to compute a correlation")
    stat_fn = spearmanr if corr_method == "spearman" else pearsonr
    validity_raw = float(stat_fn(x, y).statistic)
    if not np.isfinite(validity_raw):
        validity_raw = 0.0
    return _clip01(validity_raw), validity_raw, n


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
    std: Optional[float] = None,
    corr_method: str = "spearman",
) -> ValidityStabilityResult:
    """
    VS = M^VS(proxy_scores, target_scores)
       = ( clip_[0,1](Corr(proxy_scores, target_scores)) * (1 - STD) )^0.5.

    `proxy_scores`/`target_scores` are paired per item (same order, same
    length), both in [0,1]; `Corr` is computed over those items.

    The stability term must be supplied explicitly -- there is no default,
    because guessing wrong silently reverses the sign of what gets rewarded
    (see module docstring):
      - `repeat_scores_by_item`: item -> that item's raw repeated proxy
        measurements; STD is the pooled repeat std,
        sqrt(mean_i(var(repeat_scores_i))).
      - `std`: a precomputed instability figure, used as-is.

    If you don't have (or don't want) a stability term, use `compute_validity`
    (M^C) instead -- do not fake this one with a made-up std.
    """
    validity, validity_raw, n = _compute_clipped_corr(proxy_scores, target_scores, corr_method)

    if std is not None:
        std_val = float(std)
        stability_source = "explicit"
    elif repeat_scores_by_item is not None:
        item_vars = [
            float(np.var(np.asarray(v, dtype=float)))
            for v in repeat_scores_by_item.values()
            if len(v) > 1
        ]
        if not item_vars:
            raise ValueError(
                "repeat_scores_by_item had no item with >= 2 repeats to estimate stability"
            )
        std_val = float(np.sqrt(np.mean(item_vars)))
        stability_source = "repeat_scores"
    else:
        raise ValueError(
            "compute_validity_stability needs a stability term: pass `std` (precomputed) "
            "or `repeat_scores_by_item` (raw per-item repeat measurements). Falling back to "
            "the across-item std of proxy_scores would reward a proxy that never varies its "
            "score regardless of item quality -- see module docstring."
        )
    std_val = _clip01(std_val)
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


if __name__ == "__main__":
    compute_validity_test()
    compute_validity_matches_validity_stability_test()
    compute_vs_score_test()
    compute_vs_score_explicit_std_test()
    compute_vs_score_missing_stability_raises_test()
    print("Done, success! \a")
