"""Both divergences are expectations of a pointwise discrepancy.

Predicted before the run, for p = (0.70, 0.20, 0.10) and q = (0.05, 0.90, 0.05):

    WD = 0.7 * (0.65)^2 + 0.2 * (0.70)^2 + 0.1 * (0.05)^2 = 0.394 exactly.

The enumeration E[(p(X) - q(X))^2] must match that value and the vectorized sum
to 1e-12. The same enumeration with d(u, v) = log u - log v must match
KL(p || q) = sum_a p(a) log(p(a) / q(a)). Substituting a one-hot distribution
must collapse the two expectations to (1 - q(x))^2 and -log q(x).

A Monte Carlo average of 100_000 draws, seed 0, must land within 5 population
standard errors of each exact expectation. Finite differences of the two
expectations must match 2 p (q - p) and -p / q. One sample of log loss has a
zero partial derivative off the observed symbol. The population divergences
do not.

Run from the repository root:

    python experiments/04_brier_vs_weighted_divergence/run_expectation_identity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_comparison as rc

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"
SEED = 0
N_MC = 100_000
P_SKEW = np.array([0.70, 0.20, 0.10])
Q_INTERIOR = np.array([0.05, 0.90, 0.05])
PREDICTED_WD = 0.394


def squared_difference(u: float, v: float) -> float:
    return (u - v) ** 2


def log_ratio(u: float, v: float) -> float:
    return float(np.log(u) - np.log(v))


def expectation_of(p: np.ndarray, q: np.ndarray, discrepancy) -> float:
    """E_{X ~ p}[d(p(X), q(X))], skipping atoms that p does not charge."""
    total = 0.0
    for a, mass in enumerate(np.asarray(p, dtype=float)):
        if mass == 0.0:
            continue
        total += float(mass) * float(discrepancy(mass, float(q[a])))
    return total


def population_sd(p: np.ndarray, q: np.ndarray, discrepancy) -> float:
    values = []
    for a, mass in enumerate(p):
        if mass == 0.0:
            continue
        values.append(float(discrepancy(float(mass), float(q[a]))))
    centered = np.array(values) - expectation_of(p, q, discrepancy)
    second = 0.0
    i = 0
    for a, mass in enumerate(p):
        if mass == 0.0:
            continue
        second += float(mass) * float(centered[i] ** 2)
        i += 1
    return float(np.sqrt(second))


def monte_carlo(p: np.ndarray, q: np.ndarray, discrepancy, rng: np.random.Generator) -> float:
    draws = rng.choice(p.size, size=N_MC, p=p)
    total = 0.0
    for a in draws:
        total += float(discrepancy(float(p[a]), float(q[a])))
    return total / N_MC


def finite_difference(p: np.ndarray, q: np.ndarray, objective, step: float = 1e-6) -> np.ndarray:
    gradient = np.zeros_like(q)
    for a in range(q.size):
        up = q.copy()
        down = q.copy()
        up[a] += step
        down[a] -= step
        gradient[a] = (objective(p, up) - objective(p, down)) / (2.0 * step)
    return gradient


def one_sample_partials(q: np.ndarray, x: int, score, step: float = 1e-6) -> np.ndarray:
    gradient = np.zeros_like(q)
    for a in range(q.size):
        up = q.copy()
        down = q.copy()
        up[a] += step
        down[a] -= step
        gradient[a] = (score(up, x) - score(down, x)) / (2.0 * step)
    return gradient


def check_pair(p: np.ndarray, q: np.ndarray, rng: np.random.Generator) -> dict:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    wd_enum = expectation_of(p, q, squared_difference)
    kl_enum = expectation_of(p, q, log_ratio)
    wd_sum = rc.weighted_divergence(p, q)
    kl_sum = float(np.sum(p * np.log(p / q)))
    if abs(wd_enum - wd_sum) > 1e-12:
        raise AssertionError(f"WD enumeration {wd_enum} != sum {wd_sum}")
    if abs(kl_enum - kl_sum) > 1e-12:
        raise AssertionError(f"KL enumeration {kl_enum} != sum {kl_sum}")
    entropy = float(-np.sum(p * np.log(p)))
    log_loss = rc.expected_log_loss(p, q)
    if abs(log_loss - (kl_sum + entropy)) > 1e-12:
        raise AssertionError("expected log loss is not KL plus entropy")

    monte_carlo_rows = {}
    for name, discrepancy, exact in (
        ("weighted_squared_divergence", squared_difference, wd_sum),
        ("kl", log_ratio, kl_sum),
    ):
        sd = population_sd(p, q, discrepancy)
        estimate = monte_carlo(p, q, discrepancy, rng)
        allowance = 5.0 * sd / np.sqrt(N_MC)
        if abs(estimate - exact) > allowance:
            raise AssertionError(f"{name} Monte Carlo {estimate} outside {allowance} of {exact}")
        monte_carlo_rows[name] = {
            "estimate": estimate,
            "exact": exact,
            "absolute_error": abs(estimate - exact),
            "population_sd": sd,
            "allowance_5_se": allowance,
        }

    one_hot_rows = []
    for x in range(p.size):
        one_hot = np.zeros_like(p)
        one_hot[x] = 1.0
        wd_one = expectation_of(one_hot, q, squared_difference)
        kl_one = expectation_of(one_hot, q, log_ratio)
        local_square = (1.0 - q[x]) ** 2
        log_loss_x = float(-np.log(q[x]))
        if abs(wd_one - local_square) > 1e-12:
            raise AssertionError("one-hot WD did not collapse to the local square score")
        if abs(kl_one - log_loss_x) > 1e-12:
            raise AssertionError("one-hot KL did not collapse to log loss")
        one_hot_rows.append(
            {
                "x": x,
                "weighted_squared_divergence": wd_one,
                "kl": kl_one,
            }
        )

    grad_wd = finite_difference(p, q, rc.weighted_divergence)
    grad_kl = finite_difference(p, q, lambda pp, qq: float(np.sum(pp * np.log(pp / qq))))
    analytic_wd = 2.0 * p * (q - p)
    analytic_kl = -p / q
    if np.max(np.abs(grad_wd - analytic_wd)) > 1e-6:
        raise AssertionError(f"WD gradient {grad_wd} != {analytic_wd}")
    if np.max(np.abs(grad_kl - analytic_kl)) > 1e-6:
        raise AssertionError(f"KL gradient {grad_kl} != {analytic_kl}")

    return {
        "p": p.tolist(),
        "q": q.tolist(),
        "weighted_squared_divergence": wd_sum,
        "kl": kl_sum,
        "entropy": entropy,
        "expected_log_loss": log_loss,
        "one_hot": one_hot_rows,
        "gradient_weighted_squared_divergence": analytic_wd.tolist(),
        "gradient_kl": analytic_kl.tolist(),
        "monte_carlo": monte_carlo_rows,
    }


def check_residual_split(q_observed: float = 0.40) -> dict:
    """Same probability on the observed symbol, two splits of the remainder."""
    q_concentrated = np.array([q_observed, 1.0 - q_observed, 0.0, 0.0])
    remainder = (1.0 - q_observed) / 3.0
    q_spread = np.array([q_observed, remainder, remainder, remainder])
    # The spread forecast is a full-support vector, so both population
    # divergences are finite. The concentrated forecast puts mass 0 on two
    # symbols that p charges, so KL diverges and WD stays finite.
    p = np.array([0.25, 0.25, 0.25, 0.25])
    x = 0
    log_loss_concentrated = float(-np.log(q_concentrated[x]))
    log_loss_spread = float(-np.log(q_spread[x]))
    local_concentrated = (1.0 - q_concentrated[x]) ** 2
    local_spread = (1.0 - q_spread[x]) ** 2
    if abs(log_loss_concentrated - log_loss_spread) > 1e-12:
        raise AssertionError("log loss saw the residual split")
    if abs(local_concentrated - local_spread) > 1e-12:
        raise AssertionError("local square score saw the residual split")
    wd_spread = rc.weighted_divergence(p, q_spread)
    kl_spread = float(np.sum(p * np.log(p / q_spread)))
    wd_concentrated = rc.weighted_divergence(p, q_concentrated)
    if not np.isfinite(wd_concentrated):
        raise AssertionError("WD diverged on a zero coordinate")
    in_support_zero = np.any((p > 0.0) & (q_concentrated == 0.0))
    if not in_support_zero:
        raise AssertionError("KL stayed finite with a zero on an in-support symbol")
    if abs(wd_spread - wd_concentrated) < 1e-12:
        raise AssertionError("WD ignored the residual split")
    return {
        "log_loss": log_loss_concentrated,
        "local_square_score": local_concentrated,
        "weighted_squared_divergence_spread": wd_spread,
        "weighted_squared_divergence_concentrated": wd_concentrated,
        "kl_spread": kl_spread,
        "kl_concentrated": "infinity",
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary = check_pair(P_SKEW, Q_INTERIOR, rng)
    if abs(primary["weighted_squared_divergence"] - PREDICTED_WD) > 1e-12:
        raise AssertionError(
            f"WD {primary['weighted_squared_divergence']} != predicted {PREDICTED_WD}"
        )
    extra = []
    for p, q in (
        (np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), np.array([0.50, 0.30, 0.20])),
        (np.array([0.80, 0.20]), np.array([0.30, 0.70])),
    ):
        extra.append(check_pair(p, q, rng))
    # A zero coordinate of q makes the log ratio infinite and the square finite.
    q_zero = np.array([0.50, 0.50, 0.0])
    wd_zero = rc.weighted_divergence(P_SKEW, q_zero)
    if not np.isfinite(wd_zero):
        raise AssertionError("WD was infinite at a zero coordinate")
    near_zero_kl = expectation_of(P_SKEW, np.array([0.50, 0.50, 1e-300]), log_ratio)
    if near_zero_kl < 10.0:
        raise AssertionError(f"KL did not grow at a near-zero coordinate: {near_zero_kl}")
    split = check_residual_split()
    observed_partial = one_sample_partials(Q_INTERIOR, 0, lambda q, x: float(-np.log(q[x])))
    if abs(observed_partial[0]) < 1.0 or np.max(np.abs(observed_partial[1:])) > 1e-8:
        raise AssertionError(f"log-loss partials leaked off the observed symbol: {observed_partial}")

    summary = {
        "seed": SEED,
        "monte_carlo_draws": N_MC,
        "predicted_weighted_squared_divergence": PREDICTED_WD,
        "primary": primary,
        "extra_pairs": extra,
        "weighted_squared_divergence_at_q3_zero": wd_zero,
        "residual_split": split,
        "one_sample_log_loss_partial_at_x0": observed_partial.tolist(),
    }
    (OUT_DIR / "expectation_identity.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
