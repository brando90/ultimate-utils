"""Brier score versus the weighted squared divergence in the handwritten notes.

The notes define two different objects and then treat them as one.

Brier / 6.437 quadratic cost, summed over the whole alphabet:
    BS(q, x) = sum_a (q(a) - 1{a = x})^2
             = (1 - q(x))^2 + sum_{a != x} q(a)^2

Weighted divergence, with the true distribution used twice:
    WD(p, q) = sum_a p(a) (p(a) - q(a))^2

Substituting a one-hot into WD deletes every term except the observed symbol
and leaves only (1 - q(x))^2. That local score is not the Brier score.

This script checks three claims.

1. The population minimizer of expected Brier, of log loss, and of WD(p, q)
   is the true distribution p. The local score (1 - q(x))^2 has that same
   minimizer only for a binary alphabet. For three or more symbols it does not.
2. On the probability simplex, WD and Brier are both minimized at p, but the
   projected Hessians differ, so gradient flow converges at different rates.
3. With a free probability vector, the finite-sample minimizer of either
   empirical Brier or empirical WD is exactly the histogram. The speed gap is
   a property of the objective, not two different histograms.

Run from the repository root:

    python experiments/04_brier_vs_weighted_divergence/run_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, null_space
from scipy.optimize import minimize

OUT_DIR = Path(__file__).resolve().parent / "results"
SEED = 0


def as_prob(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if np.any(p < -1e-15):
        raise ValueError(f"negative mass: {p}")
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0:
        raise ValueError("distribution has no mass")
    return p / total


def proj_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {q >= 0, sum q = 1} (Duchi et al.)."""
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def brier_score(q: np.ndarray, x: int) -> float:
    one_hot = np.zeros_like(q)
    one_hot[x] = 1.0
    return float(np.sum((q - one_hot) ** 2))


def local_squared_score(q: np.ndarray, x: int) -> float:
    """One-hot plug-in of WD. Depends on q only through q(x)."""
    return float((1.0 - q[x]) ** 2)


def weighted_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * (p - q) ** 2))


def expected_brier(p: np.ndarray, q: np.ndarray) -> float:
    # E[sum_a (q(a) - 1{a=X})^2] = ||q - p||^2 + sum_a p(a) (1 - p(a))
    return float(np.sum((q - p) ** 2) + np.sum(p * (1.0 - p)))


def expected_log_loss(p: np.ndarray, q: np.ndarray) -> float:
    qq = np.clip(q, 1e-300, None)
    return float(-np.sum(p * np.log(qq)))


def expected_local(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * (1.0 - q) ** 2))


def local_minimizer(p: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed form for argmin_{q in simplex} E[(1 - q(X))^2].

    Stationarity on a support S of size m gives q_j = 1 - mu / p_j on S,
    with mu = (m - 1) / sum_{j in S} (1 / p_j), and q = 0 off S.
    Included atoms must satisfy p_j >= mu. Excluded atoms must satisfy
    p_j <= mu. For a binary alphabet this returns q = p.
    """
    p = as_prob(p)
    order = np.argsort(-p)
    k = p.size
    for m in range(k, 0, -1):
        support = order[:m]
        if np.any(p[support] <= 0):
            continue
        mu = (m - 1) / np.sum(1.0 / p[support])
        q = np.zeros(k)
        q[support] = 1.0 - mu / p[support]
        if np.any(q[support] < -1e-9):
            continue
        q[support] = np.clip(q[support], 0.0, None)
        excluded = np.ones(k, dtype=bool)
        excluded[support] = False
        if np.all(p[support] >= mu - 1e-8) and np.all(p[excluded] <= mu + 1e-8):
            q = as_prob(q)
            return q, float(mu)
    raise RuntimeError(f"no KKT point for p={p}")


def brute_force_local_minimum(p: np.ndarray, n_grid: int = 801) -> np.ndarray:
    """Grid search on the ternary simplex, used as an independent check."""
    p = as_prob(p)
    if p.size != 3:
        raise ValueError("brute force helper is for three symbols")
    axis = np.linspace(0.0, 1.0, n_grid)
    best_q = None
    best_loss = np.inf
    for i, q1 in enumerate(axis):
        q2 = axis[: n_grid - i]
        q3 = 1.0 - q1 - q2
        loss = (
            p[0] * (1.0 - q1) ** 2
            + p[1] * (1.0 - q2) ** 2
            + p[2] * (1.0 - q3) ** 2
        )
        j = int(np.argmin(loss))
        if loss[j] < best_loss:
            best_loss = float(loss[j])
            best_q = np.array([q1, q2[j], q3[j]])
    return best_q


def slsqp_minimum(loss_fn, k: int, rng: np.random.Generator, n_starts: int = 12) -> np.ndarray:
    cons = {"type": "eq", "fun": lambda q: np.sum(q) - 1.0}
    bounds = [(0.0, 1.0)] * k
    starts = [np.full(k, 1.0 / k)]
    starts.extend(as_prob(rng.random(k)) for _ in range(n_starts - 1))
    best_q = None
    best_loss = np.inf
    for q0 in starts:
        result = minimize(
            loss_fn,
            q0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            tol=1e-14,
            options={"maxiter": 500, "ftol": 1e-14},
        )
        if not result.success:
            continue
        q = proj_simplex(result.x)
        value = float(loss_fn(q))
        if value < best_loss:
            best_loss = value
            best_q = q
    if best_q is None:
        raise RuntimeError("SLSQP failed from every start")
    return best_q


def tangent_eigenvalues(diagonal: np.ndarray) -> np.ndarray:
    """Eigenvalues of P diag(diagonal) acting on {v : sum v = 0}.

    Gradient flow of a quadratic whose Euclidean Hessian is diag(diagonal),
    after projection onto the simplex, decays at these rates.
    """
    diagonal = np.asarray(diagonal, dtype=float)
    basis = null_space(np.ones((1, diagonal.size)))
    hessian = basis.T @ (diagonal[:, None] * basis)
    return np.sort(np.linalg.eigvalsh(hessian))


def pure_mode_decay(diagonal: np.ndarray, t: float) -> float:
    """||e(t)|| / ||e(0)|| for the slowest tangent mode. Exact, via expm."""
    diagonal = np.asarray(diagonal, dtype=float)
    k = diagonal.size
    basis = null_space(np.ones((1, k)))
    hessian = basis.T @ (diagonal[:, None] * basis)
    values, vectors = np.linalg.eigh(hessian)
    slow = vectors[:, 0]
    e0 = basis @ slow
    e0 = e0 / np.linalg.norm(e0)
    projector = np.eye(k) - np.ones((k, k)) / k
    generator = projector @ np.diag(diagonal)
    e_t = expm(-t * generator) @ e0
    return float(np.linalg.norm(e_t) / np.linalg.norm(e0))


def flow_distance_curve(
    p: np.ndarray,
    q0: np.ndarray,
    grad_fn,
    t_end: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projected gradient flow. Returns times, distance to p, and the path."""
    p = as_prob(p)
    q = as_prob(q0)
    dt = t_end / n_steps
    times = np.linspace(0.0, t_end, n_steps + 1)
    dist = np.empty(n_steps + 1)
    path = np.empty((n_steps + 1, p.size))
    for i in range(n_steps + 1):
        path[i] = q
        dist[i] = np.linalg.norm(q - p)
        if i == n_steps:
            break
        g = grad_fn(q, p)
        q = proj_simplex(q - dt * g)
    return times, dist, path


def first_passage(times: np.ndarray, dist: np.ndarray, tol: float) -> float | None:
    hit = np.nonzero(dist <= tol)[0]
    if hit.size == 0:
        return None
    return float(times[hit[0]])


def check_score_identity(rng: np.random.Generator) -> dict:
    gaps = []
    for _ in range(200):
        k = int(rng.integers(2, 8))
        q = as_prob(rng.random(k))
        x = int(rng.integers(0, k))
        one_hot = np.zeros(k)
        one_hot[x] = 1.0
        brier = brier_score(q, x)
        local = local_squared_score(q, x)
        weighted_plug_in = weighted_divergence(one_hot, q)
        dropped = float(np.sum(q**2) - q[x] ** 2)
        if abs(weighted_plug_in - local) > 1e-12:
            raise AssertionError("one-hot WD did not reduce to (1 - q(x))^2")
        if abs(brier - (local + dropped)) > 1e-12:
            raise AssertionError("Brier decomposition failed")
        gaps.append(brier - local)
    return {
        "n_random_pairs": 200,
        "max_brier_minus_local": float(np.max(gaps)),
        "mean_brier_minus_local": float(np.mean(gaps)),
        "fraction_equal_within_1e_12": float(np.mean(np.asarray(gaps) <= 1e-12)),
    }


def check_population_minimizers(rng: np.random.Generator) -> list[dict]:
    cases = [
        ("binary_skewed", np.array([0.8, 0.2])),
        ("binary_uniform", np.array([0.5, 0.5])),
        ("ternary_skewed", np.array([0.7, 0.2, 0.1])),
        ("ternary_uniform", np.array([1.0, 1.0, 1.0]) / 3.0),
        ("five_skewed", np.array([0.50, 0.25, 0.15, 0.07, 0.03])),
    ]
    rows = []
    for name, p in cases:
        p = as_prob(p)
        q_local, mu = local_minimizer(p)
        q_brier = slsqp_minimum(lambda q, p=p: expected_brier(p, q), p.size, rng)
        q_log = slsqp_minimum(lambda q, p=p: expected_log_loss(p, q), p.size, rng)
        q_wd = slsqp_minimum(lambda q, p=p: weighted_divergence(p, q), p.size, rng)
        q_loc_opt = slsqp_minimum(lambda q, p=p: expected_local(p, q), p.size, rng)
        if p.size == 3:
            q_grid = brute_force_local_minimum(p)
            grid_gap = float(np.linalg.norm(q_grid - q_local))
            if grid_gap > 2e-3:
                raise AssertionError(f"grid search disagrees with KKT for {name}: {grid_gap}")
        else:
            grid_gap = None
        for label, q_hat, target in (
            ("brier", q_brier, p),
            ("log_loss", q_log, p),
            ("weighted_divergence", q_wd, p),
            ("local_squared", q_loc_opt, q_local),
        ):
            gap = float(np.linalg.norm(q_hat - target))
            if gap > 1e-6:
                raise AssertionError(f"{name} {label} minimizer off by {gap}: {q_hat} vs {target}")
        rows.append(
            {
                "name": name,
                "alphabet_size": int(p.size),
                "p": p.tolist(),
                "q_local_closed_form": q_local.tolist(),
                "lagrange_mu": mu,
                "distance_local_minimizer_to_p": float(np.linalg.norm(q_local - p)),
                "local_recovers_p": bool(np.linalg.norm(q_local - p) < 1e-8),
                "grid_gap_ternary": grid_gap,
                "optimizer_distance_to_claimed_minimizer": {
                    "brier": float(np.linalg.norm(q_brier - p)),
                    "log_loss": float(np.linalg.norm(q_log - p)),
                    "weighted_divergence": float(np.linalg.norm(q_wd - p)),
                    "local_squared": float(np.linalg.norm(q_loc_opt - q_local)),
                },
                "expected_local_at_p": expected_local(p, p),
                "expected_local_at_q_local": expected_local(p, q_local),
            }
        )
    return rows


def check_speed(p: np.ndarray, q0: np.ndarray) -> dict:
    p = as_prob(p)
    q0 = as_prob(q0)
    brier_eigs = tangent_eigenvalues(np.full(p.size, 2.0))
    wd_eigs = tangent_eigenvalues(2.0 * p)
    log_eigs = tangent_eigenvalues(1.0 / p)
    # Exact slow-mode decay at t = 1/lambda. A pure mode must shrink by e.
    checks = {}
    for name, diagonal, eigs in (
        ("brier", np.full(p.size, 2.0), brier_eigs),
        ("weighted_divergence", 2.0 * p, wd_eigs),
        ("log_loss_linearization", 1.0 / p, log_eigs),
    ):
        t_star = 1.0 / eigs[0]
        ratio = pure_mode_decay(diagonal, t_star)
        if abs(ratio - np.exp(-1.0)) > 1e-8:
            raise AssertionError(f"{name} slow mode decay was {ratio}, expected 1/e")
        checks[name] = {
            "tangent_rates_slow_to_fast": eigs.tolist(),
            "slowest_rate": float(eigs[0]),
            "fastest_rate": float(eigs[-1]),
            "condition_number": float(eigs[-1] / eigs[0]),
            "time_to_shrink_slow_mode_by_e": float(t_star),
            "measured_shrink_factor": ratio,
        }

    def grad_brier(q, p_):
        return 2.0 * (q - p_)

    def grad_wd(q, p_):
        return 2.0 * p_ * (q - p_)

    def grad_log(q, p_):
        return -p_ / np.clip(q, 1e-12, None)

    def grad_local(q, p_):
        return -2.0 * p_ * (1.0 - q)

    curves = {}
    paths = {}
    t_end = 12.0
    n_steps = 12000
    for name, grad in (
        ("brier", grad_brier),
        ("weighted_divergence", grad_wd),
        ("log_loss", grad_log),
        ("local_squared", grad_local),
    ):
        times, dist, path = flow_distance_curve(p, q0, grad, t_end, n_steps)
        # Halve the step and demand the passage time agree. This is the
        # integrator check for the plotted trajectories.
        _, dist_fine, _ = flow_distance_curve(p, q0, grad, t_end, n_steps * 2)
        curves[name] = (times, dist)
        paths[name] = path
        for tol in (1e-1, 1e-2):
            t_coarse = first_passage(times, dist, tol)
            t_fine = first_passage(np.linspace(0.0, t_end, n_steps * 2 + 1), dist_fine, tol)
            if t_coarse is None or t_fine is None:
                continue
            if abs(t_coarse - t_fine) > 2e-3:
                raise AssertionError(f"{name} passage time unstable at tol {tol}: {t_coarse} vs {t_fine}")

    q_local, _ = local_minimizer(p)
    # Same path or only a slower copy of the same path?
    err_b = paths["brier"] - p
    err_w = paths["weighted_divergence"] - p
    # Compare directions once both errors are still appreciable.
    cosines = []
    for i in range(0, n_steps, 100):
        nb = np.linalg.norm(err_b[i])
        nw = np.linalg.norm(err_w[i])
        if nb < 1e-3 or nw < 1e-3:
            continue
        cosines.append(float(np.dot(err_b[i], err_w[i]) / (nb * nw)))
    return {
        "p": p.tolist(),
        "q0": q0.tolist(),
        "linear_rates": checks,
        "wd_over_brier_slow_time_ratio": float(
            checks["weighted_divergence"]["time_to_shrink_slow_mode_by_e"]
            / checks["brier"]["time_to_shrink_slow_mode_by_e"]
        ),
        "time_to_l2_below": {
            name: {
                "1e-1": first_passage(times, dist, 1e-1),
                "1e-2": first_passage(times, dist, 1e-2),
                "1e-3": first_passage(times, dist, 1e-3),
            }
            for name, (times, dist) in curves.items()
        },
        "terminal_distance_to_p": {name: float(dist[-1]) for name, (_, dist) in curves.items()},
        "terminal_distance_to_local_minimizer": {
            name: float(np.linalg.norm(paths[name][-1] - q_local)) for name in paths
        },
        "min_error_direction_cosine_wd_vs_brier": float(np.min(cosines)) if cosines else None,
        "curves": {name: (times, dist) for name, (times, dist) in curves.items()},
    }


def finite_sample(p: np.ndarray, rng: np.random.Generator) -> dict:
    """Both proper scores and WD have empirical minimizer equal to the histogram.

    The local score's empirical minimizer is local_minimizer(histogram), which
    stays away from p. Means and standard errors are over independent multinomial
    samples. The standard error is the sample standard deviation divided by
    sqrt(n_trials).
    """
    p = as_prob(p)
    q_local, _ = local_minimizer(p)
    n_trials = 400
    rows = []
    for n in (20, 200, 2000, 20000):
        counts = rng.multinomial(n, p, size=n_trials)
        p_hat = counts / n
        dist_hist = np.linalg.norm(p_hat - p, axis=1)
        q_loc = np.vstack([local_minimizer(row)[0] for row in p_hat])
        dist_loc = np.linalg.norm(q_loc - p, axis=1)
        dist_loc_to_limit = np.linalg.norm(q_loc - q_local, axis=1)
        def pack(values):
            mean = float(np.mean(values))
            se = float(np.std(values, ddof=1) / np.sqrt(values.size))
            return {"mean": mean, "se": se, "n_trials": int(values.size)}
        rows.append(
            {
                "n": n,
                "histogram_l2_to_p": pack(dist_hist),
                "local_erm_l2_to_p": pack(dist_loc),
                "local_erm_l2_to_population_local_minimizer": pack(dist_loc_to_limit),
            }
        )
    return {
        "p": p.tolist(),
        "population_local_minimizer": q_local.tolist(),
        "population_local_l2_to_p": float(np.linalg.norm(q_local - p)),
        "n_trials": n_trials,
        "by_n": rows,
        "note": (
            "Empirical Brier, empirical log loss, and empirical WD(p_hat, q) "
            "are all uniquely minimized at q = p_hat. Their distance to p is "
            "the histogram row. The local score is the other row."
        ),
    }


def plot_flows(speed_rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=False)
    colors = {
        "brier": "#1f4e79",
        "weighted_divergence": "#c47b00",
        "log_loss": "#2e7d32",
        "local_squared": "#b00020",
    }
    labels = {
        "brier": "Brier",
        "weighted_divergence": "Weighted divergence",
        "log_loss": "Log loss",
        "local_squared": "Local (1 - q(x))^2",
    }
    for ax, row in zip(axes, speed_rows):
        for name, (times, dist) in row["curves"].items():
            ax.plot(times, dist, color=colors[name], lw=2.0, label=labels[name])
        ax.set_yscale("log")
        ax.set_xlabel("Gradient-flow time")
        ax.set_ylabel(r"$\|q(t) - p\|_2$")
        p = np.array(row["p"])
        ax.set_title(rf"$p = ({', '.join(f'{v:.2f}' for v in p)})$")
        ax.set_xlim(0, 8)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    identity = check_score_identity(rng)
    population = check_population_minimizers(rng)
    finite = finite_sample(np.array([0.7, 0.2, 0.1]), rng)
    speed_inputs = [
        (np.array([0.7, 0.2, 0.1]), np.array([0.05, 0.90, 0.05])),
        (np.array([1.0, 1.0, 1.0]) / 3.0, np.array([0.05, 0.90, 0.05])),
        (np.array([0.8, 0.2]), np.array([0.05, 0.95])),
    ]
    speed = [check_speed(p, q0) for p, q0 in speed_inputs]
    plot_flows(speed[:2], OUT_DIR / "distance_to_truth.png")

    def strip(row: dict) -> dict:
        return {key: value for key, value in row.items() if key != "curves"}

    summary = {
        "seed": SEED,
        "score_identity": identity,
        "population_minimizers": population,
        "finite_sample_ternary_skewed": finite,
        "gradient_flow": [strip(row) for row in speed],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("identity: max Brier minus local score =", identity["max_brier_minus_local"])
    for row in population:
        print(
            f"minimizer {row['name']}: local recovers p? {row['local_recovers_p']}"
            f"  ||q_local - p|| = {row['distance_local_minimizer_to_p']:.6f}"
            f"  q_local = {np.array(row['q_local_closed_form'])}"
        )
    skewed = speed[0]
    print("skewed slow-time ratio WD/Brier =", skewed["wd_over_brier_slow_time_ratio"])
    print("skewed passage times:", json.dumps(skewed["time_to_l2_below"], indent=2))
    print("skewed terminal distances:", skewed["terminal_distance_to_p"])
    print("direction cosine min WD vs Brier, skewed =", skewed["min_error_direction_cosine_wd_vs_brier"])
    uniform = speed[1]
    print("uniform slow-time ratio WD/Brier =", uniform["wd_over_brier_slow_time_ratio"])
    print("uniform direction cosine min =", uniform["min_error_direction_cosine_wd_vs_brier"])
    binary = speed[2]
    print("binary slow-time ratio WD/Brier =", binary["wd_over_brier_slow_time_ratio"])
    print("finite sample:")
    for row in finite["by_n"]:
        h = row["histogram_l2_to_p"]
        loc = row["local_erm_l2_to_p"]
        print(
            f"  n={row['n']}: histogram {h['mean']:.4f} ± {h['se']:.4f},"
            f" local {loc['mean']:.4f} ± {loc['se']:.4f}"
        )
    print("wrote", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
