"""Convergence of the Brier score, the weighted squared divergence, and KL.

Extends run_comparison.py. Three population objectives on a finite alphabet:

    R_BS(q)  = ||q - p||_2^2
    R_WD(q)  = sum_a p(a) (p(a) - q(a))^2
    R_KL(q)  = KL(p || q) = sum_a p(a) log(p(a) / q(a))

Expected log loss differs from KL(p || q) by the entropy H(p), which does not
depend on q. Their q-gradients agree.

The script checks four claims.

1. On the probability simplex, with the Euclidean metric, the local rate of KL
   at q = p is set by 1/p(a). Rare symbols are the fast directions. The
   weighted squared divergence has local rate set by p(a). Rare symbols are
   the slow directions. The Brier score is isotropic.
2. The scalar log loss of one outcome does not depend on how the remaining
   probability is split. The Brier score does. Separately, KL(p || q) diverges
   when any in-support coordinate of q goes to 0, while the Brier excess stays
   finite. Those are different penalties.
3. In softmax logit coordinates the gradient of expected log loss is q - p,
   which stays bounded as a coordinate approaches 0. The logit gradients of
   the Brier score and of the weighted squared divergence are multiplied by
   that small coordinate, so a collapsed class is slow to revive.
4. Adding a local square penalty to log loss moves the minimizer off p when
   the alphabet has three or more symbols. Adding a multiple of the Brier
   score does not: a nonnegative combination of proper scores is proper.
   Clipping the logarithm removes the barrier and can move the minimizer.

Run from the repository root:

    python experiments/04_brier_vs_weighted_divergence/run_kl_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_comparison as rc

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"
FIG_DIR = HERE / "tmlr" / "figures"
SEED = 0

P_SKEW = np.array([0.70, 0.20, 0.10])
Q_INTERIOR = np.array([0.05, 0.90, 0.05])
COLORS = {
    "brier": "#1f4e79",
    "weighted_squared_divergence": "#c47b00",
    "kl": "#2e7d32",
}
LABELS = {
    "brier": "Brier score",
    "weighted_squared_divergence": "Weighted squared divergence",
    "kl": "KL divergence",
}


def softmax(theta: np.ndarray) -> np.ndarray:
    z = np.asarray(theta, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), 1e-300, None)
    return float(np.sum(p * np.log(p / q)))


def grad_brier(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return 2.0 * (q - p)


def grad_wd(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return 2.0 * p * (q - p)


def grad_kl(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -p / np.clip(q, 1e-300, None)


GRADS = {"brier": grad_brier, "weighted_squared_divergence": grad_wd, "kl": grad_kl}


def jacobian_transpose_times(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(J_softmax^T v)_j = q_j (v_j - <q, v>)."""
    return q * (v - np.dot(q, v))


def logit_grad(name: str, theta: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = softmax(theta)
    if name == "kl":
        # Cancellation: J^T (-p / q) = q - p.
        return q - p
    return jacobian_transpose_times(q, GRADS[name](q, p))


def check_logit_gradients(rng: np.random.Generator) -> None:
    p = rc.as_prob(rng.random(4))
    theta = rng.normal(size=4)
    q = softmax(theta)
    analytic_kl = logit_grad("kl", theta, p)
    if np.linalg.norm(analytic_kl - (q - p)) > 1e-10:
        raise AssertionError("logit gradient of KL did not reduce to q - p")
    for name, loss in (
        ("brier", lambda qq: rc.expected_brier(p, qq)),
        ("weighted_squared_divergence", lambda qq: rc.weighted_divergence(p, qq)),
        ("kl", lambda qq: kl_divergence(p, qq)),
    ):
        analytic = logit_grad(name, theta, p)
        fd = np.zeros_like(theta)
        step = 1e-6
        base = loss(softmax(theta))
        for j in range(theta.size):
            bumped = theta.copy()
            bumped[j] += step
            fd[j] = (loss(softmax(bumped)) - base) / step
        err = float(np.linalg.norm(analytic - fd))
        if err > 2e-5:
            raise AssertionError(f"{name} logit gradient finite-difference error {err}")


def integrate_flow(rhs, y0: np.ndarray, t_end: float, first_step: float | None = None):
    kwargs = dict(rtol=1e-8, atol=1e-10, dense_output=True, method="DOP853")
    if first_step is not None:
        kwargs["first_step"] = first_step
    sol = solve_ivp(rhs, (0.0, t_end), np.asarray(y0, dtype=float), **kwargs)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def probability_rhs(name: str, p: np.ndarray):
    p = rc.as_prob(p)

    def rhs(_t, q):
        qq = np.clip(q, 1e-16, None)
        qq = qq / qq.sum()
        g = GRADS[name](qq, p)
        return -(g - np.mean(g))

    return rhs


def logit_rhs(name: str, p: np.ndarray):
    p = rc.as_prob(p)

    def rhs(_t, theta):
        return -logit_grad(name, theta, p)

    return rhs


def as_display_prob(q: np.ndarray) -> np.ndarray:
    """Map an integrator state back to the simplex.

    Dense output can overshoot a coordinate by about 1e-4 even when every
    accepted step of the vector field was clipped. The distance we report is
    the distance after that Euclidean repair.
    """
    qq = np.clip(np.asarray(q, dtype=float), 0.0, None)
    total = float(qq.sum())
    if total <= 0.0:
        return np.full(qq.shape, 1.0 / qq.size)
    return qq / total


def distance_curve_from_solution(sol, t_end: float, n_grid: int, to_prob) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.linspace(0.0, t_end, n_grid)
    states = sol.sol(times).T
    probs = np.vstack([as_display_prob(to_prob(row)) for row in states])
    return times, states, probs


def first_passage(times: np.ndarray, dist: np.ndarray, tol: float) -> float | None:
    hit = np.nonzero(dist <= tol)[0]
    if hit.size == 0:
        return None
    return float(times[int(hit[0])])


def flow_record(name: str, p: np.ndarray, probs: np.ndarray, times: np.ndarray) -> dict:
    dist = np.linalg.norm(probs - p, axis=1)
    return {
        "name": name,
        "times": times,
        "distance": dist,
        "passage": {
            "1e-1": first_passage(times, dist, 1e-1),
            "1e-2": first_passage(times, dist, 1e-2),
            "1e-3": first_passage(times, dist, 1e-3),
        },
        "terminal_distance": float(dist[-1]),
        "terminal_min_coordinate": float(np.min(probs[-1])),
    }


def projected_euler(name: str, p: np.ndarray, q0: np.ndarray, t_end: float, dt: float):
    """Euclidean projected gradient flow, discretized by projection after each step.

    The equality-constrained ODE is not the same object: for the weighted
    squared divergence it leaves the positive orthant. Probabilities stay in
    the simplex here, which is the geometry used in the earlier comparison.
    """
    q = rc.as_prob(q0)
    n_steps = int(round(t_end / dt))
    times = np.linspace(0.0, t_end, n_steps + 1)
    probs = np.empty((n_steps + 1, p.size))
    for i in range(n_steps + 1):
        probs[i] = q
        if i == n_steps:
            break
        g = GRADS[name](q, p)
        q = rc.proj_simplex(q - dt * g)
    return times, probs


def run_probability_flows(p: np.ndarray, q0: np.ndarray, t_end: float, dt: float) -> dict:
    p = rc.as_prob(p)
    q0 = rc.as_prob(q0)
    records = {}
    curves = {}
    for name in GRADS:
        times, probs = projected_euler(name, p, q0, t_end, dt)
        # Keep the plotted curve short. Passage times use the full grid.
        stride = max(1, times.size // 2000)
        records[name] = flow_record(name, p, probs, times)
        curves[name] = (times[::stride], records[name]["distance"][::stride])
    return {
        "p": p.tolist(),
        "q0": q0.tolist(),
        "t_end": t_end,
        "dt": dt,
        "records": records,
        "curves": curves,
    }


def run_logit_flows(p: np.ndarray, theta0: np.ndarray, t_end: float) -> dict:
    p = rc.as_prob(p)
    theta0 = np.asarray(theta0, dtype=float)
    records = {}
    curves = {}
    for name in GRADS:
        sol = integrate_flow(logit_rhs(name, p), theta0, t_end)
        times, _, probs = distance_curve_from_solution(sol, t_end, 4001, softmax)
        records[name] = flow_record(name, p, probs, times)
        curves[name] = (times, records[name]["distance"])
    return {
        "p": p.tolist(),
        "theta0": theta0.tolist(),
        "q0": softmax(theta0).tolist(),
        "t_end": t_end,
        "records": records,
        "curves": curves,
    }


def local_rates(p: np.ndarray) -> dict:
    p = rc.as_prob(p)
    specs = {
        "brier": np.full(p.size, 2.0),
        "weighted_squared_divergence": 2.0 * p,
        "kl": 1.0 / p,
    }
    out = {}
    for name, diagonal in specs.items():
        eigs = rc.tangent_eigenvalues(diagonal)
        ratio = rc.pure_mode_decay(diagonal, 1.0 / eigs[0])
        if abs(ratio - np.exp(-1.0)) > 1e-8:
            raise AssertionError(f"{name} slow-mode check failed: {ratio}")
        out[name] = {
            "rates_slow_to_fast": eigs.tolist(),
            "slowest_rate": float(eigs[0]),
            "fastest_rate": float(eigs[-1]),
            "condition_number": float(eigs[-1] / eigs[0]),
            "time_to_shrink_slow_mode_by_e": float(1.0 / eigs[0]),
        }
    out["slow_time_ratio_wd_over_brier"] = float(
        out["weighted_squared_divergence"]["time_to_shrink_slow_mode_by_e"]
        / out["brier"]["time_to_shrink_slow_mode_by_e"]
    )
    out["slow_time_ratio_brier_over_kl"] = float(
        out["brier"]["time_to_shrink_slow_mode_by_e"] / out["kl"]["time_to_shrink_slow_mode_by_e"]
    )
    return out


def boundary_path(p: np.ndarray) -> dict:
    """Drive the smallest coordinate from its true value down to 1e-8.

    The other coordinates stay in the same proportions as p.
    """
    p = rc.as_prob(p)
    rare = int(np.argmin(p))
    levels = np.array([p[rare], 1e-2, 1e-3, 1e-4, 1e-6, 1e-8])
    rows = []
    for level in levels:
        q = p.copy()
        removed = p[rare] - level
        q[rare] = level
        others = np.ones(p.size, dtype=bool)
        others[rare] = False
        q[others] = p[others] + removed * p[others] / p[others].sum()
        q = rc.as_prob(q)
        grads = {}
        for name, fn in GRADS.items():
            g = fn(q, p)
            g = g - np.mean(g)
            grads[name] = float(np.linalg.norm(g))
        rows.append(
            {
                "q_rare": float(q[rare]),
                "brier_excess": float(np.sum((q - p) ** 2)),
                "weighted_squared_divergence": rc.weighted_divergence(p, q),
                "kl": kl_divergence(p, q),
                "projected_gradient_norm": grads,
            }
        )
    return {"rare_index": rare, "p_rare": float(p[rare]), "rows": rows}


def residual_split() -> dict:
    """Same probability on the observed symbol, two splits of the remainder."""
    q_concentrated = np.array([0.40, 0.60, 0.00, 0.00])
    q_spread = np.array([0.40, 0.20, 0.20, 0.20])
    observed = 0
    log_a = float(-np.log(q_concentrated[observed]))
    log_b = float(-np.log(q_spread[observed]))
    if abs(log_a - log_b) > 1e-12:
        raise AssertionError("log loss changed with the unobserved split")
    return {
        "observed_symbol": observed,
        "q_concentrated": q_concentrated.tolist(),
        "q_spread": q_spread.tolist(),
        "log_loss": log_a,
        "brier_concentrated": rc.brier_score(q_concentrated, observed),
        "brier_spread": rc.brier_score(q_spread, observed),
        "brier_gap_concentrated_minus_spread": rc.brier_score(q_concentrated, observed)
        - rc.brier_score(q_spread, observed),
    }


def slsqp_population(loss_fn, k: int, rng: np.random.Generator) -> np.ndarray:
    return rc.slsqp_minimum(loss_fn, k, rng, n_starts=10)


def stabilization_table(p: np.ndarray, rng: np.random.Generator) -> dict:
    p = rc.as_prob(p)
    rows = []
    for lam in (0.0, 1.0, 10.0):
        q_local = slsqp_population(
            lambda q, lam=lam: rc.expected_log_loss(p, q) + lam * rc.expected_local(p, q),
            p.size,
            rng,
        )
        q_brier = slsqp_population(
            lambda q, lam=lam: rc.expected_log_loss(p, q) + lam * rc.expected_brier(p, q),
            p.size,
            rng,
        )
        rows.append(
            {
                "lambda": lam,
                "log_plus_local_square_distance_to_p": float(np.linalg.norm(q_local - p)),
                "log_plus_local_square_q": q_local.tolist(),
                "log_plus_brier_distance_to_p": float(np.linalg.norm(q_brier - p)),
                "log_plus_brier_q": q_brier.tolist(),
            }
        )
        if rows[-1]["log_plus_brier_distance_to_p"] > 1e-5:
            raise AssertionError("log loss plus a Brier multiple left p")
    if rows[0]["log_plus_local_square_distance_to_p"] > 1e-5:
        raise AssertionError("pure log loss left p")
    if rows[-1]["log_plus_local_square_distance_to_p"] < 1e-3:
        raise AssertionError("local-square stabilization still recovered p")
    return {"p": p.tolist(), "rows": rows}


def clipped_log_table(p: np.ndarray, rng: np.random.Generator) -> dict:
    p = rc.as_prob(p)
    rows = []
    for eps in (1e-8, 1e-3, 0.05, 0.15):
        def loss(q, eps=eps):
            return float(-np.sum(p * np.log(np.maximum(q, eps))))

        q_hat = slsqp_population(loss, p.size, rng)
        rows.append(
            {
                "epsilon": eps,
                "q": q_hat.tolist(),
                "distance_to_p": float(np.linalg.norm(q_hat - p)),
                "loss_at_q": loss(q_hat),
                "loss_at_p": loss(p),
                "min_coordinate": float(np.min(q_hat)),
            }
        )
        if loss(q_hat) > loss(p) + 1e-7:
            raise AssertionError(f"clipped minimizer is worse than p at eps={eps}")
    return {"p": p.tolist(), "rows": rows}


def smoothed_target_bias(p: np.ndarray, alpha: float) -> dict:
    p = rc.as_prob(p)
    uniform = np.full(p.size, 1.0 / p.size)
    target = (1.0 - alpha) * p + alpha * uniform
    return {
        "alpha": alpha,
        "target": target.tolist(),
        "distance_to_p": float(np.linalg.norm(target - p)),
    }


def paired_logit_sgd(p: np.ndarray, rng: np.random.Generator) -> dict:
    """Paired online updates of the two scoring rules, plus oracle WD steps.

    Each seed uses the same outcomes for the Brier score and for log loss.
    The weighted squared divergence step uses p itself, so it is not a
    one-sample update. Step size is shared.
    """
    p = rc.as_prob(p)
    n_seeds = 40
    n_steps = 8000
    eta_scale = 1.0
    checkpoints = (200, 1000, 4000, 8000)
    outcomes = rng.choice(p.size, size=(n_seeds, n_steps), p=p)
    names = ("brier", "kl", "weighted_squared_divergence")
    dist = {name: np.zeros((n_seeds, len(checkpoints))) for name in names}
    for s in range(n_seeds):
        theta = {name: np.zeros(p.size) for name in names}
        next_mark = 0
        for t in range(1, n_steps + 1):
            x = int(outcomes[s, t - 1])
            for name in names:
                q = softmax(theta[name])
                if name == "kl":
                    g = q.copy()
                    g[x] -= 1.0
                elif name == "brier":
                    v = 2.0 * q
                    v[x] -= 2.0
                    g = jacobian_transpose_times(q, v)
                else:
                    g = logit_grad("weighted_squared_divergence", theta[name], p)
                theta[name] = theta[name] - (eta_scale / np.sqrt(t)) * g
            if next_mark < len(checkpoints) and t == checkpoints[next_mark]:
                for name in names:
                    dist[name][s, next_mark] = np.linalg.norm(softmax(theta[name]) - p)
                next_mark += 1
    summary = {
        "n_seeds": n_seeds,
        "n_steps": n_steps,
        "step_size": "1/sqrt(t)",
        "note": (
            "Brier and KL use one-sample logit gradients on paired outcomes. "
            "The weighted squared divergence uses the population logit gradient, "
            "so its trajectory does not depend on the seed."
        ),
        "checkpoints": [],
    }
    for j, t in enumerate(checkpoints):
        entry = {"step": t}
        for name in names:
            values = dist[name][:, j]
            entry[name] = {
                "mean": float(np.mean(values)),
                "se": float(np.std(values, ddof=1) / np.sqrt(values.size)),
            }
        gap = dist["kl"][:, j] - dist["brier"][:, j]
        entry["kl_minus_brier"] = {
            "mean": float(np.mean(gap)),
            "se": float(np.std(gap, ddof=1) / np.sqrt(gap.size)),
        }
        summary["checkpoints"].append(entry)
    summary["curves"] = {
        name: (np.array(checkpoints, dtype=float), dist[name].mean(axis=0), dist[name].std(axis=0, ddof=1) / np.sqrt(n_seeds))
        for name in names
    }
    return summary


def plot_flows(panels: list[tuple[str, dict]], path: Path, xlim: float) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.1 * len(panels), 3.6), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, result) in zip(axes, panels):
        for name, (times, dist) in result["curves"].items():
            ax.plot(times, dist, color=COLORS[name], lw=2.0, label=LABELS[name])
        ax.set_yscale("log")
        ax.set_xlim(0.0, xlim)
        ax.set_xlabel("Gradient-flow time")
        ax.set_ylabel(r"$\|q(t) - p\|_2$")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_boundary(path_rows: dict, path: Path) -> None:
    rows = [row for row in path_rows["rows"] if row["q_rare"] < path_rows["p_rare"] * 0.5]
    levels = [row["q_rare"] for row in rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    series = {
        "brier": [row["brier_excess"] for row in rows],
        "weighted_squared_divergence": [row["weighted_squared_divergence"] for row in rows],
        "kl": [row["kl"] for row in rows],
    }
    for name, values in series.items():
        ax.plot(levels, values, marker="o", color=COLORS[name], lw=2.0, label=LABELS[name])
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"Probability on the rare symbol")
    ax.set_ylabel("Objective above its value at $p$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sgd(sgd: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    sgd_labels = {
        "brier": "Brier score",
        "kl": "log loss",
        "weighted_squared_divergence": "Weighted squared divergence",
    }
    for name, (steps, mean, se) in sgd["curves"].items():
        ax.plot(steps, mean, color=COLORS[name], lw=2.0, label=sgd_labels[name])
        ax.fill_between(steps, mean - 1.96 * se, mean + 1.96 * se, color=COLORS[name], alpha=0.18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Update")
    ax.set_ylabel(r"$\|q - p\|_2$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def strip_curves(result: dict) -> dict:
    records = {}
    for name, row in result["records"].items():
        records[name] = {
            "passage": row["passage"],
            "terminal_distance": row["terminal_distance"],
            "terminal_min_coordinate": row["terminal_min_coordinate"],
        }
    kept = {key: value for key, value in result.items() if key != "curves"}
    kept["records"] = records
    return kept


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    check_logit_gradients(rng)

    rates = local_rates(P_SKEW)
    interior = run_probability_flows(P_SKEW, Q_INTERIOR, t_end=12.0, dt=1e-4)
    near_zero = np.array([1e-4, 0.50, 0.50])
    collapsed_prob = run_probability_flows(P_SKEW, near_zero, t_end=8.0, dt=1e-5)
    logits_uniform = run_logit_flows(P_SKEW, np.zeros(3), t_end=80.0)
    logits_collapsed = run_logit_flows(P_SKEW, np.array([-12.0, 0.0, 0.0]), t_end=80.0)
    boundary = boundary_path(P_SKEW)
    split = residual_split()
    stabilized = stabilization_table(P_SKEW, rng)
    clipped = clipped_log_table(P_SKEW, rng)
    smoothed = smoothed_target_bias(P_SKEW, alpha=0.10)
    sgd = paired_logit_sgd(P_SKEW, rng)
    collapse_gradients = []
    for level in (-4.0, -8.0, -12.0):
        theta = np.array([level, 0.0, 0.0])
        q = softmax(theta)
        entry = {
            "logit": level,
            "q": q.tolist(),
            "distance": float(np.linalg.norm(q - rc.as_prob(P_SKEW))),
            "logit_gradient": {},
        }
        for name in GRADS:
            g = logit_grad(name, theta, rc.as_prob(P_SKEW))
            entry["logit_gradient"][name] = {
                "vector": g.tolist(),
                "norm": float(np.linalg.norm(g)),
                "collapsed_coordinate": float(g[0]),
            }
        collapse_gradients.append(entry)

    # The interior probability-space passages should agree with the earlier
    # fixed-step integration to plotting accuracy.
    brier_1e3 = interior["records"]["brier"]["passage"]["1e-3"]
    kl_1e3 = interior["records"]["kl"]["passage"]["1e-3"]
    if brier_1e3 is None or not (3.3 <= brier_1e3 <= 3.6):
        raise AssertionError(f"Brier passage drifted from the previous run: {brier_1e3}")
    if kl_1e3 is None or not (2.3 <= kl_1e3 <= 2.7):
        raise AssertionError(f"KL passage drifted from the previous run: {kl_1e3}")
    wd_1e1 = interior["records"]["weighted_squared_divergence"]["passage"]["1e-1"]
    if wd_1e1 is None or not (4.1 <= wd_1e1 <= 4.4):
        raise AssertionError(f"weighted-divergence passage drifted: {wd_1e1}")

    plot_flows(
        [
            (r"Probability coordinates, interior start", interior),
            (r"Probability coordinates, one class at $10^{-4}$", collapsed_prob),
        ],
        FIG_DIR / "flow_probability.pdf",
        xlim=8.0,
    )
    plot_flows(
        [
            (r"Logit coordinates, start at the uniform vector", logits_uniform),
            (r"Logit coordinates, one logit at $-12$", logits_collapsed),
        ],
        FIG_DIR / "flow_logit.pdf",
        xlim=80.0,
    )
    plot_boundary(boundary, FIG_DIR / "boundary_penalty.pdf")
    plot_sgd(sgd, FIG_DIR / "sgd_logit.pdf")

    summary = {
        "seed": SEED,
        "p": P_SKEW.tolist(),
        "local_rates": rates,
        "probability_flow_interior": strip_curves(interior),
        "probability_flow_near_zero": strip_curves(collapsed_prob),
        "logit_flow_uniform_start": strip_curves(logits_uniform),
        "logit_flow_collapsed": strip_curves(logits_collapsed),
        "boundary_path": boundary,
        "residual_split": split,
        "stabilization": stabilized,
        "clipped_log": clipped,
        "smoothed_target": smoothed,
        "sgd": {key: value for key, value in sgd.items() if key != "curves"},
        "collapse_gradients": collapse_gradients,
    }
    (OUT_DIR / "kl_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("local slow rates", {k: rates[k]["slowest_rate"] for k in GRADS})
    print("slow-time WD/Brier", rates["slow_time_ratio_wd_over_brier"])
    print("slow-time Brier/KL", rates["slow_time_ratio_brier_over_kl"])
    print("interior passages", {k: v["passage"] for k, v in interior["records"].items()})
    print("near-zero passages", {k: v["passage"] for k, v in collapsed_prob["records"].items()})
    print("logit uniform passages", {k: v["passage"] for k, v in logits_uniform["records"].items()})
    print("logit collapsed passages", {k: v["passage"] for k, v in logits_collapsed["records"].items()})
    print("logit collapsed terminals", {k: v["terminal_distance"] for k, v in logits_collapsed["records"].items()})
    print("boundary last", boundary["rows"][-1])
    print("residual", split)
    print("stabilization", stabilized["rows"])
    print("clipped", clipped["rows"])
    print("smoothed", smoothed)
    for row in sgd["checkpoints"]:
        print(
            "sgd",
            row["step"],
            {name: (round(row[name]["mean"], 4), round(row[name]["se"], 4)) for name in GRADS},
            "kl-brier",
            round(row["kl_minus_brier"]["mean"], 4),
            round(row["kl_minus_brier"]["se"], 4),
        )


if __name__ == "__main__":
    main()
