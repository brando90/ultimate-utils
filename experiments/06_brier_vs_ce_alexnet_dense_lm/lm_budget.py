"""Fixed language-model token budget.

The number is chosen once from the packed corpus and the slower loss's
measured throughput, then stored. It is not the time left after the CNN runs.
Both losses and every seed use that same step count.
"""

from __future__ import annotations

TUNE_TOKENS = 8_000_000
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3)
N_SEEDS = 3
WALL_CAP_SECONDS = 48 * 3600
FLOOR_TOKENS = 20_000_000
FLOOR_CAP_SECONDS = 72 * 3600
TOKEN_STEP = 5_000_000


def program_seconds(final_tokens: int, tokens_per_sec: float) -> float:
    n_tune = len(LEARNING_RATES) * 2
    total_tokens = n_tune * TUNE_TOKENS + (N_SEEDS * 2) * final_tokens
    return total_tokens / tokens_per_sec


def choose_final_tokens(epoch_tokens: int, tokens_per_sec: float) -> dict:
    """Pick one matched final budget.

    Prefer one packed epoch when the tune plus six final runs fit in 48 hours.
    Otherwise use the largest 5,000,000-token multiple of that epoch that fits.
    If that multiple is under 20,000,000 tokens and 20,000,000 still fits in
    72 hours, use 20,000,000. A slower machine than that stops before training.
    """
    if tokens_per_sec <= 0:
        raise RuntimeError("measured tokens per second must be positive")
    epoch_tokens = int(epoch_tokens)
    if epoch_tokens < FLOOR_TOKENS:
        raise RuntimeError(f"packed epoch has {epoch_tokens} tokens, under {FLOOR_TOKENS}")
    if program_seconds(epoch_tokens, tokens_per_sec) <= WALL_CAP_SECONDS:
        chosen = epoch_tokens
        rule = "one packed epoch of the WikiText-103 raw training split"
    else:
        chosen = 0
        for tokens in range(TOKEN_STEP, epoch_tokens + 1, TOKEN_STEP):
            if program_seconds(tokens, tokens_per_sec) <= WALL_CAP_SECONDS:
                chosen = tokens
        if chosen >= FLOOR_TOKENS:
            rule = "largest 5000000-token multiple of one epoch that fits in 48 hours"
        elif program_seconds(FLOOR_TOKENS, tokens_per_sec) <= FLOOR_CAP_SECONDS:
            chosen = FLOOR_TOKENS
            rule = "20000000-token floor because one epoch exceeds 48 hours and this floor fits in 72 hours"
        else:
            hours = program_seconds(FLOOR_TOKENS, tokens_per_sec) / 3600
            raise RuntimeError(
                f"20000000 tokens for six final runs plus tuning would take {hours:.1f} hours"
            )
    projected = program_seconds(chosen, tokens_per_sec)
    return {
        "final_tokens_per_run": int(chosen),
        "tune_tokens_per_trial": TUNE_TOKENS,
        "learning_rates": list(LEARNING_RATES),
        "seeds": N_SEEDS,
        "epoch_tokens": epoch_tokens,
        "tokens_per_sec": float(tokens_per_sec),
        "projected_seconds": projected,
        "watchdog_seconds": projected * 1.5 + 3600,
        "rule": rule,
    }
