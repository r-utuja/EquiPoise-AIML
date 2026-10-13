"""Policy value on held-out patients: how well would patients do if they followed a rule?

A policy assigns each patient a drug. Only patients who happened to be randomized to the
drug the policy picks tell us anything about it, so inverse-probability weighting (IPW)
re-weights them by 1 / P(their arm): V = mean(1[T = pi(X)] * Y / e(T)). Because e is known
from randomization, this is unbiased. The doubly robust (AIPW) estimate adds an outcome
model mu(X, a): V = mean(mu(X, pi(X)) + 1[T = pi(X)] / e(T) * (Y - mu(X, T))); it stays
unbiased with known e and is usually less noisy. 95% CIs come from bootstrapping patients.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from equipoise.config import Config


def ipw_value(T: np.ndarray, Y: np.ndarray, assigned: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((T == assigned) * Y / probs[T]))


def aipw_value(T: np.ndarray, Y: np.ndarray, assigned: np.ndarray, probs: np.ndarray,
               mu: np.ndarray) -> float:
    """mu: (n, n_arms) predicted outcome under each arm (fitted on training patients)."""
    n = len(Y)
    direct = mu[np.arange(n), assigned]
    correction = (T == assigned) / probs[T] * (Y - mu[np.arange(n), T])
    return float(np.mean(direct + correction))


def bootstrap_ci(fn: Callable[[np.ndarray], float], n: int, reps: int, seed: int,
                 level: float) -> tuple[float, float]:
    """Percentile bootstrap CI of a statistic computed on resampled patient indices."""
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(reps)])
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def evaluate_policies(policies: dict[str, np.ndarray], T: np.ndarray, Y: np.ndarray,
                      mu: np.ndarray, probs: np.ndarray, cfg: Config,
                      arm_names: list[str]) -> pd.DataFrame:
    """IPW and AIPW value (letters gained) with bootstrap CIs for each policy."""
    level, reps = float(cfg.stats["ci_level"]), int(cfg.bootstrap_reps)
    rows = []
    for i, (name, assigned) in enumerate(policies.items()):
        ipw = ipw_value(T, Y, assigned, probs)
        aipw = aipw_value(T, Y, assigned, probs, mu)
        lo, hi = bootstrap_ci(lambda b, a=assigned: ipw_value(T[b], Y[b], a[b], probs),
                              len(Y), reps, cfg.seed + i, level)
        alo, ahi = bootstrap_ci(lambda b, a=assigned: aipw_value(T[b], Y[b], a[b], probs, mu[b]),
                                len(Y), reps, cfg.seed + i, level)
        row = {"policy": name, "n": len(Y), "ipw_value": ipw, "ipw_ci_low": lo,
               "ipw_ci_high": hi, "aipw_value": aipw, "aipw_ci_low": alo, "aipw_ci_high": ahi,
               "n_followed": int((T == assigned).sum())}
        for k, arm in enumerate(arm_names):
            row[f"share_{arm}"] = float(np.mean(assigned == k))
        rows.append(row)
    return pd.DataFrame(rows)
