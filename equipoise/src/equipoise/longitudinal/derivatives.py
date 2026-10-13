"""Rates of change by finite differences (letters per week).

On a grid with step h, the forward difference (f(t+h) - f(t)) / h and the backward difference
(f(t) - f(t-h)) / h estimate the slope with an error proportional to h (first order); the
central difference (f(t+h) - f(t-h)) / 2h cancels the leading error term, so its error is
proportional to h^2 (second order). We apply them to the arm-mean vision trajectories and
verify the error orders on a function whose derivative is known exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_difference(y: np.ndarray, h: float) -> np.ndarray:
    d = np.full_like(y, np.nan, dtype=float)
    d[:-1] = (y[1:] - y[:-1]) / h
    return d


def backward_difference(y: np.ndarray, h: float) -> np.ndarray:
    d = np.full_like(y, np.nan, dtype=float)
    d[1:] = (y[1:] - y[:-1]) / h
    return d


def central_difference(y: np.ndarray, h: float) -> np.ndarray:
    d = np.full_like(y, np.nan, dtype=float)
    d[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    return d


SCHEMES = {"forward": forward_difference, "backward": backward_difference,
           "central": central_difference}


def error_orders(steps: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5, 0.25)) -> pd.DataFrame:
    """Max error of each scheme for f(t) = 15 (1 - exp(-t / 10)) on [8, 40] weeks, a curve
    shaped like vision recovery, and the observed convergence order log2(e(2h) / e(h))."""
    def f(t: np.ndarray) -> np.ndarray:
        return 15 * (1 - np.exp(-t / 10))

    def df(t: np.ndarray) -> np.ndarray:
        return 1.5 * np.exp(-t / 10)

    rows = []
    for name, fn in SCHEMES.items():
        prev = None
        for h in steps:
            t = np.arange(8, 40 + h / 2, h)
            est = fn(f(t), h)
            err = float(np.nanmax(np.abs(est - df(t))))
            rows.append({"scheme": name, "h_weeks": h, "max_abs_error": err,
                         "observed_order": np.log2(prev / err) if prev else np.nan})
            prev = err
    return pd.DataFrame(rows)
