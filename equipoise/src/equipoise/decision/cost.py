"""Expected cost per drug and cost per letter gained.

For each drug, expected year-1 cost = predicted injections x price per injection +
P(rescue laser) x price of a laser session. Cost per extra letter versus bevacizumab (an
incremental cost-effectiveness ratio) = (cost of drug - cost of bevacizumab) / (letters with
drug - letters with bevacizumab); when the drug is not predicted to give more letters the
ratio is undefined and reported as such. Prices come from config/costs.yaml or are the
clearly labelled DEMO relative units.
"""

from __future__ import annotations

import numpy as np

from equipoise.config import Prices


def expected_cost(injections: np.ndarray, p_laser: np.ndarray, arms: list[str],
                  prices: Prices) -> np.ndarray:
    """(n, n_arms) expected year-1 cost in the price units."""
    per = np.array([prices.per_injection[a] for a in arms])
    return injections * per + p_laser * prices.laser_session


def cost_per_letter(cost: np.ndarray, letters: np.ndarray) -> np.ndarray:
    """(n, n_arms) incremental cost per extra letter vs the reference (column 0); NaN where the
    drug gives no extra letters (or for the reference itself)."""
    d_cost = cost - cost[:, [0]]
    d_letters = letters - letters[:, [0]]
    out = np.full(cost.shape, np.nan)
    gain = d_letters > 0
    out[gain] = d_cost[gain] / d_letters[gain]
    return out
