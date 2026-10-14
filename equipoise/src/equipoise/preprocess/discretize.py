"""Discretize baseline features into clinically meaningful bands.

HbA1c, baseline vision (Snellen-equivalent bands; 69 letters is the published 20/50 cut) and
age are cut at fixed edges from config, so the bands mean the same thing in every fold. The
bands feed M3's association rules, which need categorical items rather than numbers.
"""

from __future__ import annotations

import pandas as pd

from equipoise.config import Config


def discretize(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One categorical `<feature>_band` column per configured feature (right-open bins)."""
    out = pd.DataFrame(index=df.index)
    for col, spec in cfg.preprocess["bands"].items():
        out[f"{col}_band"] = pd.cut(df[col], bins=spec["edges"], labels=spec["labels"],
                                    right=False)
    return out


def gain_item(df: pd.DataFrame, cfg: Config, outcome: str | None = None) -> pd.Series:
    """1 if letters gained >= threshold (default 15 = 3 lines), NaN when the outcome is missing.

    This is an outcome, so it is only ever used as a target / rule consequent.
    """
    y = df[outcome or cfg.primary_outcome]
    return (y >= cfg.preprocess["gain_threshold_letters"]).astype(float).where(y.notna())
