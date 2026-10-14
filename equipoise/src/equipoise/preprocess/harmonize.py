"""Harmonize central subfield thickness (CST) across OCT machines.

Spectral-domain machines (Cirrus, Spectralis) measure the retina from a deeper boundary than
time-domain Stratus, so the same eye reads thicker on them. Three methods are available:
`zscore` (default) standardizes CST within each machine using means and SDs learned on the
training fold only; `offset` subtracts the median SD-minus-Stratus differences published by
the DRCR network (Bressler et al., JAMA Ophthalmol 2014, +43 um Cirrus, +67 um Spectralis);
`equations` applies the paper's linear Stratus-equivalent equations once the team has filled
their coefficients into config. The published regression coefficients were not available to
us, so z-scoring is the default.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

METHODS = ("zscore", "offset", "equations")


class CSTHarmonizer(TransformerMixin, BaseEstimator):
    """Transform a 2-column frame [CST, machine] into one harmonized CST column.

    Missing CST stays missing. A machine unseen in training (or missing) is mapped with the
    pooled training mean/SD (zscore) or no offset (offset/equations).
    """

    def __init__(self, method: str = "zscore", offset_um: dict[str, float] | None = None,
                 equations: dict[str, dict[str, Any]] | None = None) -> None:
        self.method = method
        self.offset_um = offset_um
        self.equations = equations

    def fit(self, X: pd.DataFrame, y: Any = None) -> CSTHarmonizer:
        if self.method not in METHODS:
            raise ValueError(f"unknown CST method {self.method!r}; choose from {METHODS}")
        cst, machine = self._split(X)
        if self.method == "equations":
            eq = self.equations or {}
            missing = [m for m, c in eq.items() if c.get("slope") is None
                       or c.get("intercept") is None]
            if not eq or missing:
                raise ValueError(
                    "CST method 'equations' needs intercept and slope for every SD machine in "
                    f"config (preprocess.cst.equations); missing: {missing or 'all'}")
        self.pooled_ = (float(np.nanmean(cst)), float(np.nanstd(cst)))
        self.stats_: dict[str, tuple[float, float]] = {}
        for m in np.unique(machine):
            v = cst[(machine == m) & ~np.isnan(cst)]
            if len(v) > 1:  # need at least two scans to estimate a spread
                self.stats_[m] = (float(v.mean()), float(v.std()))
        self.n_features_in_ = 2
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "stats_")
        cst, machine = self._split(X)
        out = np.full(len(cst), np.nan)
        for i, (c, m) in enumerate(zip(cst, machine, strict=True)):
            if np.isnan(c):
                continue
            if self.method == "zscore":
                mu, sd = self.stats_.get(m, self.pooled_)
                out[i] = (c - mu) / (sd if sd > 0 else 1.0)
            elif self.method == "offset":
                out[i] = c - (self.offset_um or {}).get(m, 0.0)
            else:
                coef = (self.equations or {}).get(m)
                out[i] = c if coef is None else coef["intercept"] + coef["slope"] * c
        return out.reshape(-1, 1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array([f"X_cst_harmonized_{self.method}"], dtype=object)

    @staticmethod
    def _split(X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=object)
        cst = pd.to_numeric(pd.Series(arr[:, 0]), errors="coerce").to_numpy(dtype=float)
        machine = pd.Series(arr[:, 1]).astype(object).where(pd.notna(arr[:, 1]), "missing")
        return cst, machine.astype(str).to_numpy()


def harmonizer_from_config(cfg_cst: dict[str, Any], method: str | None = None
                           ) -> CSTHarmonizer:
    """Build the harmonizer from `config.preprocess.cst` (optionally overriding the method)."""
    return CSTHarmonizer(method or cfg_cst["method"], dict(cfg_cst["offset_um"]),
                         dict(cfg_cst["equations"]))
