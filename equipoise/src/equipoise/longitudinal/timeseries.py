"""Population-level time series of mean vision change per arm (4-week grid).

Each patient has only ~20 visits, too few for per-patient time-series models, so this is a
demonstration on the arm-mean series. The augmented Dickey-Fuller (ADF) test asks whether a
series is stationary (no unit root); a rising recovery curve is not, and differencing
(week-to-week changes) usually fixes that. A moving average smooths noise. AR(p) models
regress a value on its own past p values, MA(q) on past q shocks, and ARIMA(p, d, q) does
this on the d-times differenced series. We choose (p, d, q) by AIC on weeks 0-52 and compare
the forecast for weeks 56-104 with what was observed. With 14 fitting points every result
here is fragile and is reported as such.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from equipoise.config import Config
from equipoise.longitudinal.tensor import TrajectoryTensor


def arm_mean_series(tensor: TrajectoryTensor, cfg: Config) -> pd.DataFrame:
    """Mean vision change from baseline per arm on the 4-week grid (+ n contributing)."""
    step = int(cfg.longitudinal["ts_step_weeks"])
    idx = np.flatnonzero(tensor.weeks % step == 0)
    change = tensor.change_from_baseline("va_letters")[:, idx]
    rows = []
    for arm in cfg.arms:
        m = tensor.treatment == arm
        for j, w in enumerate(tensor.weeks[idx]):
            col = change[m, j]
            rows.append({"arm": arm, "week": int(w), "mean_change": float(np.nanmean(col)),
                         "n": int(np.sum(~np.isnan(col)))})
    return pd.DataFrame(rows)


def adf_table(series: dict[str, np.ndarray]) -> pd.DataFrame:
    """ADF statistic and p-value for each series, raw and first-differenced (maxlag 1)."""
    rows = []
    for name, y in series.items():
        for label, z in (("level", y), ("first difference", np.diff(y))):
            stat, p, *_ = adfuller(z, maxlag=1, autolag=None)
            rows.append({"series": name, "transform": label, "n": len(z), "adf_statistic": stat,
                         "p": p, "stationary_at_0.05": bool(p < 0.05)})
    return pd.DataFrame(rows)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()


def fit_arima_by_aic(y: np.ndarray, cfg: Config
                     ) -> tuple[tuple[int, int, int], pd.DataFrame, object]:
    """Grid over (p, d, q); keep the lowest-AIC model that converged."""
    o = cfg.longitudinal["arima_orders"]
    rows, best, best_fit = [], None, None
    for p, d, q in itertools.product(o["p"], o["d"], o["q"]):
        if p == q == 0 and d == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = ARIMA(y, order=(p, d, q), trend="n" if d else "c").fit()
                aic = float(fit.aic)
            except Exception:  # noqa: BLE001 - tiny series: some orders cannot be estimated
                aic = np.nan
        kind = "AR" if q == 0 and d == 0 else "ARMA" if d == 0 else "ARIMA"
        rows.append({"p": p, "d": d, "q": q, "model": kind, "aic": aic})
        if np.isfinite(aic) and (best is None or aic < rows[best]["aic"]):
            best, best_fit = len(rows) - 1, fit
    table = pd.DataFrame(rows).sort_values("aic")
    r = rows[best]
    return (r["p"], r["d"], r["q"]), table, best_fit
