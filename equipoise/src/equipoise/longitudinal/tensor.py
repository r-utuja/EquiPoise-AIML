"""Align irregular visits onto a common weekly grid: a 3-D tensor patients x weeks x measures.

Patients are seen on slightly different days, so to compare or average trajectories we
resample each patient's study-eye vision and CST onto the same weekly grid. Linear
interpolation joins consecutive visits with straight lines; a cubic spline fits smooth
piecewise cubics with continuous first and second derivatives. We never extrapolate: weeks
before the first or after the last visit stay missing (NaN). Injections are events, not
measurements, so they are placed on the nearest week rather than interpolated. Lagrange
interpolation fits one polynomial through all points; with many points it oscillates wildly
between them (Runge's phenomenon), which is why splines are used instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from equipoise.config import Config

MEASURES = ("va_letters", "cst_um", "injected")


@dataclass
class TrajectoryTensor:
    """values[i, t, m]: patient i, week grid[t], measure MEASURES[m]."""

    pt_ids: np.ndarray
    treatment: np.ndarray
    weeks: np.ndarray
    values: np.ndarray
    method: str

    def measure(self, name: str) -> np.ndarray:
        return self.values[:, :, MEASURES.index(name)]

    def change_from_baseline(self, name: str = "va_letters") -> np.ndarray:
        x = self.measure(name)
        return x - x[:, [0]]


def interpolate(days: np.ndarray, values: np.ndarray, grid_days: np.ndarray, method: str
                ) -> np.ndarray:
    """Interpolate one patient's series onto grid_days (NaN outside the observed range)."""
    ok = ~np.isnan(values)
    d, v = days[ok], values[ok]
    out = np.full(len(grid_days), np.nan)
    if len(d) == 0:
        return out
    d, idx = np.unique(d, return_index=True)          # one value per day
    v = v[idx]
    inside = (grid_days >= d[0]) & (grid_days <= d[-1])
    if len(d) == 1:
        out[grid_days == d[0]] = v[0]
        return out
    if method == "linear" or len(d) < 4:
        out[inside] = np.interp(grid_days[inside], d, v)
    elif method == "spline":
        out[inside] = CubicSpline(d, v, bc_type="natural")(grid_days[inside])
    else:
        raise ValueError(f"unknown interpolation method {method!r}")
    return out


def build_tensor(visits: pd.DataFrame, cfg: Config, method: str = "linear",
                 pt_ids: np.ndarray | None = None) -> TrajectoryTensor:
    """Resample every patient onto the weekly grid (vision, CST interpolated; injections
    placed at the nearest week)."""
    lc = cfg.longitudinal
    weeks = np.arange(0, int(lc["grid_max_week"]) + 1, int(lc["grid_step_weeks"]))
    grid_days = weeks * 7.0
    v = visits if pt_ids is None else visits[visits.pt_id.isin(pt_ids)]
    groups = v.sort_values(["pt_id", "days_from_rand"]).groupby("pt_id", sort=True)
    ids, arms, cube = [], [], []
    for pid, g in groups:
        days = g.days_from_rand.to_numpy(float)
        sheet = np.full((len(weeks), len(MEASURES)), np.nan)
        sheet[:, 0] = interpolate(days, g.va_letters.to_numpy(float), grid_days, method)
        sheet[:, 1] = interpolate(days, g.cst_um.to_numpy(float), grid_days, method)
        inj = np.zeros(len(weeks))
        last = days.max()
        inj[weeks * 7 > last] = np.nan
        for d in days[g.injected.to_numpy() == 1]:
            inj[int(np.argmin(np.abs(grid_days - d)))] = 1
        sheet[:, 2] = inj
        ids.append(pid)
        arms.append(g.treatment.iloc[0])
        cube.append(sheet)
    return TrajectoryTensor(np.array(ids), np.array(arms), weeks, np.stack(cube), method)


def lagrange_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Power-basis coefficients (highest degree first) of the Lagrange polynomial through
    (x, y): sum_i y_i * prod_{j != i} (t - x_j) / (x_i - x_j). Written out, not scipy."""
    n = len(x)
    coef = np.zeros(n)
    for i in range(n):
        basis = np.array([1.0])
        denom = 1.0
        for j in range(n):
            if j != i:
                basis = np.convolve(basis, [1.0, -x[j]])
                denom *= x[i] - x[j]
        coef += y[i] * basis / denom
    return coef


def lagrange_eval(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate the Lagrange interpolating polynomial directly (numerically stabler than
    expanding the coefficients): L(t) = sum_i y_i * prod_{j != i} (t - x_j) / (x_i - x_j)."""
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    for i in range(len(x)):
        li = np.ones_like(t)
        for j in range(len(x)):
            if j != i:
                li *= (t - x[j]) / (x[i] - x[j])
        out += y[i] * li
    return out
