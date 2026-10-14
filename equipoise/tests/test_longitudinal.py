"""M7: interpolation, tensor, Lagrange, finite differences, HMM, early-response guard."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import lagrange as scipy_lagrange

from equipoise.config import Config
from equipoise.data.splits import get_split
from equipoise.features import LeakageError, assert_no_leakage
from equipoise.longitudinal.clusters import cluster_trajectories
from equipoise.longitudinal.derivatives import (
    backward_difference,
    central_difference,
    error_orders,
    forward_difference,
)
from equipoise.longitudinal.early_response import early_features
from equipoise.longitudinal.hmm import (
    fit_hmm,
    forward_loglik,
    label_states,
    split_sequences,
    visit_sequences,
    viterbi,
)
from equipoise.longitudinal.tensor import (
    build_tensor,
    interpolate,
    lagrange_coefficients,
    lagrange_eval,
)
from equipoise.longitudinal.timeseries import adf_table, arm_mean_series

SRC = Path(__file__).resolve().parents[1] / "src" / "equipoise"


def test_interpolation_never_extrapolates() -> None:
    days, vals = np.array([0.0, 10.0, 20.0, 30.0]), np.array([50.0, 60.0, 55.0, 70.0])
    grid = np.array([-5.0, 0.0, 5.0, 25.0, 30.0, 40.0])
    lin = interpolate(days, vals, grid, "linear")
    np.testing.assert_allclose(lin[1:5], [50.0, 55.0, 62.5, 70.0])
    assert np.isnan(lin[0]) and np.isnan(lin[5])
    spl = interpolate(days, vals, days, "spline")
    np.testing.assert_allclose(spl, vals)                     # passes through the visits


def test_tensor_shape_and_baseline(synthetic: pd.DataFrame, synthetic_visits: pd.DataFrame,
                                   cfg: Config) -> None:
    t = build_tensor(synthetic_visits, cfg)
    assert t.values.shape == (synthetic.pt_id.nunique(), 105, 3)
    base = synthetic.set_index("pt_id").loc[t.pt_ids, "X_va_letters"].to_numpy()
    np.testing.assert_allclose(t.measure("va_letters")[:, 0], base)
    assert np.all(np.nan_to_num(t.change_from_baseline()[:, 0]) == 0)


def test_lagrange_matches_scipy_and_interpolates() -> None:
    x = np.array([-2.0, -1.0, 0.5, 1.0, 3.0])
    y = np.array([3.0, -1.0, 2.0, 0.0, 5.0])
    np.testing.assert_allclose(lagrange_coefficients(x, y), scipy_lagrange(x, y).coeffs,
                               atol=1e-10)
    np.testing.assert_allclose(lagrange_eval(x, y, x), y, atol=1e-10)


def test_finite_differences() -> None:
    t = np.arange(0, 10.0, 0.5)
    y = 3 * t + 1
    for fn in (forward_difference, backward_difference, central_difference):
        d = fn(y, 0.5)
        np.testing.assert_allclose(d[~np.isnan(d)], 3.0)
    orders = error_orders().groupby("scheme").observed_order.last()
    assert orders["central"] == pytest.approx(2, abs=0.15)
    assert orders["forward"] == pytest.approx(1, abs=0.15)


@pytest.fixture(scope="module")
def hmm_setup(_synthetic: pd.DataFrame, cfg: Config):
    from equipoise.data.load import load_visits

    visits = load_visits(cfg, synthetic=True)
    ids = np.array(get_split(cfg, _synthetic, synthetic=True).train_ids)
    seq = visit_sequences(visits, ids)
    return seq, fit_hmm(seq, cfg)


def test_forward_algorithm_matches_hmmlearn(hmm_setup) -> None:
    seq, model = hmm_setup
    for x in split_sequences(seq)[:40]:
        assert forward_loglik(model, x) == pytest.approx(model.score(x), abs=1e-8)


def test_viterbi_matches_hmmlearn(hmm_setup) -> None:
    seq, model = hmm_setup
    for x in split_sequences(seq)[:40]:
        path, logp = viterbi(model, x)
        ref_logp, ref_path = model.decode(x, algorithm="viterbi")
        assert np.array_equal(path, ref_path) and logp == pytest.approx(ref_logp, abs=1e-8)
    assert set(label_states(model).values()) == {"improving", "stable", "active"}


def test_early_features_are_gated(synthetic: pd.DataFrame, synthetic_visits: pd.DataFrame,
                                  cfg: Config) -> None:
    e = early_features(synthetic_visits, synthetic, cfg)
    cols = [c for c in e.columns if c != "pt_id"]
    assert cols and all(c.startswith("E_") for c in cols)
    with pytest.raises(LeakageError):
        assert_no_leakage(cols)                               # never in baseline models
    assert_no_leakage(cols, allow_early=True)


def test_only_early_response_module_uses_early_vision() -> None:
    users = [p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
             if p.name != "features.py"     # defines (and documents) the switch
             and re.search(r"allow_early\s*=\s*True", p.read_text(encoding="utf-8"))]
    assert users == ["longitudinal/early_response.py"]


def test_series_and_clusters(synthetic_visits: pd.DataFrame, cfg: Config) -> None:
    t = build_tensor(synthetic_visits, cfg)
    s = arm_mean_series(t, cfg)
    assert set(s.arm) == set(cfg.arms) and (s[s.week == 0].mean_change == 0).all()
    adf = adf_table({a: g.mean_change.to_numpy()[:14] for a, g in s.groupby("arm")})
    assert len(adf) == 2 * len(cfg.arms)
    tc = cluster_trajectories(t, cfg)
    assert set(tc.summary.cluster) == {"fast responders", "slow responders", "non-responders"}
    fast = tc.summary.set_index("cluster").loc["fast responders", "mean_gain_wk52"]
    non = tc.summary.set_index("cluster").loc["non-responders", "mean_gain_wk52"]
    assert fast > non
