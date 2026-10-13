"""M5: from-scratch estimators, baselines, every learner on synthetic data, leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from equipoise.config import Config
from equipoise.data.splits import get_split
from equipoise.preprocess.discretize import gain_item
from equipoise.supervised.evaluate import classification_metrics, knn_algorithm_timing
from equipoise.supervised.models import (
    class_weight_variants,
    classification_models,
    count_models,
    regression_models,
)
from equipoise.supervised.scratch import (
    ArmMeanRegressor,
    IRLSLogisticRegression,
    NormalEquationsRegressor,
    normal_equations,
)


@pytest.fixture(scope="module")
def frames(_synthetic: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _synthetic.assign(gain_ge_15=gain_item(_synthetic, cfg))
    s = get_split(cfg, df, synthetic=True)
    return s.train(df), s.test(df)


def test_normal_equations_match_lstsq() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 4))
    y = 3 + X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(size=100)
    beta = normal_equations(X, y)
    sk = LinearRegression().fit(X, y)
    np.testing.assert_allclose(beta, np.r_[sk.intercept_, sk.coef_], atol=1e-10)
    np.testing.assert_allclose(NormalEquationsRegressor().fit(X, y).predict(X), sk.predict(X),
                               atol=1e-10)


def test_irls_matches_statsmodels() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3))
    p = 1 / (1 + np.exp(-(0.3 + X @ np.array([1.0, -1.0, 0.5]))))
    y = rng.binomial(1, p)
    ours = IRLSLogisticRegression().fit(X, y)
    ref = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    np.testing.assert_allclose(np.r_[ours.intercept_, ours.coef_], ref.params, atol=1e-8)
    assert ours.converged_ and ours.n_iter_ < 15
    proba = ours.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1)


def test_arm_mean_baseline() -> None:
    X = pd.DataFrame({"treatment": ["A", "A", "B", "B"]})
    m = ArmMeanRegressor().fit(X, [1.0, 3.0, 10.0, 20.0])
    np.testing.assert_allclose(m.predict(pd.DataFrame({"treatment": ["A", "B", "C"]})),
                               [2.0, 15.0, 8.5])


def test_classification_metrics_known_values() -> None:
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
    m = classification_metrics(y, p, 0.5)
    assert m["Recall"] == 0.5 and m["Precision"] == 0.5
    assert m["Specificity"] == pytest.approx(2 / 3) and m["Accuracy"] == 0.6


@pytest.mark.parametrize("factory,target", [
    (regression_models, "Y_va_change_wk52"),
    (classification_models, "gain_ge_15"),
    (classification_models, "Y_laser_yr1"),
    (class_weight_variants, "Y_laser_yr1"),
    (count_models, "Y_n_inj_yr1"),
])
def test_every_learner_runs_on_synthetic(frames, cfg: Config, factory, target: str) -> None:
    train, test = frames
    tr = train.dropna(subset=[target])
    for name, model in factory(cfg).items():
        y = tr[target].to_numpy()
        fitted = model.fit(tr, y.astype(int) if "gain" in target or "laser" in target else y)
        out = (fitted.predict_proba(test)[:, 1] if hasattr(fitted, "predict_proba")
               else fitted.predict(test))
        assert out.shape == (len(test),) and np.isfinite(out).all(), name


def test_pipelines_see_only_baseline_and_treatment(frames, cfg: Config) -> None:
    train, _ = frames
    model = regression_models(cfg)["Ridge (CV alpha)"]
    tr = train.dropna(subset=[cfg.primary_outcome])
    model.fit(tr, tr[cfg.primary_outcome])
    names = [n.split("__", 1)[1] for n in model.named_steps["pre"].get_feature_names_out()]
    assert all(n.startswith(("X_", "treatment_")) for n in names)


def test_knn_algorithms_agree(frames, cfg: Config) -> None:
    from dataclasses import replace

    train, test = frames
    fast = replace(cfg, supervised={**cfg.supervised, "knn_timing_repeats": 1})
    out = knn_algorithm_timing(train, test, cfg.primary_outcome, fast)
    assert set(out.algorithm) == {"kd_tree", "ball_tree", "brute"}
    assert out.same_predictions_as_brute.all()
