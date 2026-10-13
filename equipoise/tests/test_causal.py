"""M6: propensities, learners, semi-synthetic harness, policy value, tuning, bundle."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from equipoise.causal.data import KnownPropensity, arm_codes, known_propensities, prepare
from equipoise.causal.learners import ConstantEffect, EconMLMeta, learner_zoo
from equipoise.causal.policy import aipw_value, ipw_value
from equipoise.causal.semisynth import (
    fit_simulator,
    run_benchmark,
    scenario_effects,
    simulate,
    summarize,
    tertile_gap,
)
from equipoise.causal.tuning import box, compare_tuners, decode, grid_points
from equipoise.config import Config
from equipoise.data.splits import get_split
from equipoise.search.algorithms import continuous_pso


@pytest.fixture(scope="module")
def light(cfg: Config) -> Config:
    c = {**cfg.causal, "bootstrap_reps": 3,
         "semi_synthetic": {**cfg.causal["semi_synthetic"], "replications": 2},
         "tuning": {**cfg.causal["tuning"], "trials": 3},
         "tarnet": {**cfg.causal["tarnet"], "max_epochs": 5},
         "causal_forest": {**cfg.causal["causal_forest"], "n_estimators": 100}}
    return replace(cfg, causal=c)


@pytest.fixture(scope="module")
def data(_synthetic: pd.DataFrame, light: Config):
    s = get_split(light, _synthetic, synthetic=True)
    return prepare(s.train(_synthetic), s.test(_synthetic), light)


def test_known_propensities(cfg: Config) -> None:
    p = known_propensities(cfg)
    assert p.sum() == pytest.approx(1) and arm_codes(cfg)[cfg.reference_arm] == 0
    np.testing.assert_allclose(p, [218 / 660, 224 / 660, 218 / 660])
    kp = KnownPropensity(tuple(p)).fit(None, np.array([0, 2, 0, 2]))
    np.testing.assert_allclose(kp.predict_proba(np.zeros((2, 1))),
                               [[0.5, 0.5], [0.5, 0.5]])          # 218 vs 218 renormalized


def test_constant_effect_is_difference_in_means() -> None:
    T = np.array([0, 0, 1, 1, 2, 2])
    Y = np.array([1.0, 3.0, 6.0, 8.0, 2.0, 4.0])
    m = ConstantEffect.__new__(ConstantEffect)
    m.n_arms = 3
    np.testing.assert_allclose(m.fit(None, T, Y).effect(np.zeros((4, 1))),
                               np.tile([5.0, 1.0], (4, 1)))


def test_every_learner_runs(data, light: Config) -> None:
    tr, te, _ = data
    for name, factory in learner_zoo(light).items():
        e = factory(light).fit(tr.X, tr.T, tr.Y).effect(te.X)
        assert e.shape == (len(te), 2) and np.isfinite(e).all(), name


def test_scenarios_and_simulation(data, light: Config) -> None:
    tr, _, _ = data
    taus = scenario_effects(tr.frame, light)
    assert np.ptp(taus["a_null"], axis=0).max() == 0                     # no heterogeneity
    below = (tr.frame.X_va_below_69 == 1).to_numpy()
    assert taus["b_published_like"][below, 0].mean() > taus["b_published_like"][~below, 0].mean()
    sim = fit_simulator(tr, light)
    p = known_propensities(light)
    a = simulate(sim, "c_nonlinear", np.random.default_rng(0), p)
    b = simulate(sim, "c_nonlinear", np.random.default_rng(0), p)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert tertile_gap(np.arange(9.0)) == pytest.approx(6.0)


def test_harness_recovers_planted_effect_better_than_null_model(data, light: Config) -> None:
    """Scenario (b): when the planted effect is detectable (noise reduced), the harness shows
    a heterogeneous learner beating the constant-effect model. At the real noise level this
    need not hold with ~370 patients - which is itself a reported finding of M6."""
    tr, _, _ = data
    quiet = replace(light, causal={**light.causal, "semi_synthetic": {
        **light.causal["semi_synthetic"], "noise_scale": 0.2}})
    learners = {"constant": ConstantEffect,
                "X-learner (GB)": lambda c: EconMLMeta(c, "X", "gb")}
    s = summarize(run_benchmark(tr, learners, quiet)[0]).set_index(["scenario", "learner"])
    assert s.loc[("b_published_like", "X-learner (GB)"), "pehe_mean"] < \
        s.loc[("b_published_like", "constant"), "pehe_mean"]


def test_ipw_policy_value_is_unbiased() -> None:
    """With known propensities, the IPW estimate averages to the true policy value."""
    rng = np.random.default_rng(0)
    n, probs = 400, np.array([0.33, 0.34, 0.33])
    mu = np.column_stack([rng.normal(10, 3, n), rng.normal(13, 3, n), rng.normal(11, 3, n)])
    policy = np.argmax(mu + rng.normal(0, 2, mu.shape), axis=1)
    truth = mu[np.arange(n), policy].mean()
    estimates = []
    for _ in range(400):
        T = rng.choice(3, n, p=probs)
        Y = mu[np.arange(n), T] + rng.normal(0, 5, n)
        estimates.append(ipw_value(T, Y, policy, probs))
    se = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - truth) < 4 * se
    T = rng.choice(3, n, p=probs)
    Y = mu[np.arange(n), T]
    assert aipw_value(T, Y, policy, probs, mu) == pytest.approx(truth)   # perfect outcome model


def test_continuous_pso_minimises_quadratic() -> None:
    best, val, hist = continuous_pso(lambda z: float(np.sum((z - 1.5) ** 2)),
                                     np.array([[-5.0, 5.0], [-5.0, 5.0]]),
                                     np.random.default_rng(0), particles=10, iterations=40)
    assert val < 1e-3 and np.allclose(best, 1.5, atol=0.05)
    assert hist == sorted(hist, reverse=True)


def test_tuning_space_and_comparison(data, light: Config) -> None:
    assert decode("ridge", np.array([np.log(10.0)])) == {"alpha": pytest.approx(10.0)}
    assert box("gb").shape == (4, 2) and len(grid_points("gb", 30)) == 30
    tr, _, _ = data
    hist, summary, best = compare_tuners(tr, light, "ridge")
    assert set(summary.method) == {"Grid search", "Optuna (TPE)", "PSO"}
    assert hist.groupby("method").trial.max().eq(3).all() and "alpha" in best


def test_selection_rule_excludes_constant_and_biased(cfg: Config) -> None:
    from equipoise.causal.run import select_learner

    rows = []
    spec = {  # learner: (pehe per scenario, ate bias)
        "constant": ((1.0, 2.0, 2.0), 0.0),        # best mean, but cannot vary by patient
        "shrunk": ((1.5, 2.1, 2.1), 1.5),          # biased average effect
        "robust": ((2.0, 2.4, 2.3), 0.2),          # lowest worst case among candidates
        "spiky": ((1.2, 1.3, 3.5), 0.1),           # better mean, worse worst case
    }
    for name, (pehes, bias) in spec.items():
        for sc, v in zip(["a_null", "b_published_like", "c_nonlinear"], pehes, strict=True):
            rows.append({"scenario": sc, "learner": name, "pehe_mean": v, "ate_bias_mean": bias})
    sel = select_learner(pd.DataFrame(rows),
                         {"constant": 0.0, "shrunk": 1.0, "robust": 1.0, "spiky": 1.0}, cfg)
    s = sel.set_index("learner")
    assert s.loc["robust", "chosen"] and sel.chosen.sum() == 1
    assert "by construction" in s.loc["constant", "status"]
    assert "bias" in s.loc["shrunk", "status"]
