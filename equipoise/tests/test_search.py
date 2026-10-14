"""M4: objective pieces, budget handling, and every search algorithm on a known optimum."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from equipoise.config import Config
from equipoise.data.splits import get_split
from equipoise.search.algorithms import ALGORITHMS, run_algorithm
from equipoise.search.objective import (
    BudgetExhausted,
    Evaluator,
    FoldData,
    column_owner,
    make_evaluator,
    ridge_fit_predict,
)
from equipoise.search.selection import rank_features

N_FEATURES, SIGNAL = 12, (0, 3, 7)


@pytest.fixture(scope="module")
def toy() -> Evaluator:
    """12 features; only 0, 3 and 7 drive y, so the optimum mask is exactly those three."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, N_FEATURES))
    y = X[:, list(SIGNAL)] @ np.array([3.0, -2.0, 2.5]) + rng.normal(size=400)
    idx = np.arange(400)
    folds = []
    for k in range(4):
        va = idx[k::4]
        tr = np.setdiff1d(idx, va)
        folds.append((X[tr], y[tr], X[va], y[va]))
    data = FoldData(folds, [f"f{i}" for i in range(N_FEATURES)],
                    [np.array([i]) for i in range(N_FEATURES)], np.array([], dtype=int))
    return Evaluator(data, alpha=1.0, penalty=0.05, budget=600)


def test_ridge_closed_form_matches_sklearn() -> None:
    rng = np.random.default_rng(3)
    X, Xv = rng.normal(size=(80, 5)), rng.normal(size=(20, 5))
    y = rng.normal(size=80)
    np.testing.assert_allclose(ridge_fit_predict(X, y, Xv, 2.5),
                               Ridge(alpha=2.5).fit(X, y).predict(Xv), atol=1e-10)


def test_column_owner_maps_every_column() -> None:
    names = ["numeric__X_age", "numeric__X_cst_harmonized_zscore", "nominal__X_race_White",
             "binary__X_prior_prp", "treatment__treatment_Aflibercept"]
    raw = ["X_age", "X_cst_um", "X_race", "X_prior_prp", "X_prior_antivegf"]
    assert column_owner(names, raw) == ["X_age", "X_cst_um", "X_race", "X_prior_prp", None]


def test_evaluator_memoises_and_enforces_budget(toy: Evaluator) -> None:
    ev = replace(toy.fresh(), budget=3)
    m = np.zeros(N_FEATURES, np.int8)
    v1 = ev(m)
    assert ev(m) == v1 and ev.n_evals == 1          # cached: no budget used
    ev(np.ones(N_FEATURES, np.int8))
    m2 = m.copy()
    m2[0] = 1
    ev(m2)
    with pytest.raises(BudgetExhausted):
        ev(np.eye(N_FEATURES, dtype=np.int8)[5])
    assert list(ev.history) == sorted(ev.history, reverse=True)   # best-so-far never worsens


def test_call_cap_stops_cycling(toy: Evaluator) -> None:
    ev = replace(toy.fresh(), budget=10, call_factor=2)
    m = np.zeros(N_FEATURES, np.int8)
    with pytest.raises(BudgetExhausted):
        for _ in range(100):
            ev(m)                                       # the same mask forever
    assert "call cap" in ev.stopped_by


@pytest.mark.parametrize("name", [n for n in ALGORITHMS if n != "Random search"])
def test_algorithms_find_known_optimum(toy: Evaluator, name: str) -> None:
    params = {"beam": {"width": 4}, "tabu": {"tenure": 4},
              "ga": {"population": 20, "elite": 2, "patience_generations": 30},
              "pso": {"particles": 15}}
    res = run_algorithm(name, toy, seed=1, params=params)
    assert set(np.flatnonzero(res.best_mask)) == set(SIGNAL), name
    assert res.n_evals <= toy.budget


def test_algorithms_are_reproducible(toy: Evaluator) -> None:
    for name in ALGORITHMS:
        a = run_algorithm(name, toy, seed=7, params={})
        b = run_algorithm(name, toy, seed=7, params={})
        assert a.history == b.history and np.array_equal(a.best_mask, b.best_mask), name


def test_real_objective_on_synthetic(synthetic: pd.DataFrame, cfg: Config) -> None:
    train = get_split(cfg, synthetic, synthetic=True).train(synthetic)
    ev = make_evaluator(train, cfg)
    assert ev.n_features == len([c for c in synthetic.columns if c.startswith("X_")])
    full, empty = np.ones(ev.n_features, np.int8), np.zeros(ev.n_features, np.int8)
    assert ev.cv_rmse(full) > 0 and ev.cv_rmse(empty) > 0
    covered = np.concatenate(ev.data.columns_of + [ev.data.always])
    assert sorted(covered) == list(range(ev.data.folds[0][0].shape[1]))


def test_rankings_put_baseline_vision_on_top(synthetic: pd.DataFrame, cfg: Config) -> None:
    """The synthetic outcome is driven by baseline vision, so every method should see it."""
    train = get_split(cfg, synthetic, synthetic=True).train(synthetic)
    ranks, _ = rank_features(train, cfg)
    assert "X_va_letters" in list(ranks.feature.head(3))
