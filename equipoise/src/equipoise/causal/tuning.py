"""Tune the chosen learner's base model: grid search vs Optuna (TPE) vs PSO, same budget.

A CATE learner's quality cannot be scored on real data (individual effects are unobserved),
so we tune what can be checked: its base outcome model. The objective is the 5-fold CV RMSE
of a per-arm outcome model (T-learner style) predicting each patient's observed letters, on
training patients only. Grid search tries a fixed lattice of settings; Optuna's TPE builds a
probability model of good vs bad settings and samples where good ones are likely; PSO moves a
swarm through the (continuous, log-scaled) hyperparameter box. Each gets the same number of
trials and seed, and we record the best value after every trial.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from equipoise.causal.data import CausalData
from equipoise.causal.learners import base_learner
from equipoise.config import Config
from equipoise.search.algorithms import continuous_pso

# name -> (kind, low, high); "log" floats are searched on a log scale, "int" are rounded
SPACES: dict[str, dict[str, tuple[str, float, float]]] = {
    "gb": {"learning_rate": ("log", 0.01, 0.3), "max_depth": ("int", 1, 4),
           "min_samples_leaf": ("int", 5, 60), "n_estimators": ("int", 50, 400)},
    "ridge": {"alpha": ("log", 0.01, 1000.0)},
}
GRIDS: dict[str, dict[str, list[Any]]] = {
    "gb": {"learning_rate": [0.02, 0.05, 0.1], "max_depth": [1, 2, 3],
           "min_samples_leaf": [10, 40], "n_estimators": [100, 300]},
}


def decode(base: str, z: np.ndarray) -> dict[str, Any]:
    """Map a point of the unit-free search box to hyperparameters."""
    out: dict[str, Any] = {}
    for (name, (kind, lo, hi)), val in zip(SPACES[base].items(), z, strict=True):
        if kind == "log":
            out[name] = float(np.exp(val))
        else:
            out[name] = int(round(np.clip(val, lo, hi)))
    return out


def box(base: str) -> np.ndarray:
    return np.array([[np.log(lo), np.log(hi)] if kind == "log" else [lo, hi]
                     for kind, lo, hi in SPACES[base].values()], float)


def grid_points(base: str, n: int) -> list[dict[str, Any]]:
    """The first n points of a lattice (full factorial for GB; log-spaced alphas for Ridge)."""
    if base in GRIDS:
        names = list(GRIDS[base])
        combos = list(itertools.product(*GRIDS[base].values()))
        return [dict(zip(names, c, strict=True)) for c in combos[:n]]
    _, lo, hi = SPACES[base]["alpha"]
    return [{"alpha": float(a)} for a in np.logspace(np.log10(lo), np.log10(hi), n)]


def make_objective(data: CausalData, cfg: Config, base: str) -> Callable[[dict], float]:
    """CV RMSE (letters) of per-arm outcome models with the given hyperparameters."""
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    folds = list(skf.split(data.X, data.T))

    def objective(params: dict[str, Any]) -> float:
        model = base_learner(base, cfg, params)
        sq = 0.0
        for i_tr, i_va in folds:
            pred = np.empty(len(i_va))
            for a in np.unique(data.T):
                tr = i_tr[data.T[i_tr] == a]
                va = data.T[i_va] == a
                m = clone(model).fit(data.X[tr], data.Y[tr])
                pred[va] = m.predict(data.X[i_va][va])
            sq += float(np.sum((data.Y[i_va] - pred) ** 2))
        return float(np.sqrt(sq / len(data.Y)))

    return objective


def compare_tuners(data: CausalData, cfg: Config, base: str
                   ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run grid, Optuna-TPE and PSO with the same budget.

    Returns (per-trial history, summary, best params overall).
    """
    t = cfg.causal["tuning"]
    budget = int(t["trials"])
    objective = make_objective(data, cfg, base)
    default = dict(cfg.causal["base_learners"][base])
    hist, best_params = [], {}

    def record(method: str, values: list[float], params: list[dict]) -> None:
        best = np.inf
        for i, (v, p) in enumerate(zip(values, params, strict=True)):
            best = min(best, v)
            hist.append({"method": method, "trial": i + 1, "value": v, "best_so_far": best,
                         "params": str(p)})
        k = int(np.argmin(values))
        best_params[method] = (values[k], params[k])

    pts = grid_points(base, budget)
    record("Grid search", [objective(p) for p in pts], pts)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed))

    def opt_obj(trial: optuna.Trial) -> float:
        p: dict[str, Any] = {}
        for name, (kind, lo, hi) in SPACES[base].items():
            p[name] = (trial.suggest_float(name, lo, hi, log=True) if kind == "log"
                       else trial.suggest_int(name, int(lo), int(hi)))
        return objective(p)

    study.optimize(opt_obj, n_trials=budget)
    record("Optuna (TPE)", [tr.value for tr in study.trials], [tr.params for tr in study.trials])

    pso_vals, pso_params = [], []

    def pso_obj(z: np.ndarray) -> float:
        p = decode(base, z)
        v = objective(p)
        pso_vals.append(v)
        pso_params.append(p)
        return v

    n_part = int(t["pso_particles"])
    continuous_pso(pso_obj, box(base), np.random.default_rng(cfg.seed), n_part,
                   int(np.ceil(budget / n_part)))
    record("PSO", pso_vals[:budget], pso_params[:budget])

    default_value = objective(default)
    summary = pd.DataFrame([{"method": m, "trials": budget, "best_cv_rmse": v,
                             "best_params": str(p), "default_cv_rmse": default_value}
                            for m, (v, p) in best_params.items()])
    winner = min(best_params.values(), key=lambda vp: vp[0])
    return pd.DataFrame(hist), summary, winner[1] if winner[0] < default_value else default
