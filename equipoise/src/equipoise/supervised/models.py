"""Model zoo for M5: regression, classification and count models as sklearn Pipelines.

Every model except the baselines is `Pipeline([build_preprocessor(...), estimator])`, so
imputation, harmonization, encoding and scaling are re-fitted inside each training fold.
Inputs are the baseline X_ features plus the randomized arm (one-hot, reference bevacizumab),
because outcome models must know which drug the patient received.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LassoCV,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    RidgeCV,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from equipoise.config import Config
from equipoise.preprocess.encode import build_preprocessor
from equipoise.supervised.scratch import (
    ArmMeanRegressor,
    ArmRateClassifier,
    IRLSLogisticRegression,
    NormalEquationsRegressor,
)


def _pipe(cfg: Config, est: Any, full_rank: bool = False) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(cfg, include_treatment=True,
                                                full_rank=full_rank)),
                     ("model", est)])


def regression_models(cfg: Config) -> dict[str, Any]:
    s, seed = cfg.supervised, cfg.seed
    return {
        "Baseline: arm mean": ArmMeanRegressor(),
        "OLS (sklearn)": _pipe(cfg, LinearRegression(), full_rank=True),
        "OLS (normal equations, numpy)": _pipe(cfg, NormalEquationsRegressor(), full_rank=True),
        "Ridge (CV alpha)": _pipe(cfg, RidgeCV(alphas=np.array(s["ridge_alphas"], float))),
        "Lasso (CV alpha)": _pipe(cfg, LassoCV(alphas=int(s["lasso_n_alphas"]), cv=5,
                                               random_state=seed)),
        "kNN": _pipe(cfg, KNeighborsRegressor(n_neighbors=int(s["knn_neighbors"]))),
        "Decision tree": _pipe(cfg, DecisionTreeRegressor(**s["tree"], random_state=seed)),
        "Random forest": _pipe(cfg, RandomForestRegressor(**s["random_forest"],
                                                          random_state=seed, n_jobs=1)),
        "Gradient boosting": _pipe(cfg, GradientBoostingRegressor(**s["gradient_boosting"],
                                                                  random_state=seed)),
    }


def classification_models(cfg: Config) -> dict[str, Any]:
    s, seed = cfg.supervised, cfg.seed
    irls = s["irls"]
    return {
        "Baseline: arm event rate": ArmRateClassifier(),
        "Baseline: majority class": DummyClassifier(strategy="prior"),
        "Logistic regression": _pipe(cfg, LogisticRegression(C=float(s["logistic_C"]),
                                                             max_iter=2000)),
        "Logistic regression (IRLS, scratch)": _pipe(
            cfg, IRLSLogisticRegression(int(irls["max_iter"]), float(irls["tol"])),
            full_rank=True),
        "kNN": _pipe(cfg, KNeighborsClassifier(n_neighbors=int(s["knn_neighbors"]))),
        "Decision tree": _pipe(cfg, DecisionTreeClassifier(**s["tree"], random_state=seed)),
        "Gaussian Naive Bayes": _pipe(cfg, GaussianNB()),
        "Bernoulli Naive Bayes": _pipe(cfg, BernoulliNB(binarize=0.0)),
    }


def class_weight_variants(cfg: Config) -> dict[str, Any]:
    """Logistic regression and tree with and without class_weight='balanced'."""
    s, seed = cfg.supervised, cfg.seed
    out = {}
    for cw in (None, "balanced"):
        tag = "balanced" if cw else "none"
        out[f"Logistic regression [class_weight={tag}]"] = _pipe(
            cfg, LogisticRegression(C=float(s["logistic_C"]), max_iter=2000, class_weight=cw))
        out[f"Decision tree [class_weight={tag}]"] = _pipe(
            cfg, DecisionTreeClassifier(**s["tree"], random_state=seed, class_weight=cw))
    return out


def count_models(cfg: Config) -> dict[str, Any]:
    s = cfg.supervised
    return {
        "Baseline: arm mean": ArmMeanRegressor(),
        "Poisson GLM": _pipe(cfg, PoissonRegressor(alpha=float(s["poisson_alpha"]),
                                                   max_iter=1000)),
        "Gradient boosting (Poisson loss)": _pipe(cfg, HistGradientBoostingRegressor(
            loss="poisson", **s["hist_gb_poisson"], random_state=cfg.seed)),
    }
