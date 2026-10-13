"""CATE learners for aflibercept - bevacizumab and ranibizumab - bevacizumab.

- S-learner: one outcome model with treatment as an input; effect = prediction with the drug
  switched to arm a minus with it switched to bevacizumab.
- T-learner: a separate outcome model per arm; effect = difference of their predictions.
- X-learner: T-learner first, then impute each patient's individual effect from the other
  arm's model and regress those imputed effects on X, blending by the (known) propensity.
- DR-learner: build a doubly-robust pseudo-outcome per patient from cross-fitted outcome
  models plus inverse-propensity-weighted residuals, then regress it on X.
- Causal forest (econml CausalForestDML): residualize outcome and treatment on X with
  cross-fitting, then grow "honest" trees that split where the effect differs; gives
  confidence intervals for each patient's effect.
- Constant effect: the plain difference in arm means for everyone (no heterogeneity); the
  yardstick every heterogeneous learner must beat.
Each meta-learner runs with a gradient-boosting and a Ridge base learner. Propensities are
the known randomization probabilities (see data.py), never estimated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from equipoise.causal.data import KnownPropensity, known_propensities
from equipoise.config import Config


def base_learner(kind: str, cfg: Config, params: dict[str, Any] | None = None) -> Any:
    """Gradient boosting or Ridge regressor from config (optionally overridden by tuning)."""
    p = {**cfg.causal["base_learners"][kind], **(params or {})}
    if kind == "gb":
        return GradientBoostingRegressor(**p, random_state=cfg.seed)
    return Ridge(**p)


class CATEModel:
    """Common interface: fit(X, T, Y); effect(X) -> (n, n_comparisons) in code order 1..k."""

    name = "base"
    has_intervals = False

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.n_arms = len(cfg.arms)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> CATEModel:
        raise NotImplementedError

    def effect(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ConstantEffect(CATEModel):
    name = "Constant effect (difference in means)"

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> ConstantEffect:
        self.ate_ = np.array([Y[T == k].mean() - Y[T == 0].mean()
                              for k in range(1, self.n_arms)])
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.ate_, (len(X), 1))


class EconMLMeta(CATEModel):
    """S / T / X / DR learner from econml with a chosen base learner."""

    KINDS = ("S", "T", "X", "DR")

    def __init__(self, cfg: Config, kind: str, base: str,
                 base_params: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.kind, self.base, self.base_params = kind, base, base_params
        self.name = f"{kind}-learner ({'GB' if base == 'gb' else 'Ridge'})"

    def _make(self) -> Any:
        cfg = self.cfg
        m = base_learner(self.base, cfg, self.base_params)
        cats = list(range(self.n_arms))
        prop = KnownPropensity(tuple(known_propensities(cfg)))
        if self.kind == "S":
            return SLearner(overall_model=m, categories=cats)
        if self.kind == "T":
            return TLearner(models=m, categories=cats)
        if self.kind == "X":
            return XLearner(models=m, cate_models=clone(m), propensity_model=prop,
                            categories=cats)
        return DRLearner(model_propensity=prop, model_regression=m, model_final=clone(m),
                         cv=int(cfg.causal["cv_folds"]), categories=cats,
                         random_state=cfg.seed)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> EconMLMeta:
        self.est_ = self._make()
        self.est_.fit(Y, T, X=X)
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([self.est_.effect(X, T0=0, T1=k) for k in range(1, self.n_arms)])


class CausalForest(CATEModel):
    name = "Causal forest (CausalForestDML)"
    has_intervals = True

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> CausalForest:
        cfg, cf = self.cfg, self.cfg.causal["causal_forest"]
        self.est_ = CausalForestDML(
            model_y=base_learner("gb", cfg),
            model_t=KnownPropensity(tuple(known_propensities(cfg))),
            discrete_treatment=True, categories=list(range(self.n_arms)),
            n_estimators=int(cf["n_estimators"]), min_samples_leaf=int(cf["min_samples_leaf"]),
            max_depth=cf["max_depth"], cv=int(cfg.causal["cv_folds"]),
            random_state=cfg.seed)
        self.est_.fit(Y, T, X=X)
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([self.est_.effect(X, T0=0, T1=k) for k in range(1, self.n_arms)])

    def effect_interval(self, X: np.ndarray, alpha: float = 0.05
                        ) -> tuple[np.ndarray, np.ndarray]:
        bounds = [self.est_.effect_interval(X, T0=0, T1=k, alpha=alpha)
                  for k in range(1, self.n_arms)]
        return (np.column_stack([b[0] for b in bounds]),
                np.column_stack([b[1] for b in bounds]))

    def feature_importances(self) -> np.ndarray:
        return np.asarray(self.est_.feature_importances_)


def learner_zoo(cfg: Config, include_tarnet: bool = True) -> dict[str, Any]:
    """Factories (cfg -> unfitted model) for every learner compared in M6."""
    zoo: dict[str, Any] = {"Constant effect (difference in means)": ConstantEffect}
    for kind in EconMLMeta.KINDS:
        for base in ("gb", "ridge"):
            m = EconMLMeta(cfg, kind, base)
            zoo[m.name] = (lambda c, k=kind, b=base: EconMLMeta(c, k, b))
    zoo[CausalForest.name] = CausalForest
    if include_tarnet:
        from equipoise.causal.tarnet import TARNetLearner

        zoo[TARNetLearner.name] = TARNetLearner
    return zoo
