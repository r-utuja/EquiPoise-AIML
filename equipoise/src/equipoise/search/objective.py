"""The shared objective every search algorithm minimises.

A candidate is a binary mask over the raw baseline features (1 = use the feature). Its cost
is J(mask) = 5-fold CV RMSE of a Ridge model predicting week-52 letters gained from the
selected features plus the arm dummies, + `feature_penalty` * (number of selected features),
so a feature must earn its place. The median imputer, scaler and encoders in
`build_preprocessor` act column by column, so each fold's fully preprocessed matrix is
computed once and a mask simply selects its columns; Ridge is then solved in closed form.
`Evaluator` counts unique evaluations and stops every algorithm at the same budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from equipoise.config import Config
from equipoise.data.splits import cv_folds
from equipoise.preprocess.columns import feature_groups
from equipoise.preprocess.encode import build_preprocessor


class BudgetExhausted(Exception):
    """Raised by the evaluator when the evaluation budget is used up."""


def ridge_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, alpha: float
                      ) -> np.ndarray:
    """Closed-form Ridge with an unpenalized intercept: centre, solve (X'X + aI) b = X'y."""
    mx, my = Xtr.mean(axis=0), ytr.mean()
    Xc = Xtr - mx
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ (ytr - my))
    return (Xva - mx) @ beta + my


def column_owner(output_names: list[str], raw_features: list[str]) -> list[str | None]:
    """Map each preprocessed column to the raw feature it came from (None = treatment)."""
    owners: list[str | None] = []
    by_length = sorted(raw_features, key=len, reverse=True)
    for name in output_names:
        n = name.split("__", 1)[1]
        if n.startswith("treatment_"):
            owners.append(None)
        elif n.startswith("X_cst_harmonized"):
            owners.append("X_cst_um")
        else:
            owners.append(next(f for f in by_length if n == f or n.startswith(f + "_")))
    return owners


@dataclass
class FoldData:
    """Per-fold preprocessed matrices and the column -> feature bookkeeping."""

    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    features: list[str]
    columns_of: list[np.ndarray]   # per raw feature: its column indices
    always: np.ndarray             # treatment columns, always included


def build_fold_data(train: pd.DataFrame, cfg: Config) -> FoldData:
    """Preprocess each CV fold once (median imputation; fitted on that fold's training part)."""
    tr = train.dropna(subset=[cfg.primary_outcome]).reset_index(drop=True)
    y = tr[cfg.primary_outcome].to_numpy(float)
    features = list(feature_groups(cfg).all)
    folds, owners = [], None
    for i_tr, i_va in cv_folds(tr, cfg.n_folds, cfg.seed):
        pre = build_preprocessor(cfg, imputer="median", include_treatment=True)
        Ztr = pre.fit_transform(tr.iloc[i_tr])
        Zva = pre.transform(tr.iloc[i_va])
        folds.append((Ztr, y[i_tr], Zva, y[i_va]))
        if owners is None:
            owners = column_owner(list(pre.get_feature_names_out()), features)
    own = np.array(owners, dtype=object)
    columns_of = [np.flatnonzero(own == f) for f in features]
    return FoldData(folds, features, columns_of, np.flatnonzero(own == None))  # noqa: E711


@dataclass
class Evaluator:
    """Budgeted, memoised objective J(mask). Records the best-so-far after every evaluation."""

    data: FoldData
    alpha: float
    penalty: float
    budget: int
    cache: dict[bytes, float] = field(default_factory=dict)
    history: list[float] = field(default_factory=list)   # best J after each unique evaluation
    best_value: float = np.inf
    best_mask: np.ndarray | None = None
    call_factor: int = 20          # safety cap on total calls (cache hits included)
    calls: int = 0
    stopped_by: str = "running"

    @property
    def n_features(self) -> int:
        return len(self.data.features)

    @property
    def n_evals(self) -> int:
        return len(self.cache)

    def cv_rmse(self, mask: np.ndarray) -> float:
        cols = np.concatenate([self.data.always,
                               *[self.data.columns_of[i] for i in np.flatnonzero(mask)]])
        cols = np.sort(cols.astype(int))
        sq, n = 0.0, 0
        for Ztr, ytr, Zva, yva in self.data.folds:
            pred = ridge_fit_predict(Ztr[:, cols], ytr, Zva[:, cols], self.alpha)
            sq += float(np.sum((yva - pred) ** 2))
            n += len(yva)
        return float(np.sqrt(sq / n))

    def __call__(self, mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=np.int8)
        key = mask.tobytes()
        self.calls += 1
        if self.calls > self.call_factor * self.budget:
            # an algorithm cycling among already-evaluated masks would otherwise never stop
            self.stopped_by = f"call cap ({self.call_factor} x budget) - search was cycling"
            raise BudgetExhausted
        if key in self.cache:
            return self.cache[key]
        if self.n_evals >= self.budget:
            self.stopped_by = "evaluation budget"
            raise BudgetExhausted
        value = self.cv_rmse(mask) + self.penalty * int(mask.sum())
        self.cache[key] = value
        if value < self.best_value:
            self.best_value, self.best_mask = value, mask.copy()
        self.history.append(self.best_value)
        return value

    def fresh(self) -> Evaluator:
        """Same data and settings, empty cache and history (one per algorithm run)."""
        return Evaluator(self.data, self.alpha, self.penalty, self.budget,
                         call_factor=self.call_factor)


def make_evaluator(train: pd.DataFrame, cfg: Config) -> Evaluator:
    s = cfg.search
    return Evaluator(build_fold_data(train, cfg), float(s["ridge_alpha"]),
                     float(s["feature_penalty"]), int(s["budget"]))
