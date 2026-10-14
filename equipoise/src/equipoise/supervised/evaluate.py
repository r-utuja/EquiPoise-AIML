"""Cross-validated and held-out evaluation for regression, classification and count models.

Models are compared by 5-fold cross-validation on the training split (folds stratified by
arm); the held-out test set is scored once, with models refitted on the whole training split.
Classification metrics: accuracy, precision (positive predictive value), recall
(sensitivity), specificity, F1 (harmonic mean of precision and recall), ROC-AUC (ranking
quality across all thresholds) and PR-AUC (average precision, informative when positives are
rare).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsRegressor

from equipoise.config import Config
from equipoise.data.splits import cv_folds
from equipoise.preprocess.encode import build_preprocessor
from equipoise.supervised.scratch import IRLSLogisticRegression, normal_equations


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y, pred)
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)), "MAE": mean_absolute_error(y, pred),
            "R2": r2_score(y, pred)}


def count_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 1e-6, None)
    return {"Poisson deviance": mean_poisson_deviance(y, pred),
            "MAE": mean_absolute_error(y, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred)))}


def classification_metrics(y: np.ndarray, proba: np.ndarray, threshold: float
                           ) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "F1": f1_score(y, pred, zero_division=0),
        "AUC": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "PR-AUC": average_precision_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
    }


def _predict(model: Any, X: pd.DataFrame, kind: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if kind == "classification" else model.predict(X)


def _score(kind: str, y: np.ndarray, out: np.ndarray, cfg: Config) -> dict[str, float]:
    if kind == "regression":
        return regression_metrics(y, out)
    if kind == "count":
        return count_metrics(y, out)
    return classification_metrics(y, out, float(cfg.supervised["classification_threshold"]))


def evaluate(models: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame, y_col: str,
             kind: str, cfg: Config) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """CV on train + one held-out score per model.

    Returns (table, held-out predictions per model, models refitted on all of train).
    Rows with a missing target are dropped (complete case).
    """
    tr = train.dropna(subset=[y_col]).reset_index(drop=True)
    te = test.dropna(subset=[y_col]).reset_index(drop=True)
    y_tr, y_te = tr[y_col].to_numpy(float), te[y_col].to_numpy(float)
    if kind == "classification":
        y_tr, y_te = y_tr.astype(int), y_te.astype(int)
    folds = list(cv_folds(tr, cfg.n_folds, cfg.seed))
    rows, preds, fitted = [], {}, {}
    for name, model in models.items():
        fold_scores = []
        for i_tr, i_va in folds:
            m = clone(model).fit(tr.iloc[i_tr], y_tr[i_tr])
            fold_scores.append(_score(kind, y_tr[i_va], _predict(m, tr.iloc[i_va], kind), cfg))
        cv = pd.DataFrame(fold_scores)
        final = clone(model).fit(tr, y_tr)
        out = _predict(final, te, kind)
        held = _score(kind, y_te, out, cfg)
        row = {"Model": name, "n_train": len(tr), "n_test": len(te)}
        for k in cv.columns:
            row[f"CV {k}"] = cv[k].mean()
            row[f"CV {k} sd"] = cv[k].std(ddof=1)
        row.update({f"Test {k}": v for k, v in held.items()})
        rows.append(row)
        preds[name], fitted[name] = out, final
    return pd.DataFrame(rows), {"y_test": y_te, **preds}, fitted


def normal_equations_check(train: pd.DataFrame, y_col: str, cfg: Config) -> dict[str, float]:
    """Our numpy normal-equation coefficients vs sklearn LinearRegression on the same design."""
    tr = train.dropna(subset=[y_col])
    X = build_preprocessor(cfg, include_treatment=True, full_rank=True).fit_transform(tr)
    y = tr[y_col].to_numpy(float)
    ours = normal_equations(X, y)
    sk = LinearRegression().fit(X, y)
    ref = np.r_[sk.intercept_, sk.coef_]
    return {"n": len(y), "n_coefficients": len(ours),
            "max_abs_diff_vs_sklearn": float(np.abs(ours - ref).max()),
            "condition_number_XtX": float(np.linalg.cond(np.c_[np.ones(len(X)), X].T
                                                         @ np.c_[np.ones(len(X)), X]))}


def irls_check(train: pd.DataFrame, y: pd.Series, cfg: Config) -> dict[str, float]:
    """Our IRLS logistic coefficients vs statsmodels Logit (maximum likelihood) on one design."""
    keep = y.notna().to_numpy()
    tr, yy = train[keep], y[keep].to_numpy(int)
    X = build_preprocessor(cfg, include_treatment=True, full_rank=True).fit_transform(tr)
    irls = cfg.supervised["irls"]
    ours = IRLSLogisticRegression(int(irls["max_iter"]), float(irls["tol"])).fit(X, yy)
    ref = sm.Logit(yy, sm.add_constant(X)).fit(disp=0, maxiter=200, method="newton")
    beta = np.r_[ours.intercept_, ours.coef_]
    return {"n": len(yy), "n_coefficients": len(beta), "iterations": ours.n_iter_,
            "converged": float(ours.converged_),
            "max_abs_diff_vs_statsmodels": float(np.abs(beta - ref.params).max())}


def knn_algorithm_timing(train: pd.DataFrame, test: pd.DataFrame, y_col: str, cfg: Config
                         ) -> pd.DataFrame:
    """Fit and query time (median of repeats) for kd_tree, ball_tree and brute force.

    All three must return identical predictions; only speed differs. Timings are wall-clock
    and vary slightly between runs and machines.
    """
    tr = train.dropna(subset=[y_col])
    pre = build_preprocessor(cfg, include_treatment=True).fit(tr)
    Xtr, Xte, y = pre.transform(tr), pre.transform(test), tr[y_col].to_numpy(float)
    reps, k = int(cfg.supervised["knn_timing_repeats"]), int(cfg.supervised["knn_neighbors"])
    rows, preds = [], {}
    for algo in ("kd_tree", "ball_tree", "brute"):
        fit_t, query_t = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            m = KNeighborsRegressor(n_neighbors=k, algorithm=algo).fit(Xtr, y)
            t1 = time.perf_counter()
            preds[algo] = m.predict(Xte)
            t2 = time.perf_counter()
            fit_t.append(t1 - t0)
            query_t.append(t2 - t1)
        rows.append({"algorithm": algo, "fit_ms": 1e3 * float(np.median(fit_t)),
                     "query_ms": 1e3 * float(np.median(query_t)),
                     "n_train": Xtr.shape[0], "n_query": Xte.shape[0],
                     "dimensions": Xtr.shape[1]})
    out = pd.DataFrame(rows)
    out["same_predictions_as_brute"] = [bool(np.allclose(preds[a], preds["brute"]))
                                        for a in out.algorithm]
    return out
