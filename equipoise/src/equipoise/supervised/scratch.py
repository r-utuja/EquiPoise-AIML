"""Estimators written from scratch, plus the arm-mean baseline.

OLS via the normal equations: the least-squares coefficients solve (X^T X) b = X^T y, solved
here with `numpy.linalg.solve` (never by inverting X^T X explicitly). Logistic regression via
Newton-Raphson, also called iteratively reweighted least squares (IRLS): starting from b = 0,
repeat b <- b + (X^T W X)^-1 X^T (y - p), where p = sigmoid(X b) and W = diag(p (1 - p)),
until the step is tiny. Each step is a weighted least-squares fit, hence the name. Both are
checked against sklearn / statsmodels. The arm-mean baseline predicts the training mean of
the patient's own arm: any useful model has to beat it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def _with_intercept(X: Any) -> np.ndarray:
    X = np.asarray(X, float)
    return np.column_stack([np.ones(len(X)), X])


def normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients [intercept, b1, ...] from (X^T X) b = X^T y."""
    A = _with_intercept(X)
    return np.linalg.solve(A.T @ A, A.T @ np.asarray(y, float))


class NormalEquationsRegressor(RegressorMixin, BaseEstimator):
    """OLS fitted by solving the normal equations with numpy."""

    def fit(self, X: Any, y: Any) -> NormalEquationsRegressor:
        beta = normal_equations(X, y)
        self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "coef_")
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class IRLSLogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary logistic regression by Newton-Raphson / IRLS (unpenalized when l2 = 0)."""

    def __init__(self, max_iter: int = 50, tol: float = 1e-8, l2: float = 0.0) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.l2 = l2

    def fit(self, X: Any, y: Any) -> IRLSLogisticRegression:
        A = _with_intercept(X)
        y = np.asarray(y, float)
        self.classes_ = np.array([0, 1])
        beta = np.zeros(A.shape[1])
        penalty = self.l2 * np.eye(A.shape[1])
        penalty[0, 0] = 0.0  # never shrink the intercept
        self.n_iter_, self.converged_ = 0, False
        for it in range(1, self.max_iter + 1):
            p = _sigmoid(A @ beta)
            w = p * (1 - p)
            grad = A.T @ (y - p) - penalty @ beta
            hess = (A * w[:, None]).T @ A + penalty
            step = np.linalg.solve(hess, grad)
            beta = beta + step
            self.n_iter_ = it
            if np.max(np.abs(step)) < self.tol:
                self.converged_ = True
                break
        self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "coef_")
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def predict_proba(self, X: Any) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ArmMeanRegressor(RegressorMixin, BaseEstimator):
    """Baseline: predict the training mean outcome of the patient's randomized arm.

    Takes the raw DataFrame (it only reads the `treatment` column), so it is used without
    the feature preprocessor.
    """

    def fit(self, X: pd.DataFrame, y: Any) -> ArmMeanRegressor:
        y = pd.Series(np.asarray(y, float), index=X.index)
        self.means_ = y.groupby(X["treatment"]).mean().to_dict()
        self.overall_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "means_")
        return X["treatment"].map(self.means_).fillna(self.overall_).to_numpy(float)


class ArmRateClassifier(ClassifierMixin, BaseEstimator):
    """Baseline classifier: the training event rate of the patient's arm as probability."""

    def fit(self, X: pd.DataFrame, y: Any) -> ArmRateClassifier:
        y = pd.Series(np.asarray(y, float), index=X.index)
        self.classes_ = np.array([0, 1])
        self.rates_ = y.groupby(X["treatment"]).mean().to_dict()
        self.overall_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "rates_")
        p = X["treatment"].map(self.rates_).fillna(self.overall_).to_numpy(float)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
