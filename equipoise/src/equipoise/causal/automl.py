"""AutoGluon as the base learner of a T-learner (optional `[automl]` extra).

AutoGluon Tabular trains and stacks many model types automatically within a time limit. Here
it is the per-arm outcome model of a T-learner: one AutoGluon predictor per drug, effect =
difference of their predictions. We compare it with the hand-configured base learners on the
same training patients (CV-free leaderboard on AutoGluon's internal validation) and on one
semi-synthetic replication, where the true effects are known.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from equipoise.causal.learners import CATEModel
from equipoise.config import Config


def available() -> bool:
    # find_spec on a dotted name imports the parent and raises if it is missing
    try:
        return importlib.util.find_spec("autogluon.tabular") is not None
    except ModuleNotFoundError:
        return False


class AutoGluonTLearner(CATEModel):
    name = "T-learner (AutoGluon)"

    def __init__(self, cfg: Config, path: Path) -> None:
        super().__init__(cfg)
        self.path = path

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> AutoGluonTLearner:
        from autogluon.tabular import TabularPredictor

        a = self.cfg.causal["automl"]
        if self.path.exists():
            shutil.rmtree(self.path)
        self.predictors_ = []
        self.leaderboards_ = []
        for arm in range(self.n_arms):
            d = pd.DataFrame(X[T == arm], columns=[f"x{i}" for i in range(X.shape[1])])
            d["y"] = Y[T == arm]
            p = TabularPredictor(label="y", problem_type="regression", eval_metric="rmse",
                                 path=str(self.path / f"arm_{arm}"), verbosity=0)
            p.fit(d, time_limit=int(a["time_limit_s"]), presets=str(a["presets"]))
            self.predictors_.append(p)
            self.leaderboards_.append(p.leaderboard(silent=True).assign(arm_code=arm))
        return self

    def _predict(self, arm: int, X: np.ndarray) -> np.ndarray:
        d = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        return self.predictors_[arm].predict(d).to_numpy(float)

    def effect(self, X: np.ndarray) -> np.ndarray:
        base = self._predict(0, X)
        return np.column_stack([self._predict(k, X) - base for k in range(1, self.n_arms)])

    def leaderboard(self) -> pd.DataFrame:
        return pd.concat(self.leaderboards_, ignore_index=True)
