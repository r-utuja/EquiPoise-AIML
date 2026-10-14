"""Causal data set-up: features, treatment codes, outcome and the known propensities.

Treatment was randomized, so the probability of each arm is known: the trial's arm
proportions (224 / 218 / 218 of 660, about 0.34 / 0.33 / 0.33), the same for every patient.
We never fit a propensity model; `KnownPropensity` is a stand-in classifier that simply
returns these constants, so econml learners that ask for a propensity model use the truth.
Treatment is coded 0 = bevacizumab (reference, cheapest), then the comparison arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from equipoise.config import Config
from equipoise.preprocess.encode import build_preprocessor


def arm_codes(cfg: Config) -> dict[str, int]:
    """{arm: code} with the reference arm coded 0."""
    order = [cfg.reference_arm, *cfg.comparison_arms]
    return {a: i for i, a in enumerate(order)}


def known_propensities(cfg: Config) -> np.ndarray:
    """Randomization probabilities per treatment code (empirical trial arm proportions)."""
    codes = arm_codes(cfg)
    p = np.zeros(len(codes))
    for arm, code in codes.items():
        p[code] = cfg.arm_sizes[arm] / cfg.n_patients
    return p


class KnownPropensity(ClassifierMixin, BaseEstimator):
    """'Classifier' returning the known randomization probabilities for every patient."""

    def __init__(self, probabilities: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)) -> None:
        self.probabilities = probabilities

    def fit(self, X: Any, y: Any, **_: Any) -> KnownPropensity:
        # Fitted on a subset of arms (e.g. the X-learner's "control vs arm k" problems), the
        # known probabilities are renormalized over those arms: P(k | 0 or k) = p_k/(p_0+p_k).
        self.classes_ = np.unique(np.asarray(y).astype(int))
        p = np.asarray(self.probabilities, float)[self.classes_]
        self.class_probabilities_ = p / p.sum()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.tile(self.class_probabilities_, (len(X), 1))

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), int(self.classes_[np.argmax(self.class_probabilities_)]))


@dataclass
class CausalData:
    """Arrays for one patient set (training or held-out)."""

    frame: pd.DataFrame
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.Y)


def prepare(train: pd.DataFrame, test: pd.DataFrame, cfg: Config,
            features: list[str] | None = None) -> tuple[CausalData, CausalData, Any]:
    """Complete-case train/test arrays with the preprocessor fitted on training patients.

    `features` optionally restricts the raw X_ columns (used to test M4's selected subset).
    """
    y = cfg.primary_outcome
    tr = train.dropna(subset=[y]).reset_index(drop=True)
    te = test.dropna(subset=[y]).reset_index(drop=True)
    pre = build_preprocessor(cfg, features=features).fit(tr)
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    codes = arm_codes(cfg)

    def pack(d: pd.DataFrame) -> CausalData:
        return CausalData(d, pre.transform(d), d["treatment"].map(codes).to_numpy(int),
                          d[y].to_numpy(float), names)

    return pack(tr), pack(te), pre
