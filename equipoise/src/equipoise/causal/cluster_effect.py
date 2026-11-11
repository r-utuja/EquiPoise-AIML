"""Unsupervised counterpart to the causal forest: cluster patients, then look up arm means.

k-means groups training patients using baseline features only (no outcome, no treatment).
Because treatment was randomized, the difference between two arms' mean week-52 change inside
a cluster is an unbiased estimate of that cluster's treatment effect. A new patient receives the
numbers of the nearest cluster. 95% intervals come from bootstrap resamples drawn within each
cluster and arm (they are only means, so this is cheap). Same interface as the other learners.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from equipoise.causal.learners import CATEModel
from equipoise.config import Config


class ClusterEffectModel(CATEModel):
    """k-means on X, then per cluster x arm outcome means; effect = arm mean - reference mean."""

    name = "Cluster x arm means (k-means)"
    has_intervals = True

    def __init__(self, cfg: Config, k: int | None = None) -> None:
        super().__init__(cfg)
        self.k = int(k if k is not None else cfg.represent["cluster_effect_k"])

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> ClusterEffectModel:
        # clustering sees X only; T and Y are used afterwards, for the cell means
        self.km_ = KMeans(n_clusters=self.k, n_init=10, random_state=self.cfg.seed).fit(X)
        self.labels_ = self.km_.labels_
        self._T, self._Y = np.asarray(T, int), np.asarray(Y, float)
        self.n_ = np.array([[np.sum((self.labels_ == c) & (self._T == a))
                             for a in range(self.n_arms)] for c in range(self.k)])
        self.mean_ = self._cell_means(self._Y, self.labels_)
        self._boot: np.ndarray | None = None
        return self

    def _cell_means(self, Y: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """(k, n_arms) means; an empty cell falls back to that arm's overall mean."""
        out = np.empty((self.k, self.n_arms))
        for a in range(self.n_arms):
            fallback = self._Y[self._T == a].mean()
            for c in range(self.k):
                m = (labels == c) & (self._T == a)
                out[c, a] = Y[m].mean() if m.any() else fallback
        return out

    def _bootstrap_effects(self) -> np.ndarray:
        """(reps, k, n_comparisons) effects, resampling within each cluster and arm."""
        if self._boot is None:
            rng = np.random.default_rng(self.cfg.seed)
            reps = int(self.cfg.bootstrap_reps)
            cells = {(c, a): self._Y[(self.labels_ == c) & (self._T == a)]
                     for c in range(self.k) for a in range(self.n_arms)}
            means = np.empty((reps, self.k, self.n_arms))
            for (c, a), y in cells.items():
                if len(y):
                    means[:, c, a] = rng.choice(y, (reps, len(y))).mean(axis=1)
                else:
                    means[:, c, a] = self.mean_[c, a]
            self._boot = means[:, :, 1:] - means[:, :, [0]]
        return self._boot

    def cluster_of(self, X: np.ndarray) -> np.ndarray:
        return self.km_.predict(X)

    def predict_outcome(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        return self.mean_[self.cluster_of(X), np.asarray(T, int)]

    def predict_effect(self, X: np.ndarray) -> np.ndarray:
        c = self.cluster_of(X)
        return self.mean_[c][:, 1:] - self.mean_[c][:, [0]]

    effect = predict_effect

    def effect_interval(self, X: np.ndarray, alpha: float = 0.05
                        ) -> tuple[np.ndarray, np.ndarray]:
        c, B = self.cluster_of(X), self._bootstrap_effects()
        return (np.quantile(B, alpha / 2, axis=0)[c], np.quantile(B, 1 - alpha / 2, axis=0)[c])

    def p_benefit(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """P(effect > threshold) per patient and comparison, from the bootstrap draws."""
        return (self._bootstrap_effects() > threshold).mean(axis=0)[self.cluster_of(X)]
