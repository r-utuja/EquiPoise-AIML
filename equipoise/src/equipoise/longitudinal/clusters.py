"""Cluster vision trajectories into response patterns (fast / slow / non-responders).

Each training patient's interpolated vision change over weeks 0-52 is a vector of 53 numbers;
k-means groups similar curves. Clusters are named from their mean curves by the gain they
reach at week 52: highest = "fast responders", lowest = "non-responders", the rest "slow
responders". Only patients observed through week 52 are clustered (no curve is
extrapolated). The response pattern is an outcome, so it is never a model input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from equipoise.config import Config
from equipoise.longitudinal.tensor import TrajectoryTensor


@dataclass
class TrajectoryClusters:
    summary: pd.DataFrame        # one row per cluster
    by_arm: pd.DataFrame         # % of each arm in each cluster (+ chi-square p)
    weeks: np.ndarray
    centres: dict[str, np.ndarray]


def cluster_trajectories(tensor: TrajectoryTensor, cfg: Config) -> TrajectoryClusters:
    until = int(cfg.longitudinal["trajectory_until_week"])
    keep_w = tensor.weeks <= until
    curves = tensor.change_from_baseline("va_letters")[:, keep_w]
    complete = ~np.isnan(curves).any(axis=1)
    curves = curves[complete]
    k = int(cfg.longitudinal["trajectory_k"])
    km = KMeans(n_clusters=k, n_init=20, random_state=cfg.seed).fit(curves)
    order = np.argsort(km.cluster_centers_[:, -1])            # by week-52 gain
    names = {int(c): "slow responders" for c in order}
    names[int(order[0])] = "non-responders"
    names[int(order[-1])] = "fast responders"
    labels = np.array([names[int(c)] for c in km.labels_])
    wk12 = int(np.argmin(np.abs(tensor.weeks[keep_w] - 12)))
    summary = pd.DataFrame([{
        "cluster": names[c], "n": int((km.labels_ == c).sum()),
        "mean_gain_wk12": float(km.cluster_centers_[c, wk12]),
        "mean_gain_wk52": float(km.cluster_centers_[c, -1])} for c in range(k)])
    summary = summary.sort_values("mean_gain_wk52", ascending=False)
    summary["silhouette_overall"] = float(silhouette_score(curves, km.labels_))
    tab = pd.crosstab(labels, tensor.treatment[complete])
    _, p, _, _ = stats.chi2_contingency(tab)
    by_arm = (tab / tab.sum(axis=0) * 100).round(1).reset_index(names="cluster")
    by_arm["chi_square_p_arm_vs_cluster"] = p
    centres = {names[c]: km.cluster_centers_[c] for c in range(k)}
    return TrajectoryClusters(summary, by_arm, tensor.weeks[keep_w], centres)
