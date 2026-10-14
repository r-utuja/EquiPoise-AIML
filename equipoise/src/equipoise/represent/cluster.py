"""Cluster patients on scaled baseline features, then look at the treatment effect per cluster.

k-means places k centres to minimise within-cluster squared distance; we scan k and pick the
k with the best silhouette (how much closer a patient is to its own cluster than to the next
one), showing the elbow of the inertia curve as well. Ward's hierarchical clustering merges
the two clusters whose union increases within-cluster variance least, drawn as a dendrogram.
DBSCAN finds dense regions and leaves sparse points as noise. Within each cluster we compare
arms with Welch t-tests. This is exploratory: clusters were not pre-specified, subgroups are
small, and many comparisons are made, so nothing here is evidence of a real effect modifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from equipoise.config import Config
from equipoise.stats.replication import welch_ci


def kmeans_scan(Z: np.ndarray, k_range: tuple[int, int], seed: int) -> pd.DataFrame:
    """Inertia (elbow) and silhouette for each k."""
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Z, km.labels_)})
    return pd.DataFrame(rows)


def ward_labels(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Ward linkage matrix and labels cut into k clusters (0-based)."""
    L = linkage(Z, method="ward")
    return L, fcluster(L, t=k, criterion="maxclust") - 1


def dbscan_labels(Z: np.ndarray, min_samples: int, eps_quantile: float
                  ) -> tuple[np.ndarray, float, np.ndarray]:
    """DBSCAN labels (-1 = noise), eps from the k-distance curve, and that curve."""
    dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)
    k_dist = np.sort(dist[:, -1])
    eps = float(np.quantile(k_dist, eps_quantile))
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z), eps, k_dist


def effects_by_cluster(df: pd.DataFrame, labels: np.ndarray, method: str, cfg: Config
                       ) -> pd.DataFrame:
    """Arm differences (vs reference) in the primary outcome within each cluster."""
    level = float(cfg.stats["ci_level"])
    y = cfg.primary_outcome
    d = df.assign(cluster=labels)
    rows = []
    for c, g in d.groupby("cluster"):
        g = g.dropna(subset=[y])
        for arm in cfg.comparison_arms:
            a = g.loc[g.treatment == arm, y]
            b = g.loc[g.treatment == cfg.reference_arm, y]
            row = {"method": method, "cluster": int(c), "n_cluster": len(g),
                   "comparison": f"{arm} - {cfg.reference_arm}", "n_arm": len(a),
                   "n_reference": len(b), "mean_baseline_va": float(g.X_va_letters.mean())}
            if min(len(a), len(b)) >= 3:
                diff, lo, hi, p = welch_ci(a, b, level)
                row.update({"difference": diff, "ci_low": lo, "ci_high": hi, "p": p})
            else:
                row.update({"difference": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                            "p": np.nan})
            rows.append(row)
    return pd.DataFrame(rows).assign(note="exploratory; not pre-specified")


def cluster_profile(df: pd.DataFrame, labels: np.ndarray, method: str) -> pd.DataFrame:
    """Mean of key baseline features per cluster, to describe what each cluster is."""
    cols = ["X_va_letters", "X_cst_um", "X_hba1c", "X_age", "X_diabetes_duration_yrs",
            "X_prior_antivegf", "X_prior_prp", "X_insulin"]
    prof = df[cols].assign(cluster=labels).groupby("cluster").mean()
    prof.insert(0, "n", pd.Series(labels).value_counts().sort_index().to_numpy())
    return prof.reset_index().assign(method=method)
