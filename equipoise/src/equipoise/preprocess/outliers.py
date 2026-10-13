"""Flag unusual patients three ways; never drop them by default.

The IQR rule marks a value outside [Q1 - k*IQR, Q3 + k*IQR] of the training data; a patient
is flagged when several features are out of range. Isolation Forest isolates points with
random splits: outliers need few splits, so they get high anomaly scores. DBSCAN groups points
that have enough neighbours within distance eps (picked from the k-distance curve); points in
no dense region are "noise". All three are fitted on training patients and applied to all.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from equipoise.config import Config
from equipoise.preprocess.columns import feature_groups
from equipoise.preprocess.encode import build_preprocessor

METHODS = ("iqr", "isolation_forest", "dbscan")


@dataclass
class OutlierResult:
    """Per-patient flags plus the DBSCAN k-distance curve used to choose eps."""

    flags: pd.DataFrame
    k_distances: np.ndarray
    eps: float


def iqr_flags(train: pd.DataFrame, df: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Series]:
    """Number of continuous features outside the training IQR fences, and the flag."""
    g = feature_groups(cfg)
    cols = [*g.continuous, *g.log, g.cst]
    k = float(cfg.preprocess["outliers"]["iqr_k"])
    q1, q3 = train[cols].quantile(0.25), train[cols].quantile(0.75)
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    n_out = ((df[cols] < lo) | (df[cols] > hi)).sum(axis=1)
    return n_out, n_out >= int(cfg.preprocess["outliers"]["iqr_min_features"])


def detect_outliers(train: pd.DataFrame, df: pd.DataFrame, cfg: Config) -> OutlierResult:
    """Flag patients in `df` with IQR, Isolation Forest and DBSCAN fitted on `train`."""
    o = cfg.preprocess["outliers"]
    pre = build_preprocessor(cfg, imputer="median").fit(train)
    Z_train, Z = pre.transform(train), pre.transform(df)

    n_out, iqr = iqr_flags(train, df, cfg)
    iso = IsolationForest(contamination=float(o["isolation_contamination"]),
                          random_state=cfg.seed).fit(Z_train)

    k = int(o["dbscan_min_samples"])
    dist, _ = NearestNeighbors(n_neighbors=k).fit(Z_train).kneighbors(Z_train)
    k_dist = np.sort(dist[:, -1])
    eps = float(np.quantile(k_dist, float(o["dbscan_eps_quantile"])))
    db = DBSCAN(eps=eps, min_samples=k).fit(Z_train)
    core = Z_train[db.core_sample_indices_]
    # a patient is DBSCAN noise if no core point of the training clusters is within eps
    nearest_core, _ = NearestNeighbors(n_neighbors=1).fit(core).kneighbors(Z)

    flags = pd.DataFrame({
        "pt_id": df["pt_id"].to_numpy(),
        "n_features_outside_iqr": n_out.to_numpy(),
        "iqr": iqr.to_numpy(),
        "isolation_forest": iso.predict(Z) == -1,
        "isolation_score": -iso.score_samples(Z),
        "dbscan": nearest_core[:, 0] > eps,
    })
    return OutlierResult(flags, k_dist, eps)


def overlap_table(flags: pd.DataFrame) -> pd.DataFrame:
    """Counts flagged by each method, each pair, all three, and any."""
    rows = [{"methods": m, "n_flagged": int(flags[m].sum())} for m in METHODS]
    for a, b in combinations(METHODS, 2):
        rows.append({"methods": f"{a} & {b}", "n_flagged": int((flags[a] & flags[b]).sum())})
    rows.append({"methods": "all three", "n_flagged": int(flags[list(METHODS)].all(axis=1).sum())})
    rows.append({"methods": "any", "n_flagged": int(flags[list(METHODS)].any(axis=1).sum())})
    out = pd.DataFrame(rows)
    out["pct_of_patients"] = 100 * out["n_flagged"] / len(flags)
    return out
