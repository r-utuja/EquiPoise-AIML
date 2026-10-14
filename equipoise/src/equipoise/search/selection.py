"""Classical feature selection: filter, embedded and wrapper methods.

Filter methods score each feature on its own, without a model: mutual information (any kind
of dependence with letters gained), absolute Pearson correlation (linear dependence) and the
chi-square statistic (dependence with the binary "gained >= 15 letters", on non-negative
min-max-scaled features). Embedded methods select while fitting: the Lasso path records the
penalty at which each feature first gets a non-zero coefficient (earlier = more important),
and a random forest's impurity importances measure how much each feature's splits reduce
error. The wrapper method, recursive feature elimination (RFE), repeatedly fits Ridge and
drops the weakest feature. Encoded columns are mapped back to their raw feature (best score
per feature), and all methods use training patients only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, chi2, mutual_info_regression
from sklearn.linear_model import Ridge, lasso_path
from sklearn.preprocessing import MinMaxScaler

from equipoise.config import Config
from equipoise.preprocess.columns import feature_groups
from equipoise.preprocess.discretize import gain_item
from equipoise.preprocess.encode import build_preprocessor
from equipoise.search.objective import column_owner


def encoded_train(train: pd.DataFrame, cfg: Config
                  ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Median-imputed, scaled, encoded features (no treatment), target, column names, owners."""
    tr = train.dropna(subset=[cfg.primary_outcome])
    pre = build_preprocessor(cfg, imputer="median")
    Z = pre.fit_transform(tr)
    names = list(pre.get_feature_names_out())
    owners = column_owner(names, list(feature_groups(cfg).all))
    return Z, tr[cfg.primary_outcome].to_numpy(float), names, owners


def _by_feature(scores: np.ndarray, owners: list[str], how: str = "max") -> pd.Series:
    s = pd.Series(scores, index=owners)
    return s.groupby(level=0).max() if how == "max" else s.groupby(level=0).sum()


def lasso_path_entry(Z: np.ndarray, y: np.ndarray, n_alphas: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Lasso path on standardized features: (alphas, coefficients [columns x alphas])."""
    alphas, coefs, _ = lasso_path(Z, y - y.mean(), alphas=n_alphas)
    return alphas, coefs


def rank_features(train: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    """One row per raw feature with each method's score and rank (1 = most important)."""
    s = cfg.search
    Z, y, names, owners = encoded_train(train, cfg)
    tr = train.dropna(subset=[cfg.primary_outcome])
    out = pd.DataFrame(index=sorted(set(owners)))

    out["mutual_information"] = _by_feature(
        mutual_info_regression(Z, y, random_state=cfg.seed), owners)
    corr = np.array([abs(np.corrcoef(Z[:, j], y)[0, 1]) if Z[:, j].std() > 0 else 0.0
                     for j in range(Z.shape[1])])
    out["abs_correlation"] = _by_feature(corr, owners)
    gain = gain_item(tr, cfg).to_numpy(int)
    chi, _ = chi2(MinMaxScaler().fit_transform(Z), gain)
    out["chi_square_gain15"] = _by_feature(np.nan_to_num(chi), owners)

    alphas, coefs = lasso_path_entry(Z, y, int(s["lasso_path_alphas"]))
    first = np.array([alphas[np.flatnonzero(np.abs(c) > 0)[0]] if np.any(np.abs(c) > 0)
                      else 0.0 for c in coefs])
    out["lasso_entry_alpha"] = _by_feature(first, owners)

    rf = RandomForestRegressor(**cfg.supervised["random_forest"], random_state=cfg.seed,
                               n_jobs=1).fit(Z, y)
    out["rf_importance"] = _by_feature(rf.feature_importances_, owners, how="sum")

    rfe = RFE(Ridge(alpha=float(s["ridge_alpha"])),
              n_features_to_select=int(s["rfe_n_features"])).fit(Z, y)
    out["rfe_rank"] = -_by_feature(-rfe.ranking_.astype(float), owners)   # best (min) rank

    ranks = pd.DataFrame(index=out.index)
    for col in out.columns:
        ranks[col] = out[col].rank(ascending=(col == "rfe_rank"), method="min")
    out = out.join(ranks.add_suffix(" (rank)"))
    out["mean_rank"] = ranks.mean(axis=1)
    out = out.sort_values("mean_rank").reset_index(names="feature")
    extras = {"lasso_alphas": alphas, "lasso_coefs": coefs, "column_names": names,
              "owners": owners}
    return out, extras
