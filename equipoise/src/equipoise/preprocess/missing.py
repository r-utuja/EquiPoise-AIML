"""Missing-data strategies: complete-case vs kNN vs MICE (pooled with Rubin's rules).

44 of 660 patients have no week-52 vision. Complete-case analysis simply drops them, which
is unbiased only if they are missing completely at random. kNN imputation fills each missing
value from the most similar patients (one imputed dataset, so its CI is too narrow). MICE
(multiple imputation by chained equations) imputes each variable from the others with random
draws, m times; the arm differences from the m datasets are pooled with Rubin's rules so the
CI includes the extra uncertainty from not knowing the missing values. Imputers are also scored
on baseline features by hiding observed values and measuring how well each recovers them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

from equipoise.config import Config
from equipoise.data.schema import DR_SEVERITY_CLINICAL
from equipoise.preprocess.columns import NOMINAL, feature_groups


@dataclass(frozen=True)
class Estimate:
    """Arm difference vs the reference arm (letters) with its CI."""

    strategy: str
    comparison: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    df: float
    note: str = ""


def imputation_design(df: pd.DataFrame, cfg: Config, with_outcome: bool = True
                      ) -> pd.DataFrame:
    """All-numeric matrix (NaN kept) for imputation: features, arm dummies, outcome.

    Nominal features become one-hot columns that are all NaN when the category is missing;
    DR severity becomes its ordinal code; skewed labs are logged.
    """
    g = feature_groups(cfg)
    out = pd.DataFrame(index=df.index)
    for col in (*g.continuous, *g.binary, g.cst):
        out[col] = df[col].astype(float)
    for col in g.log:
        out[col] = np.log(df[col].astype(float))
    codes = {k: i for i, k in enumerate(DR_SEVERITY_CLINICAL)}
    out["X_dr_severity_clinical"] = df["X_dr_severity_clinical"].map(codes).astype(float)
    for col in NOMINAL:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
        dummies.loc[df[col].isna(), :] = np.nan
        out = out.join(dummies)
    if with_outcome:
        for arm in cfg.comparison_arms:
            out[f"arm_{arm}"] = (df["treatment"] == arm).astype(float)
        out[cfg.primary_outcome] = df[cfg.primary_outcome].astype(float)
    return out


def _ols_arm_differences(y: np.ndarray, treatment: pd.Series, cfg: Config
                         ) -> dict[str, tuple[float, float, float]]:
    """OLS of outcome on arm dummies: {arm: (difference vs reference, SE, residual df)}."""
    X = np.column_stack([np.ones(len(y))] + [(treatment == a).to_numpy(float)
                                             for a in cfg.comparison_arms])
    fit = sm.OLS(y, X).fit()
    return {a: (float(fit.params[i + 1]), float(fit.bse[i + 1]), float(fit.df_resid))
            for i, a in enumerate(cfg.comparison_arms)}


def _ci(est: float, se: float, dof: float, level: float) -> tuple[float, float]:
    t = stats.t.ppf(0.5 + level / 2, dof)
    return est - t * se, est + t * se


def rubin_pool(estimates: np.ndarray, variances: np.ndarray, df_complete: float
               ) -> tuple[float, float, float, float]:
    """Rubin's rules: pooled estimate, total SE, degrees of freedom, fraction of missing info.

    T = W + (1 + 1/m) B, with W the mean within-imputation variance and B the
    between-imputation variance. Degrees of freedom use the Barnard-Rubin small-sample
    adjustment, capped by the complete-data df.
    """
    m = len(estimates)
    q = float(np.mean(estimates))
    w = float(np.mean(variances))
    b = float(np.var(estimates, ddof=1))
    t = w + (1 + 1 / m) * b
    lam = (1 + 1 / m) * b / t if t > 0 else 0.0
    df_old = (m - 1) / lam**2 if lam > 0 else np.inf
    df_obs = (df_complete + 1) / (df_complete + 3) * df_complete * (1 - lam)
    dof = 1 / (1 / df_old + 1 / df_obs) if np.isfinite(df_old) else df_obs
    return q, float(np.sqrt(t)), float(dof), float(lam)


def compare_strategies(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Arm differences in the primary outcome under complete-case, kNN and MICE."""
    level = float(cfg.stats["ci_level"])
    y_col, p = cfg.primary_outcome, cfg.preprocess
    rows: list[Estimate] = []

    obs = df[y_col].notna()
    for arm, (est, se, dof) in _ols_arm_differences(
            df.loc[obs, y_col].to_numpy(float), df.loc[obs, "treatment"], cfg).items():
        rows.append(Estimate("complete-case", arm, est, se, *_ci(est, se, dof, level),
                             int(obs.sum()), dof))

    design = imputation_design(df, cfg)
    scaler = StandardScaler().fit(design)
    j = list(design.columns).index(y_col)
    knn = KNNImputer(n_neighbors=int(p["knn_neighbors"]))
    filled = knn.fit_transform(scaler.transform(design))
    y_knn = filled[:, j] * scaler.scale_[j] + scaler.mean_[j]
    for arm, (est, se, dof) in _ols_arm_differences(y_knn, df["treatment"], cfg).items():
        rows.append(Estimate("kNN (single imputation)", arm, est, se,
                             *_ci(est, se, dof, level), len(df), dof,
                             "SE ignores imputation uncertainty"))

    m = int(p["mice_imputations"])
    per_arm: dict[str, list[tuple[float, float, float]]] = {a: [] for a in cfg.comparison_arms}
    for i in range(m):
        imp = IterativeImputer(max_iter=int(p["mice_max_iter"]), sample_posterior=True,
                               random_state=cfg.seed + i)
        y_i = imp.fit_transform(design)[:, j]
        for arm, res in _ols_arm_differences(y_i, df["treatment"], cfg).items():
            per_arm[arm].append(res)
    for arm, res in per_arm.items():
        est_, se_, dof_ = map(np.array, zip(*res, strict=True))
        q, se, dof, lam = rubin_pool(est_, se_**2, float(dof_[0]))
        rows.append(Estimate(f"MICE (m={m}, Rubin)", arm, q, se, *_ci(q, se, dof, level),
                             len(df), dof, f"fraction of missing information {lam:.2f}"))

    out = pd.DataFrame([r.__dict__ for r in rows])
    out["comparison"] = out["comparison"] + " - " + cfg.reference_arm
    return out


def score_feature_imputers(train: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Hide a share of observed numeric feature values and score each imputer.

    Returns one row per (imputer, feature): normalized RMSE = RMSE / SD of the feature
    (0 = perfect, about 1 = no better than guessing the mean). Training patients only.
    """
    design = imputation_design(train, cfg, with_outcome=False)
    g = feature_groups(cfg)
    targets = [c for c in (*g.continuous, *g.log, g.cst) if c in design.columns]
    rng = np.random.default_rng(cfg.seed)
    masked = design.copy()
    hidden: dict[str, np.ndarray] = {}
    frac = float(cfg.preprocess["imputation_mask_fraction"])
    for col in targets:
        idx = np.flatnonzero(design[col].notna().to_numpy())
        pick = rng.choice(idx, max(1, int(frac * len(idx))), replace=False)
        hidden[col] = pick
        masked.iloc[pick, masked.columns.get_loc(col)] = np.nan

    scaler = StandardScaler().fit(masked)
    imputers = {
        "median": SimpleImputer(strategy="median"),
        "kNN": KNNImputer(n_neighbors=int(cfg.preprocess["knn_neighbors"])),
        "MICE (single)": IterativeImputer(max_iter=int(cfg.preprocess["mice_max_iter"]),
                                          random_state=cfg.seed),
    }
    rows = []
    for name, imp in imputers.items():
        filled = imp.fit_transform(scaler.transform(masked)) * scaler.scale_ + scaler.mean_
        filled = pd.DataFrame(filled, columns=design.columns, index=design.index)
        for col, pick in hidden.items():
            truth = design[col].to_numpy()[pick]
            pred = filled[col].to_numpy()[pick]
            sd = float(np.nanstd(design[col]))
            rows.append({"imputer": name, "feature": col, "n_hidden": len(pick),
                         "nrmse": float(np.sqrt(np.mean((truth - pred) ** 2)) / sd)})
    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Missing count and % per column with any missing value, overall and by arm."""
    cols = [c for c in df.columns if df[c].isna().any()]
    rows = []
    for c in cols:
        row = {"column": c, "n_missing": int(df[c].isna().sum()),
               "pct_missing": 100 * float(df[c].isna().mean())}
        for arm in cfg.arms:
            row[f"n_missing_{arm}"] = int(df.loc[df.treatment == arm, c].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_missing", ascending=False)
