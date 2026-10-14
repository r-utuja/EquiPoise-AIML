"""Further inference on the primary outcome: variances, correlations, interaction regression.

Levene's test (robust to non-normality) and Bartlett's test ask whether the spread of letters
gained differs between arms; pairwise F-tests compare two variances directly. The partial
correlation between gain and baseline vision removes the part of both explained by baseline
CST. The multiple correlation R is the correlation between observed gain and its best linear
prediction from several baseline features. Finally, the regression
`gain ~ treatment * baseline vision + covariates` tests whether the drug effect changes with
baseline vision: a significant interaction term is the classical test for heterogeneity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from equipoise.config import Config


def variance_tests(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Levene (median-centred), Bartlett, and pairwise two-sided F-tests on the outcome."""
    y = cfg.primary_outcome
    groups = {a: df.loc[df.treatment == a, y].dropna() for a in cfg.arms}
    lev = stats.levene(*groups.values(), center="median")
    bart = stats.bartlett(*groups.values())
    rows = [
        {"test": "Levene (Brown-Forsythe)", "comparison": "all arms",
         "statistic": lev.statistic, "p": lev.pvalue},
        {"test": "Bartlett", "comparison": "all arms", "statistic": bart.statistic,
         "p": bart.pvalue},
    ]
    for a in cfg.comparison_arms:
        x, z = groups[a], groups[cfg.reference_arm]
        f = x.var(ddof=1) / z.var(ddof=1)
        cdf = stats.f.cdf(f, len(x) - 1, len(z) - 1)
        rows.append({"test": "F-test (variance ratio)", "comparison":
                     f"{a} / {cfg.reference_arm}", "statistic": f,
                     "p": float(2 * min(cdf, 1 - cdf))})
    out = pd.DataFrame(rows)
    sds = {f"sd_{a}": float(g.std(ddof=1)) for a, g in groups.items()}
    return out.assign(**{k: v for k, v in sds.items()})


def partial_correlation(df: pd.DataFrame, x: str, y: str, controls: list[str]
                        ) -> dict[str, float]:
    """Pearson r between x and y after regressing both on `controls` (residual method)."""
    d = df[[x, y, *controls]].dropna().astype(float)
    Z = np.column_stack([np.ones(len(d)), d[controls].to_numpy()])
    rx = d[x] - Z @ np.linalg.lstsq(Z, d[x], rcond=None)[0]
    ry = d[y] - Z @ np.linalg.lstsq(Z, d[y], rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = len(d) - 2 - len(controls)
    t = r * np.sqrt(dof / (1 - r**2))
    raw = stats.pearsonr(d[x], d[y])
    return {"n": len(d), "partial_r": r, "partial_p": float(2 * stats.t.sf(abs(t), dof)),
            "raw_r": float(raw.statistic), "raw_p": float(raw.pvalue)}


def multiple_correlation(df: pd.DataFrame, y: str, features: list[str]) -> dict[str, float]:
    """Multiple correlation R of y on features (complete cases), with R^2, adjusted R^2, F."""
    d = df[[y, *features]].dropna().astype(float)
    fit = smf.ols(f"{y} ~ " + " + ".join(features), data=d).fit()
    return {"n": len(d), "k": len(features), "R": float(np.sqrt(fit.rsquared)),
            "R2": float(fit.rsquared), "R2_adj": float(fit.rsquared_adj),
            "F": float(fit.fvalue), "p": float(fit.f_pvalue)}


def interaction_regression(df: pd.DataFrame, cfg: Config, modifier: str
                           ) -> tuple[pd.DataFrame, dict[str, float]]:
    """OLS `y ~ C(treatment) * modifier + covariates`; returns coefficients and the
    F-test comparing it with the model without the interaction (complete cases)."""
    y = cfg.primary_outcome
    covs = [c for c in cfg.stats["regression_covariates"] if c != modifier]
    trt = f"C(treatment, Treatment('{cfg.reference_arm}'))"
    d = df[[y, "treatment", modifier, *covs]].dropna()
    rhs = " + ".join(covs)
    main = smf.ols(f"{y} ~ {trt} + {modifier} + {rhs}", data=d).fit()
    full = smf.ols(f"{y} ~ {trt} * {modifier} + {rhs}", data=d).fit()
    comp = anova_lm(main, full)
    ci = full.conf_int(alpha=1 - float(cfg.stats["ci_level"]))
    coefs = pd.DataFrame({
        "term": full.params.index.str.replace(f"{trt}[T.", "treatment[", regex=False),
        "coef": full.params.values, "se": full.bse.values, "ci_low": ci[0].values,
        "ci_high": ci[1].values, "p": full.pvalues.values,
    }).assign(model=f"treatment x {modifier}")
    test = {"n": int(full.nobs), "F_interaction": float(comp["F"].iloc[1]),
            "p_interaction": float(comp["Pr(>F)"].iloc[1]),
            "R2_without": float(main.rsquared), "R2_with": float(full.rsquared)}
    return coefs, test
