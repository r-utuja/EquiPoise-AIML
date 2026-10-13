"""Replicate the Protocol T results at 1 and 2 years from the analysis table.

For each horizon we use the patients who completed it (as README section 9.1 does). Mean
letters gained per arm get t-based 95% CIs; the three arms are compared with one-way ANOVA,
then pairwise Welch t-tests (no equal-variance assumption) with Holm's correction for the
three comparisons. Injection counts are skewed, so they are summarized by medians and
compared with Mann-Whitney U; laser rates are proportions compared with chi-square.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests

from equipoise.config import Config

HORIZONS = {
    1: {"has": "M_has_wk52", "va": "Y_va_change_wk52", "cst": "Y_cst_change_wk52",
        "inj": "Y_n_inj_yr1", "laser": "Y_laser_yr1"},
    2: {"has": "M_has_wk104", "va": "Y_va_change_wk104", "cst": "Y_cst_change_wk104",
        "inj": "Y_n_inj_total", "laser": "Y_laser_any"},
}
SUBGROUPS = {"all": None, "va_below_69": 1, "va_69_or_better": 0}


def load_targets(cfg: Config) -> dict[str, Any]:
    path = cfg.paths.root / "config" / "replication_targets.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def mean_ci(x: pd.Series, level: float) -> tuple[float, float, float, float]:
    """Mean, SD and t-based CI of a sample."""
    x = x.dropna().astype(float)
    m, sd, n = float(x.mean()), float(x.std(ddof=1)), len(x)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n)
    return m, sd, m - h, m + h


def welch_ci(a: pd.Series, b: pd.Series, level: float) -> tuple[float, float, float, float]:
    """Difference in means a - b with Welch-Satterthwaite CI; returns (diff, lo, hi, p)."""
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    diff = float(a.mean() - b.mean())
    h = stats.t.ppf(0.5 + level / 2, dof) * np.sqrt(va + vb)
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    return diff, diff - h, diff + h, p


def _pairs(cfg: Config) -> list[tuple[str, str]]:
    """Arm pairs, comparison arms first against the reference, then between them."""
    ref = cfg.reference_arm
    out = [(a, ref) for a in cfg.comparison_arms]
    out += [p for p in combinations(cfg.comparison_arms, 2)]
    return out


def means_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per horizon x subgroup x outcome x arm: n, mean, SD, CI (letters / um)."""
    level = float(cfg.stats["ci_level"])
    rows = []
    for h, c in HORIZONS.items():
        done = df[df[c["has"]] == 1]
        for sg, val in SUBGROUPS.items():
            sub = done if val is None else done[done.X_va_below_69 == val]
            outcomes = {"letters": c["va"]}
            if sg == "all":
                outcomes["cst_um"] = c["cst"]
            for name, col in outcomes.items():
                for arm in cfg.arms:
                    x = sub.loc[sub.treatment == arm, col]
                    m, sd, lo, hi = mean_ci(x, level)
                    rows.append({"horizon_years": h, "subgroup": sg, "outcome": name,
                                 "arm": arm, "n": int(x.notna().sum()), "mean": m, "sd": sd,
                                 "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def burden_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Injections (median, IQR) and laser (% of patients) per horizon and arm."""
    rows = []
    for h, c in HORIZONS.items():
        done = df[df[c["has"]] == 1]
        for arm in cfg.arms:
            g = done[done.treatment == arm]
            rows.append({"horizon_years": h, "arm": arm, "n": len(g),
                         "median_injections": float(g[c["inj"]].median()),
                         "q1_injections": float(g[c["inj"]].quantile(0.25)),
                         "q3_injections": float(g[c["inj"]].quantile(0.75)),
                         "mean_injections": float(g[c["inj"]].mean()),
                         "laser_n": int(g[c["laser"]].sum()),
                         "laser_pct": 100 * float(g[c["laser"]].mean())})
    return pd.DataFrame(rows)


def hypothesis_tests(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """ANOVA + Holm-adjusted Welch t-tests (letters, CST), Mann-Whitney (injections),
    chi-square (laser), per horizon; letters also within the va_below_69 subgroups."""
    level = float(cfg.stats["ci_level"])
    rows: list[dict[str, Any]] = []

    def welch_family(sub: pd.DataFrame, col: str, h: int, sg: str, outcome: str) -> None:
        groups = [sub.loc[sub.treatment == a, col].dropna() for a in cfg.arms]
        f, p = stats.f_oneway(*groups)
        rows.append({"horizon_years": h, "subgroup": sg, "outcome": outcome,
                     "test": "one-way ANOVA", "comparison": "all arms", "statistic": f,
                     "p": p, "p_holm": np.nan, "difference": np.nan, "ci_low": np.nan,
                     "ci_high": np.nan})
        fam = []
        for a, b in _pairs(cfg):
            d, lo, hi, p = welch_ci(sub.loc[sub.treatment == a, col],
                                    sub.loc[sub.treatment == b, col], level)
            fam.append({"horizon_years": h, "subgroup": sg, "outcome": outcome,
                        "test": "Welch t-test", "comparison": f"{a} - {b}", "statistic": np.nan,
                        "p": p, "difference": d, "ci_low": lo, "ci_high": hi})
        adj = multipletests([r["p"] for r in fam], method="holm")[1]
        for r, pa in zip(fam, adj, strict=True):
            rows.append({**r, "p_holm": pa})

    for h, c in HORIZONS.items():
        done = df[df[c["has"]] == 1]
        for sg, val in SUBGROUPS.items():
            sub = done if val is None else done[done.X_va_below_69 == val]
            welch_family(sub, c["va"], h, sg, "letters")
        welch_family(done, c["cst"], h, "all", "cst_um")

        fam = []
        for a, b in _pairs(cfg):
            x, y = (done.loc[done.treatment == k, c["inj"]] for k in (a, b))
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            fam.append({"horizon_years": h, "subgroup": "all", "outcome": "injections",
                        "test": "Mann-Whitney U", "comparison": f"{a} - {b}", "statistic": u,
                        "p": p, "difference": float(x.median() - y.median()),
                        "ci_low": np.nan, "ci_high": np.nan})
        adj = multipletests([r["p"] for r in fam], method="holm")[1]
        rows += [{**r, "p_holm": pa} for r, pa in zip(fam, adj, strict=True)]

        tab = pd.crosstab(done.treatment, done[c["laser"]])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        rows.append({"horizon_years": h, "subgroup": "all", "outcome": "laser",
                     "test": "chi-square", "comparison": "all arms", "statistic": chi2,
                     "p": p, "p_holm": np.nan, "difference": np.nan, "ci_low": np.nan,
                     "ci_high": np.nan})
    return pd.DataFrame(rows)


def versus_targets(means: pd.DataFrame, burden: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Side by side: published, README 'our table', and this run (rounded as published)."""
    t = load_targets(cfg)
    arms = list(cfg.arms)

    def run_values(h: int, key: str) -> list[float]:
        if key.startswith("mean_letters"):
            sg = {"mean_letters_gained": "all", "mean_letters_gained_va_below_69": "va_below_69",
                  "mean_letters_gained_va_69_or_better": "va_69_or_better"}[key]
            m = means[(means.horizon_years == h) & (means.subgroup == sg)
                      & (means.outcome == "letters")].set_index("arm")["mean"]
            return [round(float(m[a]), 1) for a in arms]
        if key == "mean_cst_change_um":
            m = means[(means.horizon_years == h) & (means.outcome == "cst_um")].set_index("arm")
            return [round(float(m.loc[a, "mean"])) for a in arms]
        b = burden[burden.horizon_years == h].set_index("arm")
        col = "median_injections" if key == "median_injections" else "laser_pct"
        return [round(float(b.loc[a, col])) for a in arms]

    rows = []
    for year_key, h in (("year1", 1), ("year2", 2)):
        for key, vals in t[year_key].items():
            fmt = lambda v: " / ".join(str(x) for x in v) if v else ""  # noqa: E731
            rows.append({"horizon_years": h, "result": key,
                         "published (Afl / Bev / Ran)": fmt(vals.get("published")),
                         "README our table": fmt(vals.get("our_table")),
                         "this run": fmt(run_values(h, key))})
    return pd.DataFrame(rows)
