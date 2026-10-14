"""Table 1: baseline characteristics by randomized arm.

Randomization should make the three arms alike at baseline. Each feature is summarized per
arm (mean ± SD for roughly symmetric numbers, median [IQR] for skewed ones, n (%) for
categories) and compared with one-way ANOVA, Kruskal-Wallis (skewed) or chi-square. With
about 40 features, a couple of p < 0.05 are expected by chance alone even when
randomization worked, so p-values here are descriptive, not evidence of imbalance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from equipoise.config import Config
from equipoise.data.schema import ANALYSIS_COLUMNS


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table1(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per feature (one per level for categorical features)."""
    arms = list(cfg.arms)
    groups = {a: df[df.treatment == a] for a in arms}
    skew_cut = float(cfg.stats["skew_threshold"])
    cols = {f"{a} (N={len(groups[a])})": a for a in arms}
    rows: list[dict[str, object]] = []

    def base(name: str) -> dict[str, object]:
        return {"Variable": name, f"Overall (N={len(df)})": ""}

    for spec in (c for c in ANALYSIS_COLUMNS if c.role == "feature"):
        col = spec.name
        miss = int(df[col].isna().sum())
        if spec.categories is not None:
            tab = pd.crosstab(df[col], df.treatment).reindex(columns=arms, fill_value=0)
            chi2, p, _, expected = stats.chi2_contingency(tab)
            head = base(col.replace("X_", ""))
            head.update({k: "" for k in cols})
            head.update({"p": _fmt_p(p), "test": "chi-square"
                         + (" (expected < 5 in some cells)" if (expected < 5).any() else ""),
                         "missing": miss})
            rows.append(head)
            for level in tab.index:
                r = {"Variable": f"  {level}",
                     f"Overall (N={len(df)})": f"{int(tab.loc[level].sum())} "
                     f"({100 * tab.loc[level].sum() / df[col].notna().sum():.1f}%)"}
                for label, a in cols.items():
                    n = int(tab.loc[level, a])
                    r[label] = f"{n} ({100 * n / groups[a][col].notna().sum():.1f}%)"
                r.update({"p": "", "test": "", "missing": ""})
                rows.append(r)
            continue

        values = {a: groups[a][col].dropna().astype(float) for a in arms}
        r = base(col.replace("X_", ""))
        if spec.binary:
            tab = pd.crosstab(df[col], df.treatment).reindex(columns=arms, fill_value=0)
            _, p, _, _ = stats.chi2_contingency(tab) if tab.shape[0] == 2 else (0, 1.0, 0, 0)
            fmt = lambda s: f"{int(s.sum())} ({100 * s.mean():.1f}%)"  # noqa: E731
            test = "chi-square"
        elif abs(float(df[col].skew())) > skew_cut:
            _, p = stats.kruskal(*values.values())
            fmt = lambda s: (f"{s.median():.1f} [{s.quantile(0.25):.1f}, "  # noqa: E731
                             f"{s.quantile(0.75):.1f}]")
            test = "Kruskal-Wallis"
        else:
            _, p = stats.f_oneway(*values.values())
            fmt = lambda s: f"{s.mean():.1f} ± {s.std():.1f}"  # noqa: E731
            test = "one-way ANOVA"
        r[f"Overall (N={len(df)})"] = fmt(df[col].dropna().astype(float))
        for label, a in cols.items():
            r[label] = fmt(values[a])
        r.update({"p": _fmt_p(float(p)), "test": test, "missing": miss})
        rows.append(r)
    return pd.DataFrame(rows)


def n_significant(t1: pd.DataFrame, alpha: float) -> tuple[int, int]:
    """(number of features with p < alpha, number of features tested)."""
    ps = pd.to_numeric(t1["p"].replace({"<0.001": "0.0005", "": np.nan}), errors="coerce")
    ps = ps.dropna()
    return int((ps < alpha).sum()), int(len(ps))
