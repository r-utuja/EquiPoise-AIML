"""Cross-check the visit table against the one-row-per-patient analysis table.

Outcomes re-derived from `protocolT_visits.csv` (vision change at weeks 52/104, CST change,
injection counts and laser) are compared patient by patient with the `Y_` columns built by
M0. Vision change must agree within a tolerance; the other quantities are reported, because
M0 counts treatments by treatment date while the visit table records them by visit date, so
a treatment given a few days after a visit near week 52 can legitimately fall either side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEAR1_FALLBACK_DAY = 364  # same fallback M0 uses when a patient has no 52-week visit


def outcomes_from_visits(visits: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
    """Re-derive Y_ outcomes per patient from the visit table."""
    out = pd.DataFrame({"pt_id": analysis["pt_id"]})
    for wk in (52, 104):
        at = visits[visits["visit_label"] == f"{wk} week"].drop_duplicates("pt_id")
        at = at.set_index("pt_id")
        base = visits[visits["visit_label"] == "Baseline"].set_index("pt_id")
        out[f"Y_va_change_wk{wk}"] = (out["pt_id"].map(at["va_letters"])
                                      - out["pt_id"].map(base["va_letters"]))
        out[f"Y_cst_change_wk{wk}"] = (out["pt_id"].map(at["cst_um"])
                                       - out["pt_id"].map(base["cst_um"]))
    day52 = visits[visits["visit_label"] == "52 week"].drop_duplicates("pt_id")
    cut = visits["pt_id"].map(day52.set_index("pt_id")["days_from_rand"]).fillna(
        YEAR1_FALLBACK_DAY)
    yr1 = visits["days_from_rand"] < cut
    inj, las = visits["injected"] == 1, visits["laser"] == 1
    count = lambda mask: out["pt_id"].map(  # noqa: E731
        visits[mask].groupby("pt_id").size()).fillna(0).astype(int)
    out["Y_n_inj_yr1"] = count(inj & yr1)
    out["Y_n_inj_total"] = count(inj)
    out["Y_laser_yr1"] = (count(las & yr1) > 0).astype(int)
    out["Y_laser_any"] = (count(las) > 0).astype(int)
    return out


def cross_check(visits: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
    """One row per outcome: patients compared, mismatches, max absolute difference."""
    derived = outcomes_from_visits(visits, analysis).set_index("pt_id")
    ref = analysis.set_index("pt_id")
    rows = []
    for col in derived.columns:
        a, b = derived[col], ref.loc[derived.index, col]
        both = a.notna() & b.notna()
        diff = (a[both] - b[both]).abs()
        rows.append({
            "outcome": col,
            "n_compared": int(both.sum()),
            "n_missing_mismatch": int((a.isna() != b.isna()).sum()),
            "n_value_mismatch": int((diff > 0).sum()),
            "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
        })
    return pd.DataFrame(rows)
