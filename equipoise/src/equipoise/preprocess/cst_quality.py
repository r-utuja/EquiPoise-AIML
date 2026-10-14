"""Apply the reading-centre CST quality policy (primary vs sensitivity analysis).

The DRCR reading centre rated some central-subfield grades "Unacceptable" (scan off the fovea
or plotting errors), and some patients changed OCT machine between baseline and week 52. The
primary analysis keeps every value, as M0 and the published trial did (`keep`). The
sensitivity analysis (`missing`) treats flagged baseline CST, and CST change built from a
flagged or machine-switched scan, as missing, so the report can show whether the flags matter.
"""

from __future__ import annotations

import pandas as pd

POLICIES = ("keep", "missing")


def apply_cst_quality(df: pd.DataFrame, quality: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Return a copy of the analysis table with the policy applied (quality joined on pt_id)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown CST quality policy {policy!r}; choose from {POLICIES}")
    out = df.copy()
    if policy == "keep":
        return out
    q = quality.set_index("pt_id").loc[out["pt_id"]].reset_index(drop=True)
    q.index = out.index
    out.loc[q["M_cst_flag_baseline"] == 1, "X_cst_um"] = float("nan")
    bad52 = (q["M_cst_flag_baseline"] == 1) | (q["M_cst_flag_wk52"] == 1) \
        | (q["M_oct_switch_wk52"] == 1)
    out.loc[bad52, "Y_cst_change_wk52"] = float("nan")
    bad104 = (q["M_cst_flag_baseline"] == 1) | (q["M_cst_flag_wk104"] == 1)
    out.loc[bad104, "Y_cst_change_wk104"] = float("nan")
    return out
