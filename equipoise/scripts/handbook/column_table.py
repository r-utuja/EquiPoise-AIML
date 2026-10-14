"""Handbook helper: one row per column of the analysis table, from the schema plus the data.

Prints a Markdown table (Column | Plain meaning | Role | Type | Unit / levels | Allowed range |
Missing | Typical value) for the handbook's data chapter. Only aggregates are printed (count of
missing values, median and IQR, or the most common level with its share), never a patient row.

Usage:
    python scripts/handbook/column_table.py            # real data (data/protocolT_analysis.csv)
    python scripts/handbook/column_table.py --synthetic
"""

from __future__ import annotations

import argparse

import pandas as pd

from equipoise.config import load_config
from equipoise.data.load import load_analysis
from equipoise.data.schema import ANALYSIS_COLUMNS

ROLE = {"id": "identifier", "feature": "X_ feature", "outcome": "Y_ outcome",
        "meta": "M_ bookkeeping"}


def unit(desc: str) -> str:
    if "(" in desc and desc.endswith(")"):
        return desc[desc.rfind("(") + 1:-1]
    return ""


def typical(s: pd.Series, binary: bool, categorical: bool) -> str:
    s = s.dropna()
    if s.empty:
        return "n/a"
    if binary:
        return f"{int(s.sum())} of {len(s)} = 1 ({100 * s.mean():.1f}%)"
    if categorical:
        vc = s.value_counts()
        return f"most common: {vc.index[0]} ({100 * vc.iloc[0] / len(s):.1f}%)"
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"median {med:.4g} (IQR {q1:.4g} to {q3:.4g})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    df = load_analysis(cfg, synthetic=args.synthetic)
    print(f"Source: {'data/synthetic/' if args.synthetic else 'data/'}protocolT_analysis.csv, "
          f"{len(df)} rows x {df.shape[1]} columns\n")
    print("| Column | Plain meaning | Role | Stored type | Unit / levels | Allowed range | "
          "Missing | Typical value (aggregate) |")
    print("|---|---|---|---|---|---|---|---|")
    for c in ANALYSIS_COLUMNS:
        s = df[c.name]
        categorical = c.categories is not None
        levels = " / ".join(c.categories) if categorical else unit(c.description)
        if c.binary:
            levels = "0 = no, 1 = yes"
        rng = f"{c.min:g} to {c.max:g}" if c.min is not None and c.max is not None else (
            f">= {c.min:g}" if c.min is not None else "")
        val = "(identifier, not summarised)" if c.name == "pt_id" else typical(
            s, c.binary, categorical)
        print(f"| `{c.name}` | {c.description} | {ROLE[c.role]} | {c.dtype} | {levels} | "
              f"{rng} | {int(s.isna().sum())} | {val} |")


if __name__ == "__main__":
    main()
