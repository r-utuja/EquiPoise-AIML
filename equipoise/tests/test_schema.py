"""Schema validation of the analysis and visit tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equipoise.data.schema import (
    ANALYSIS_COLUMNS,
    SchemaError,
    dtype_name,
    validate_analysis,
    validate_visits,
)


def test_synthetic_analysis_valid(synthetic: pd.DataFrame) -> None:
    validate_analysis(synthetic, real=False)
    assert list(synthetic.columns) == [c.name for c in ANALYSIS_COLUMNS]


def test_synthetic_visits_valid(synthetic: pd.DataFrame, synthetic_visits: pd.DataFrame) -> None:
    validate_visits(synthetic_visits, real=False, analysis=synthetic)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.drop(columns="X_hba1c"), "missing columns"),
        (lambda d: d.assign(X_extra=1), "unexpected columns"),
        (lambda d: d.assign(X_va_letters=d.X_va_letters.where(d.index != 0, 90)), "above 78"),
        (lambda d: d.assign(X_female=d.X_female.where(d.index != 0, 2)), "0/1"),
        (lambda d: d.assign(treatment=d.treatment.where(d.index != 0, "Placebo")), "categories"),
        (lambda d: d.assign(X_age=d.X_age.astype(float).where(d.index != 0, np.nan)),
         "missing values"),
        (lambda d: d.assign(M_has_wk52=1), "M_has_wk52 is inconsistent"),
        (lambda d: d.assign(X_va_below_69=0), "X_va_below_69 is inconsistent"),
        (lambda d: pd.concat([d, d.iloc[[0]]]), "not unique"),
    ],
)
def test_corrupted_tables_fail(synthetic: pd.DataFrame, mutate, message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        validate_analysis(mutate(synthetic), real=False)


def test_real_mode_checks_trial_size(synthetic: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="expected 660 patients"):
        validate_analysis(synthetic.iloc[:600], real=True)


def test_visit_arm_mismatch_fails(synthetic: pd.DataFrame, synthetic_visits: pd.DataFrame
                                  ) -> None:
    v = synthetic_visits.copy()
    first = v.pt_id == v.pt_id.iloc[0]
    arm = v.loc[first, "treatment"].iloc[0]
    v.loc[first, "treatment"] = "Bevacizumab" if arm != "Bevacizumab" else "Aflibercept"
    with pytest.raises(SchemaError, match="treatment differs"):
        validate_visits(v, real=False, analysis=synthetic)


def test_real_table_matches_schema(real_analysis: pd.DataFrame) -> None:
    validate_analysis(real_analysis, real=True)
    spec = {c.name: c.dtype for c in ANALYSIS_COLUMNS}
    assert {c: dtype_name(t) for c, t in real_analysis.dtypes.items()} == spec


def test_synthetic_dtypes_match_schema(synthetic: pd.DataFrame) -> None:
    spec = {c.name: c.dtype for c in ANALYSIS_COLUMNS}
    assert {c: dtype_name(t) for c, t in synthetic.dtypes.items()} == spec
