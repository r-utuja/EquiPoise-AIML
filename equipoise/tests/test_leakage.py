"""Leakage guard: only X_ columns may reach a baseline feature matrix."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from equipoise.features import (
    POST_RANDOMIZATION_TABLES,
    LeakageError,
    assert_no_leakage,
    baseline_feature_columns,
    baseline_feature_matrix,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "equipoise"


def test_feature_matrix_has_only_baseline_columns(synthetic: pd.DataFrame) -> None:
    X = baseline_feature_matrix(synthetic)
    assert len(X.columns) > 0
    assert all(c.startswith("X_") for c in X.columns)
    for forbidden in ("treatment", "pt_id", "study_eye"):
        assert forbidden not in X.columns
    assert not any(c.startswith(("Y_", "M_")) for c in X.columns)


def test_treatment_only_when_explicit(synthetic: pd.DataFrame) -> None:
    X = baseline_feature_matrix(synthetic, include_treatment=True)
    assert X.columns[-1] == "treatment"
    assert_no_leakage([*X.columns, "treatment_Aflibercept"], allow_treatment=True)
    with pytest.raises(LeakageError):
        assert_no_leakage(X.columns)


@pytest.mark.parametrize(
    "bad", ["Y_va_change_wk52", "Y_n_inj_yr1", "M_has_wk52", "M_final_status", "treatment",
            "treatment_Aflibercept", "pt_id", "study_eye"],
)
def test_any_non_baseline_column_is_rejected(synthetic: pd.DataFrame, bad: str) -> None:
    cols = [*baseline_feature_columns(synthetic.columns), bad]
    with pytest.raises(LeakageError, match=bad):
        assert_no_leakage(cols)


def test_outcomes_rejected_even_with_treatment_allowed(synthetic: pd.DataFrame) -> None:
    with pytest.raises(LeakageError):
        assert_no_leakage(["X_age", "treatment", "Y_va_change_wk52"], allow_treatment=True)


def test_feature_columns_cover_every_x_column(synthetic: pd.DataFrame) -> None:
    expected = [c for c in synthetic.columns if c.startswith("X_")]
    assert baseline_feature_columns(synthetic.columns) == expected


def test_package_never_reads_post_randomization_tables() -> None:
    """No package code (other than the list in features.py) names a post-randomization table."""
    pattern = re.compile("|".join(rf"\b{t}\b" for t in POST_RANDOMIZATION_TABLES))
    offenders = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if p.name != "features.py" and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
