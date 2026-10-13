"""The single gate through which baseline features reach any model.

Only `X_` columns (measured at or before randomization) may be model inputs. `Y_` columns
are targets, `M_` columns are bookkeeping, and `pt_id` / `study_eye` are identifiers. The
randomized `treatment` may enter a matrix only when a learner explicitly takes it as the
treatment variable (`allow_treatment=True`). `assert_no_leakage` enforces this and is
checked by `tests/test_leakage.py`.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

FEATURE_PREFIX = "X_"
OUTCOME_PREFIX = "Y_"
META_PREFIX = "M_"
# Early post-baseline vision (weeks 4/8/12). Allowed ONLY in longitudinal/early_response.py.
EARLY_PREFIX = "E_"
TREATMENT_COL = "treatment"
ID_COLS = ("pt_id", "study_eye")

# Raw DRCR tables recorded after randomization. They are never a source of features.
POST_RANDOMIZATION_TABLES = (
    "TAdvEvent",
    "TMedication",
    "TMedUpFU",
    "TFUExamQuestions",
    "TOcularExamFU",
    "TVisitComp",
    "TNonStudyEyeInj",
    "TStudyProcedure",
    "TPostRandPtFinalStatus",
)


class LeakageError(RuntimeError):
    """Raised when a post-randomization or identifier column reaches a feature matrix."""


def baseline_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return the `X_` columns, in their original order."""
    return [c for c in columns if c.startswith(FEATURE_PREFIX)]


def _is_treatment_column(col: str) -> bool:
    return col == TREATMENT_COL or col.startswith(f"{TREATMENT_COL}_")


def assert_no_leakage(columns: Iterable[str], allow_treatment: bool = False,
                      allow_early: bool = False) -> None:
    """Raise `LeakageError` if any column other than `X_` features (and, when allowed,
    the treatment variable or its one-hot dummies) is present.

    `allow_early=True` additionally admits `E_` early-vision columns; only the M7
    early-response model may pass it.
    """
    bad = []
    for col in columns:
        col = str(col)
        if col.startswith(FEATURE_PREFIX):
            continue
        if allow_treatment and _is_treatment_column(col):
            continue
        if allow_early and col.startswith(EARLY_PREFIX):
            continue
        bad.append(col)
    if bad:
        raise LeakageError(
            f"Non-baseline columns in a feature matrix: {bad}. Only X_ columns are model inputs"
            + ("" if allow_treatment else "; treatment needs allow_treatment=True") + "."
        )


def baseline_feature_matrix(
    df: pd.DataFrame, include_treatment: bool = False
) -> pd.DataFrame:
    """Return a copy of the baseline feature matrix (optionally with `treatment` last)."""
    cols = baseline_feature_columns(df.columns)
    if include_treatment:
        cols = [*cols, TREATMENT_COL]
    assert_no_leakage(cols, allow_treatment=include_treatment)
    return df[cols].copy()
