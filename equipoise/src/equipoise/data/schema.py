"""Column schema and validation for the analysis table and the visit table.

The schema lists every column with its role (id / X_ feature / Y_ outcome / M_ bookkeeping),
storage dtype, allowed range or category levels and whether it may be missing. The columns
and dtypes match `data/protocolT_analysis.csv` built by `scripts/build_protocolT_table.py`
(README section 7). `validate_analysis` and `validate_visits` collect every problem and raise
one `SchemaError`, so a broken table fails loudly with a full list of what is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ARMS = ("Aflibercept", "Bevacizumab", "Ranibizumab")
ARM_SIZES = {"Aflibercept": 224, "Bevacizumab": 218, "Ranibizumab": 218}
N_PATIENTS = 660


class SchemaError(ValueError):
    """Raised when a table does not match its schema."""


@dataclass(frozen=True)
class ColumnSpec:
    """One column: storage dtype, role, allowed values and nullability."""

    name: str
    dtype: str  # "int64", "float64" or "str" (pandas string dtype)
    role: str  # "id", "feature", "outcome", "meta", "visit"
    description: str
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    categories: tuple[str, ...] | None = None
    binary: bool = False
    integer: bool = False  # integer-valued even when stored as float (because of NaN)


def _bin(name: str, desc: str, role: str = "feature", nullable: bool = False) -> ColumnSpec:
    dtype = "float64" if nullable else "int64"
    return ColumnSpec(name, dtype, role, desc, nullable, 0, 1, binary=True, integer=True)


RACES = (
    "White",
    "Black/African American",
    "Asian",
    "More than one race",
    "Native Hawaiian/Other Pacific Islander",
    "American Indian/Alaskan Native",
)
DIABETES_TYPES = ("Type 2", "Type 1", "Uncertain")
DR_SEVERITY_CLINICAL = (
    "Microaneurysms only",
    "Mild/moderate NPDR",
    "Severe NPDR",
    "PDR and/or prior scatter",
)
OCT_MACHINES = ("Zeiss Stratus", "Zeiss Cirrus", "Heidelberg Spectralis")
FINAL_STATUSES = ("Completed", "DropPostOutcome", "DropPreOutcome")

# Order matches build_protocolT_table.py: ids, then sorted X_, sorted Y_, sorted M_.
ANALYSIS_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("pt_id", "int64", "id", "DRCR patient ID", min=1, integer=True),
    ColumnSpec("treatment", "str", "id", "Randomized arm", categories=ARMS),
    ColumnSpec("study_eye", "str", "id", "Randomized eye (OD right / OS left)",
               categories=("OD", "OS")),
    ColumnSpec("X_age", "int64", "feature", "Age at enrollment (years)", min=18, max=100,
               integer=True),
    ColumnSpec("X_bmi", "float64", "feature", "Body-mass index (kg/m^2)", True, 12, 80),
    _bin("X_cad", "Coronary artery disease"),
    ColumnSpec("X_cst_um", "float64", "feature",
               "Central subfield thickness, machine units (um)", True, 150, 1200),
    ColumnSpec("X_dbp", "float64", "feature", "Diastolic blood pressure (mmHg)", min=30, max=150),
    ColumnSpec("X_diabetes_duration_yrs", "int64", "feature", "Diabetes duration (years)",
               min=0, max=90, integer=True),
    ColumnSpec("X_diabetes_type", "str", "feature", "Diabetes type",
               categories=DIABETES_TYPES),
    ColumnSpec("X_dr_severity_clinical", "str", "feature", "Clinical DR severity",
               True, categories=DR_SEVERITY_CLINICAL),
    ColumnSpec("X_dr_severity_rc", "float64", "feature",
               "Reading-centre DR severity (ETDRS recoded 1-10)", True, 1, 10, integer=True),
    _bin("X_epiretinal_membrane", "Epiretinal membrane on OCT", nullable=True),
    ColumnSpec("X_fellow_va_letters", "float64", "feature", "Fellow-eye visual acuity (letters)",
               min=0, max=100, integer=True),
    _bin("X_female", "Female sex"),
    ColumnSpec("X_hba1c", "float64", "feature", "HbA1c (%)", True, 3, 20),
    _bin("X_hispanic", "Hispanic or Latino"),
    _bin("X_hypertension", "Hypertension"),
    _bin("X_insulin", "Uses insulin"),
    ColumnSpec("X_iop", "float64", "feature", "Intra-ocular pressure (mmHg)", min=0, max=60,
               integer=True),
    ColumnSpec("X_log_uacr", "float64", "feature", "log(1 + urine albumin/creatinine ratio)",
               True, 0, 15),
    ColumnSpec("X_oct_machine", "str", "feature", "OCT machine used at baseline", True,
               categories=OCT_MACHINES),
    _bin("X_prior_antivegf", "Prior anti-VEGF for DME in study eye"),
    _bin("X_prior_dme_trt", "Any prior DME treatment in study eye"),
    _bin("X_prior_focal_laser", "Prior focal/grid laser in study eye"),
    _bin("X_prior_mi", "Prior myocardial infarction"),
    _bin("X_prior_prp", "Prior panretinal photocoagulation in study eye"),
    _bin("X_prior_stroke", "Prior stroke"),
    _bin("X_pseudophakic", "Pseudophakic study eye"),
    ColumnSpec("X_race", "str", "feature", "Race", True, categories=RACES),
    _bin("X_renal_disease", "Renal disease"),
    ColumnSpec("X_sbp", "float64", "feature", "Systolic blood pressure (mmHg)", min=60, max=250),
    _bin("X_smoker_current", "Current smoker"),
    _bin("X_subretinal_fluid", "Subretinal fluid in centre subfield", nullable=True),
    _bin("X_va_below_69", "Baseline vision 20/50 or worse (< 69 letters)"),
    ColumnSpec("X_va_letters", "float64", "feature", "Baseline study-eye visual acuity (letters)",
               min=24, max=78, integer=True),
    ColumnSpec("Y_cst_change_wk104", "float64", "outcome", "CST change at 2 years (um)",
               True, -1000, 600),
    ColumnSpec("Y_cst_change_wk52", "float64", "outcome", "CST change at 1 year (um)",
               True, -1000, 600),
    _bin("Y_laser_any", "Rescue laser at any time", role="outcome"),
    _bin("Y_laser_yr1", "Rescue laser before the 1-year visit", role="outcome"),
    ColumnSpec("Y_n_inj_total", "int64", "outcome", "Study-eye injections over 2 years",
               min=0, max=30, integer=True),
    ColumnSpec("Y_n_inj_yr1", "int64", "outcome", "Study-eye injections before 1-year visit",
               min=0, max=14, integer=True),
    ColumnSpec("Y_va_change_wk104", "float64", "outcome", "Vision change at 2 years (letters)",
               True, -100, 100, integer=True),
    ColumnSpec("Y_va_change_wk52", "float64", "outcome",
               "Vision change at 1 year (letters), primary outcome", True, -100, 100,
               integer=True),
    ColumnSpec("M_final_status", "str", "meta", "Final study status",
               categories=FINAL_STATUSES),
    _bin("M_has_wk104", "Has 2-year vision", role="meta"),
    _bin("M_has_wk52", "Has 1-year vision", role="meta"),
)

VISITS_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("pt_id", "int64", "id", "DRCR patient ID", min=1, integer=True),
    ColumnSpec("treatment", "str", "id", "Randomized arm", categories=ARMS),
    ColumnSpec("days_from_rand", "int64", "visit", "Days from randomization", min=0,
               max=1000, integer=True),  # late 104-week visits reach ~908 days
    ColumnSpec("visit_label", "str", "visit", "Protocol visit label", True),
    ColumnSpec("va_letters", "float64", "visit", "Study-eye visual acuity (letters)", True,
               0, 100, integer=True),
    # Floor is physical impossibility, not plausibility: the reading centre recorded a few
    # 93-96 um grades (likely segmentation errors); M1 flags outliers, the build keeps raw.
    ColumnSpec("cst_um", "float64", "visit", "Central subfield thickness (um, machine units)",
               True, 50, 1500),
    ColumnSpec("oct_machine", "str", "visit", "OCT machine", True, categories=OCT_MACHINES),
    _bin("injected", "Study-eye anti-VEGF injection given at this visit", role="visit"),
    _bin("laser", "Study-eye laser given at this visit", role="visit"),
)

# Reading-centre CST quality flags (scripts/build_cst_quality.py): bookkeeping, never features.
CST_QUALITY_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("pt_id", "int64", "id", "DRCR patient ID", min=1, integer=True),
    _bin("M_cst_flag_baseline", "Baseline CST grade rated unacceptable", role="meta"),
    _bin("M_cst_flag_wk52", "Week-52 CST grade rated unacceptable", role="meta"),
    _bin("M_cst_flag_wk104", "Week-104 CST grade rated unacceptable", role="meta"),
    _bin("M_oct_switch_wk52", "Week-52 OCT machine differs from baseline", role="meta"),
)

ANALYSIS_SCHEMA = {c.name: c for c in ANALYSIS_COLUMNS}
VISITS_SCHEMA = {c.name: c for c in VISITS_COLUMNS}


def _check_columns(df: pd.DataFrame, specs: tuple[ColumnSpec, ...], errors: list[str]) -> None:
    expected = [c.name for c in specs]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing:
        errors.append(f"missing columns: {missing}")
    if extra:
        errors.append(f"unexpected columns: {extra}")
    for spec in specs:
        if spec.name not in df.columns:
            continue
        s = df[spec.name]
        if s.isna().any() and not spec.nullable:
            errors.append(f"{spec.name}: {int(s.isna().sum())} missing values, not allowed")
        nn = s.dropna()
        if spec.categories is not None:
            bad = sorted(set(nn.astype(str)) - set(spec.categories))
            if bad:
                errors.append(f"{spec.name}: unexpected categories {bad}")
            continue
        if spec.dtype == "str":
            continue
        if not pd.api.types.is_numeric_dtype(s):
            errors.append(f"{spec.name}: expected numeric dtype, got {s.dtype}")
            continue
        if spec.integer and not np.allclose(nn, np.round(nn)):
            errors.append(f"{spec.name}: expected integer values")
        if spec.binary and not set(nn.unique()) <= {0, 1}:
            errors.append(f"{spec.name}: expected 0/1 values, got {sorted(nn.unique())[:5]}")
        if spec.min is not None and (nn < spec.min).any():
            errors.append(f"{spec.name}: {int((nn < spec.min).sum())} values below {spec.min}")
        if spec.max is not None and (nn > spec.max).any():
            errors.append(f"{spec.name}: {int((nn > spec.max).sum())} values above {spec.max}")


def _raise(errors: list[str], what: str) -> None:
    if errors:
        raise SchemaError(f"{what} failed validation:\n  - " + "\n  - ".join(errors))


def validate_analysis(df: pd.DataFrame, real: bool) -> pd.DataFrame:
    """Validate the one-row-per-patient analysis table; return it unchanged if valid.

    `real=True` additionally enforces the trial's 660 patients and 224/218/218 arm sizes.
    """
    errors: list[str] = []
    _check_columns(df, ANALYSIS_COLUMNS, errors)
    if not errors:
        if not df["pt_id"].is_unique:
            errors.append("pt_id is not unique")
        below = (df["X_va_letters"] < 69).astype(int)
        if not (below == df["X_va_below_69"]).all():
            errors.append("X_va_below_69 is inconsistent with X_va_letters < 69")
        for wk in (52, 104):
            has = df[f"Y_va_change_wk{wk}"].notna().astype(int)
            if not (has == df[f"M_has_wk{wk}"]).all():
                errors.append(f"M_has_wk{wk} is inconsistent with Y_va_change_wk{wk}")
        if (df["Y_n_inj_total"] < df["Y_n_inj_yr1"]).any():
            errors.append("Y_n_inj_total < Y_n_inj_yr1 for some patients")
        if (df["Y_laser_any"] < df["Y_laser_yr1"]).any():
            errors.append("Y_laser_any < Y_laser_yr1 for some patients")
        if real:
            if len(df) != N_PATIENTS:
                errors.append(f"expected {N_PATIENTS} patients, got {len(df)}")
            sizes = df["treatment"].value_counts().to_dict()
            if sizes != ARM_SIZES:
                errors.append(f"arm sizes {sizes} != {ARM_SIZES}")
    _raise(errors, "analysis table")
    return df


def validate_visits(df: pd.DataFrame, real: bool, analysis: pd.DataFrame | None = None
                    ) -> pd.DataFrame:
    """Validate the long patient x visit table; return it unchanged if valid.

    If `analysis` is given, every visit patient must exist there with the same arm.
    `real=True` additionally requires at most 660 distinct patients.
    """
    errors: list[str] = []
    _check_columns(df, VISITS_COLUMNS, errors)
    if not errors:
        if df.duplicated(["pt_id", "days_from_rand", "visit_label"]).any():
            errors.append("duplicate (pt_id, days_from_rand, visit_label) rows")
        if analysis is not None:
            arm = analysis.set_index("pt_id")["treatment"]
            unknown = set(df["pt_id"]) - set(arm.index)
            if unknown:
                errors.append(f"{len(unknown)} visit patients not in the analysis table")
            elif not (df["pt_id"].map(arm) == df["treatment"]).all():
                errors.append("visit treatment differs from analysis-table treatment")
        if real and df["pt_id"].nunique() > N_PATIENTS:
            errors.append(f"more than {N_PATIENTS} patients")
    _raise(errors, "visit table")
    return df


def validate_cst_quality(df: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
    """Validate the CST quality-flag table (one row per analysis-table patient)."""
    errors: list[str] = []
    _check_columns(df, CST_QUALITY_COLUMNS, errors)
    if not errors and set(df["pt_id"]) != set(analysis["pt_id"]):
        errors.append("patients differ from the analysis table")
    _raise(errors, "CST quality table")
    return df


def numeric_casts(specs: tuple[ColumnSpec, ...]) -> dict[str, str]:
    """dtype casts for the numeric columns only. Text columns are left as read, because
    `astype("str")` turns missing values into the string "nan" on pandas 2."""
    return {c.name: c.dtype for c in specs if c.dtype != "str"}


def dtype_name(dtype: object) -> str:
    """Normalized dtype name: any text dtype (pandas 2 object, pandas 3 str) -> "str"."""
    name = str(dtype)
    return "str" if name in {"object", "string", "str"} or name.startswith("string") else name


def feature_spec_table() -> pd.DataFrame:
    """Feature | Data Type | Description | Role table for the report (README section 7)."""
    return pd.DataFrame(
        [
            {"Feature": c.name, "Data Type": "category" if c.categories else
             ("binary" if c.binary else c.dtype), "Description": c.description, "Role": c.role}
            for c in ANALYSIS_COLUMNS
        ]
    )
