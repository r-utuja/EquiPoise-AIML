"""Generate fake Protocol T tables with exactly the real schema, for tests and CI.

Writes three files to data/synthetic/:
  protocolT_analysis.csv  one row per patient, same 47 columns/dtypes as the real table
  protocolT_visits.csv    one row per patient x visit, same columns as the real visit table
  protocolT_truth.csv     noise-free potential outcomes (week-52 letters) per arm, so tests
                          can check that methods recover the planted effect

Column names and dtypes come from the real `data/protocolT_analysis.csv` when it exists, and
otherwise from the schema in `equipoise.data.schema` (README section 7). No real values are
copied: every distribution below is a hand-set, plausible range for a DME trial population.

Planted treatment effect (config/default.yaml, `synthetic:`): relative to bevacizumab,
aflibercept adds `effect_aflibercept_va_below_69` letters when X_va_below_69 = 1 and
`effect_aflibercept_va_at_or_above_69` otherwise; ranibizumab adds `effect_ranibizumab`.

The visit table is generated first and the Y_ outcome columns are derived from it the same
way the real build script derives them, so the two tables agree exactly.

Usage:
    python scripts/make_synthetic_data.py [--n N] [--seed S] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from equipoise.config import load_config
from equipoise.data.schema import (
    ANALYSIS_COLUMNS,
    ARM_SIZES,
    CST_QUALITY_COLUMNS,
    DIABETES_TYPES,
    DR_SEVERITY_CLINICAL,
    OCT_MACHINES,
    RACES,
    VISITS_COLUMNS,
    dtype_name,
    numeric_casts,
    validate_analysis,
    validate_visits,
)

# ----------------------------------------------------------------------------------------
# Hand-set generator parameters (plausible DME-trial ranges; NOT estimated from real data).
# ----------------------------------------------------------------------------------------
RACE_PROBS = (0.78, 0.16, 0.02, 0.02, 0.01, 0.01)
DIABETES_TYPE_PROBS = (0.90, 0.08, 0.02)
DR_CLINICAL_PROBS = (0.03, 0.51, 0.26, 0.20)
OCT_MACHINE_PROBS = (0.04, 0.49, 0.47)
# Machine-specific CST offset vs Stratus (um): spectral-domain machines read thicker.
CST_MACHINE_OFFSET = {"Zeiss Stratus": 0.0, "Zeiss Cirrus": 50.0, "Heidelberg Spectralis": 70.0}
FEATURE_MISSING_RATE = {  # share of patients with the feature missing
    "X_bmi": 0.09, "X_cst_um": 0.015, "X_dr_severity_clinical": 0.004,
    "X_dr_severity_rc": 0.015, "X_epiretinal_membrane": 0.002, "X_hba1c": 0.01,
    "X_log_uacr": 0.04, "X_oct_machine": 0.002, "X_race": 0.035, "X_subretinal_fluid": 0.002,
}
VA_LOW, VA_HIGH = 24, 78           # eligibility range (letters)
VA_SUBGROUP_CUT = 69                # 20/50 or worse
# Outcome model for week-52 letters under bevacizumab: worse baseline vision gains more.
BASE_GAIN, GAIN_PER_LETTER_BELOW_65, GAIN_AGE_SLOPE = 9.0, 0.35, -0.08
# Year-2 change as a share of the year-1 change plus extra noise.
YEAR2_CARRYOVER, YEAR2_NOISE_SD = 0.95, 6.0
# CST response at week 52 as a share of excess thickness over 250 um, per arm.
CST_RESPONSE = {"Aflibercept": 0.55, "Bevacizumab": 0.35, "Ranibizumab": 0.48}
CST_NOISE_SD = 60.0
# Per-visit injection probabilities after the 6 mandatory 4-weekly injections.
INJ_PROB_YR1 = {"Aflibercept": 0.40, "Bevacizumab": 0.55, "Ranibizumab": 0.50}
INJ_PROB_YR2 = {"Aflibercept": 0.30, "Bevacizumab": 0.40, "Ranibizumab": 0.35}
MANDATORY_INJECTIONS = 6
LASER_PROB_YR1 = {"Aflibercept": 0.37, "Bevacizumab": 0.56, "Ranibizumab": 0.46}
LASER_PROB_YR2_EXTRA = {"Aflibercept": 0.06, "Bevacizumab": 0.08, "Ranibizumab": 0.07}
LASER_EARLIEST_WEEK = 24
# Dropout: share leaving before week 52 / between weeks 52 and 104; a few completers miss
# the week-52 visit itself.
DROP_PRE52, DROP_POST52, MISSED_WK52_VISIT = 0.064, 0.08, 0.003
EXTRA_CST_MISSING_WK52 = 0.02
VISIT_JITTER_DAYS = 6
VISIT_WEEKS = tuple(range(0, 105, 4))              # every 4 weeks through week 104
OCT_WEEKS = (0, 4, 8, 12, 16, 20, 24, 32, 40, 52, 64, 78, 104)
VA_VISIT_NOISE_SD = 3.0
VA_TIME_CONSTANT_WEEKS = 10.0                      # speed of approach to the week-52 gain


def visit_label(week: int) -> str:
    """DRCR-style label: year-2 visits other than week 104 are all 'Post 52 week'."""
    if week == 0:
        return "Baseline"
    if 52 < week < 104:
        return "Post 52 week"
    return f"{week} week"


def schema_dtypes(real_table: Path) -> dict[str, str]:
    """Column -> dtype, from the real table if present, else from the schema."""
    spec = {c.name: c.dtype for c in ANALYSIS_COLUMNS}
    if not real_table.is_file():
        return spec
    real = {c: dtype_name(t) for c, t in pd.read_csv(real_table, low_memory=False).dtypes.items()}
    if list(real) != list(spec):
        raise SystemExit(f"Real table columns differ from schema:\n{list(real)}\n{list(spec)}")
    return real


def arm_assignment(n: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly permuted arms in the trial's 224:218:218 proportions."""
    total = sum(ARM_SIZES.values())
    counts = {a: round(n * k / total) for a, k in ARM_SIZES.items()}
    counts["Aflibercept"] += n - sum(counts.values())
    arms = np.concatenate([[a] * k for a, k in counts.items()])
    return rng.permutation(arms)


def baseline(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Baseline (X_) features plus identifiers."""
    b = lambda p: rng.binomial(1, p, n)  # noqa: E731
    df = pd.DataFrame({
        "pt_id": np.sort(rng.choice(np.arange(1, 2 * n + 1), n, replace=False)),
        "treatment": arm_assignment(n, rng),
        "study_eye": rng.choice(["OD", "OS"], n),
    })
    df["X_age"] = np.clip(rng.normal(62, 10, n).round(), 24, 87).astype(int)
    df["X_female"] = b(0.45)
    df["X_race"] = rng.choice(RACES, n, p=RACE_PROBS)
    df["X_hispanic"] = b(0.18)
    df["X_diabetes_type"] = rng.choice(DIABETES_TYPES, n, p=DIABETES_TYPE_PROBS)
    df["X_diabetes_duration_yrs"] = np.minimum(
        rng.gamma(4.0, 4.5, n).round(), df["X_age"] - 5).clip(0).astype(int)
    type1 = (df["X_diabetes_type"] == "Type 1").to_numpy()
    df["X_insulin"] = np.where(type1, 1, b(0.55))
    df["X_hba1c"] = np.clip(rng.normal(7.7, 1.4, n), 4.9, 15).round(1)
    df["X_bmi"] = np.clip(rng.normal(32.5, 7, n), 18, 65).round(2)
    df["X_sbp"] = np.clip(rng.normal(138, 18, n), 90, 200).round(1)
    df["X_dbp"] = np.clip(rng.normal(77, 10, n), 50, 110).round(1)
    df["X_smoker_current"] = b(0.11)
    df["X_hypertension"] = b(0.80)
    df["X_prior_mi"] = b(0.08)
    df["X_cad"] = b(0.15)
    df["X_prior_stroke"] = b(0.06)
    df["X_renal_disease"] = b(0.12)
    df["X_log_uacr"] = np.clip(rng.normal(4.0, 1.7, n), 0.9, 9.8).round(4)
    df["X_va_letters"] = np.clip(rng.normal(65, 11, n).round(), VA_LOW, VA_HIGH).astype(int)
    df["X_fellow_va_letters"] = np.clip(rng.normal(74, 16, n).round(), 0, 98).astype(int)
    machine = rng.choice(OCT_MACHINES, n, p=OCT_MACHINE_PROBS)
    df["X_oct_machine"] = machine
    true_cst = np.clip(rng.lognormal(np.log(380), 0.27, n), 220, 900)
    df["X_cst_um"] = (true_cst + pd.Series(machine).map(CST_MACHINE_OFFSET).to_numpy()).round()
    df["X_subretinal_fluid"] = b(0.27)
    df["X_epiretinal_membrane"] = b(0.10)
    df["X_iop"] = np.clip(rng.normal(16, 3, n).round(), 6, 29).astype(int)
    df["X_pseudophakic"] = b(0.24)
    dr = rng.choice(DR_SEVERITY_CLINICAL, n, p=DR_CLINICAL_PROBS)
    df["X_dr_severity_clinical"] = dr
    dr_level = pd.Series(dr).map({k: i for i, k in enumerate(DR_SEVERITY_CLINICAL)}).to_numpy()
    df["X_dr_severity_rc"] = np.clip(2 + 2 * dr_level + rng.integers(-1, 2, n), 1, 10)
    df["X_prior_prp"] = np.where(dr_level == 3, b(0.6), b(0.03))
    df["X_prior_antivegf"] = b(0.11)
    df["X_prior_focal_laser"] = b(0.36)
    df["X_prior_dme_trt"] = np.maximum.reduce(
        [df["X_prior_antivegf"], df["X_prior_focal_laser"], b(0.25)])
    df["X_va_below_69"] = (df["X_va_letters"] < VA_SUBGROUP_CUT).astype(int)
    return df


def potential_outcomes(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    """Noise-free week-52 letters gained under each arm (the planted truth)."""
    mu0 = (BASE_GAIN + GAIN_PER_LETTER_BELOW_65 * (65 - df["X_va_letters"])
           + GAIN_AGE_SLOPE * (df["X_age"] - 62))
    below = df["X_va_below_69"] == 1
    tau_afl = np.where(below, s["effect_aflibercept_va_below_69"],
                       s["effect_aflibercept_va_at_or_above_69"])
    tau_ran = np.full(len(df), float(s["effect_ranibizumab"]))
    return pd.DataFrame({
        "pt_id": df["pt_id"],
        "mu_Bevacizumab": mu0,
        "mu_Aflibercept": mu0 + tau_afl,
        "mu_Ranibizumab": mu0 + tau_ran,
        "tau_Aflibercept": tau_afl,
        "tau_Ranibizumab": tau_ran,
    })


def visits(df: pd.DataFrame, truth: pd.DataFrame, noise_sd: float,
           rng: np.random.Generator) -> pd.DataFrame:
    """Per-visit study-eye vision, CST, injections and laser, with dropout."""
    rows = []
    for i, p in enumerate(df.itertuples(index=False)):
        arm = p.treatment
        u = rng.random()
        if u < DROP_PRE52:
            last_week = int(rng.choice([w for w in VISIT_WEEKS if 0 < w < 52]))
        elif u < DROP_PRE52 + DROP_POST52:
            last_week = int(rng.choice([w for w in VISIT_WEEKS if 52 < w < 104]))
        else:
            last_week = 104
        skip52 = last_week == 104 and rng.random() < MISSED_WK52_VISIT
        gain52 = round(truth[f"mu_{arm}"].iat[i] + rng.normal(0, noise_sd))
        va52 = int(np.clip(p.X_va_letters + gain52, 0, 100))
        gain104 = YEAR2_CARRYOVER * (va52 - p.X_va_letters) + rng.normal(0, YEAR2_NOISE_SD)
        va104 = int(np.clip(round(p.X_va_letters + gain104), 0, 100))
        cst0 = p.X_cst_um
        cst52 = (np.nan if np.isnan(cst0) else max(
            150.0, cst0 - CST_RESPONSE[arm] * max(cst0 - 250, 0) + rng.normal(0, CST_NOISE_SD)))
        cst104 = np.nan if np.isnan(cst52) else max(150.0, cst52 + rng.normal(-10, 30))
        laser_week = None
        if rng.random() < LASER_PROB_YR1[arm]:
            laser_week = int(rng.choice([w for w in VISIT_WEEKS if LASER_EARLIEST_WEEK <= w < 52]))
        elif rng.random() < LASER_PROB_YR2_EXTRA[arm]:
            laser_week = int(rng.choice([w for w in VISIT_WEEKS if 52 < w < 104]))
        for wk in VISIT_WEEKS:
            if wk > last_week or (skip52 and wk == 52):
                continue
            day = 0 if wk == 0 else wk * 7 + int(rng.integers(-VISIT_JITTER_DAYS,
                                                              VISIT_JITTER_DAYS + 1))
            if wk == 52:
                va = va52
            elif wk == 104:
                va = va104
            elif wk < 52:
                frac = (1 - np.exp(-wk / VA_TIME_CONSTANT_WEEKS)) / (
                    1 - np.exp(-52 / VA_TIME_CONSTANT_WEEKS))
                va = p.X_va_letters + frac * (va52 - p.X_va_letters)
                va = int(np.clip(round(va + (rng.normal(0, VA_VISIT_NOISE_SD) if wk else 0)),
                                 0, 100))
            else:
                frac = (wk - 52) / 52
                va = int(np.clip(round(va52 + frac * (va104 - va52)
                                       + rng.normal(0, VA_VISIT_NOISE_SD)), 0, 100))
            cst = np.nan
            if wk in OCT_WEEKS and not np.isnan(cst0):
                if wk == 0:
                    cst = cst0
                elif wk <= 52:
                    cst = round(cst0 + (cst52 - cst0) * min(1.0, wk / 24))
                else:
                    cst = round(cst52 + (cst104 - cst52) * (wk - 52) / 52)
            if wk < MANDATORY_INJECTIONS * 4:
                inj = 1
            elif wk < 52:
                inj = int(rng.random() < INJ_PROB_YR1[arm])
            elif wk < 104:
                inj = int(rng.random() < INJ_PROB_YR2[arm])
            else:
                inj = 0
            rows.append({
                "pt_id": p.pt_id, "treatment": arm, "days_from_rand": day,
                "visit_label": visit_label(wk),
                "va_letters": va, "cst_um": cst,
                "oct_machine": p.X_oct_machine if not np.isnan(cst) else np.nan,
                "injected": inj, "laser": int(laser_week == wk),
            })
    v = pd.DataFrame(rows)
    # a few week-52 OCT scans are missing even when the visit happened
    wk52 = v.index[v["visit_label"] == "52 week"]
    drop = rng.choice(wk52, int(EXTRA_CST_MISSING_WK52 * len(wk52)), replace=False)
    v.loc[drop, ["cst_um", "oct_machine"]] = np.nan
    return v


CST_FLAG_RATE = {"baseline": 0.2, "wk52": 0.2, "wk104": 0.05}  # share rated unacceptable


def cst_quality(df: pd.DataFrame, v: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Fake reading-centre quality flags; machine never switches in the synthetic data."""
    q = pd.DataFrame({"pt_id": df["pt_id"]})
    for key, rate in CST_FLAG_RATE.items():
        q[f"M_cst_flag_{key}"] = rng.binomial(1, rate, len(df))
    q["M_oct_switch_wk52"] = 0
    return q[[c.name for c in CST_QUALITY_COLUMNS]]


def derive_outcomes(df: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """Y_ and M_ columns from the visit table, mirroring build_protocolT_table.py."""
    df = df.copy()
    for wk in (52, 104):
        lab = f"{wk} week"
        va = v[v.visit_label == lab].set_index("pt_id")["va_letters"]
        cst = v[v.visit_label == lab].set_index("pt_id")["cst_um"]
        df[f"Y_va_change_wk{wk}"] = df["pt_id"].map(va) - df["X_va_letters"]
        df[f"Y_cst_change_wk{wk}"] = df["pt_id"].map(cst) - df["X_cst_um"]
    day52 = v[v.visit_label == "52 week"].set_index("pt_id")["days_from_rand"]
    cut = v["pt_id"].map(day52).fillna(364)
    inj, las = v[v.injected == 1], v[v.laser == 1]
    df["Y_n_inj_yr1"] = df["pt_id"].map(
        inj[inj.days_from_rand < cut[inj.index]].groupby("pt_id").size()).fillna(0).astype(int)
    df["Y_n_inj_total"] = df["pt_id"].map(inj.groupby("pt_id").size()).fillna(0).astype(int)
    df["Y_laser_yr1"] = df["pt_id"].isin(
        las[las.days_from_rand < cut[las.index]].pt_id).astype(int)
    df["Y_laser_any"] = df["pt_id"].isin(las.pt_id).astype(int)
    last_day = v.groupby("pt_id")["days_from_rand"].max()
    reached104 = df["pt_id"].map(last_day) >= 100 * 7
    df["M_final_status"] = np.where(
        reached104, "Completed",
        np.where(df["Y_va_change_wk52"].notna(), "DropPostOutcome", "DropPreOutcome"))
    df["M_has_wk52"] = df["Y_va_change_wk52"].notna().astype(int)
    df["M_has_wk104"] = df["Y_va_change_wk104"].notna().astype(int)
    return df


def blank_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Make some baseline features missing at roughly the real table's rates."""
    df = df.copy()
    for col, rate in FEATURE_MISSING_RATE.items():
        k = max(1, int(round(rate * len(df))))
        idx = rng.choice(df.index, k, replace=False)
        df.loc[idx, col] = np.nan
    return df


def main() -> None:
    cfg = load_config()
    s = cfg.synthetic
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=int(s["n_patients"]))
    ap.add_argument("--seed", type=int, default=int(s["seed"]))
    ap.add_argument("--out", type=Path, default=cfg.paths.synthetic_dir)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dtypes = schema_dtypes(cfg.paths.analysis_table)
    df = baseline(args.n, rng)
    # CST missing must be decided before visits are generated (OCT rows depend on it)
    df = blank_features(df, rng)
    truth = potential_outcomes(df, s)
    v = visits(df, truth, float(s["outcome_noise_sd"]), rng)
    df = derive_outcomes(df, v)

    casts = {c: t for c, t in dtypes.items() if t != "str"}
    df = df[list(dtypes)].astype(casts).sort_values("pt_id").reset_index(drop=True)
    v = v[[c.name for c in VISITS_COLUMNS]].astype(numeric_casts(VISITS_COLUMNS))
    v = v.sort_values(["pt_id", "days_from_rand"]).reset_index(drop=True)
    validate_analysis(df, real=False)
    validate_visits(v, real=False, analysis=df)

    args.out.mkdir(parents=True, exist_ok=True)
    # LF line endings so the committed files hash identically on every OS
    df.to_csv(args.out / cfg.paths.analysis_table.name, index=False, lineterminator="\n")
    v.to_csv(args.out / cfg.paths.visits_table.name, index=False, lineterminator="\n")
    cst_quality(df, v, rng).to_csv(args.out / cfg.paths.cst_quality_table.name, index=False,
                                   lineterminator="\n")
    truth.round(4).to_csv(args.out / cfg.paths.synthetic_truth_table.name, index=False,
                          lineterminator="\n")
    print(f"wrote {args.out}: analysis {df.shape}, visits {v.shape}, "
          f"missing wk52 = {int((df.M_has_wk52 == 0).sum())}")


if __name__ == "__main__":
    main()
