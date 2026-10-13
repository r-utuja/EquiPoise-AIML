"""
Build the Equipoise visit table from the DRCR Protocol T public dataset (M7 input).

Output: one row per patient x completed study visit, study eye only:

  pt_id, treatment, days_from_rand, visit_label,
  va_letters               study-eye ETDRS letters        (TVAReVATest)
  cst_um, oct_machine      reading-centre CST, machine    (RCOCTGradingResult, GradeType ST)
  injected                 study-eye anti-VEGF given      (TIntraInj, as in M0)
  laser                    study-eye laser given          (TLaserTrt, as in M0)

How visits are matched
  The spine is TVAReVATest: every completed protocol visit has exactly one row there
  (missed visits and the 2-3-day post-injection safety visits have none). Every DRCR form
  carries the visit's ParentLoginVisitID, so OCT grades, injections and laser are joined to
  the visit by that exact key rather than by nearest date. `days_from_rand` is the visit date
  from TVisitInfo (day 0 = randomization = enrollment in Protocol T). As a sanity check the
  OCT scan date must lie within `visits.oct_window_days` of the visit date, and treatment
  dates within `visits.treatment_window_days` after it; otherwise the build fails.

Study-eye selection and the injection/laser filters reuse build_protocolT_table.py (M0).
The week-52 vision change derived here is cross-checked against Y_va_change_wk52 in
data/protocolT_analysis.csv; the build fails if any patient differs by more than the
configured tolerance.

Usage:
    python scripts/build_protocolT_visits.py "<path to 'Data Tables - Text files'>" <output.csv>

Data source attribution (required by the DRCR terms of use):
    The source of the data is the DRCR Retina Network (FAIN UG1EY014231).
    Protocol T: A Comparative Effectiveness Study of Intravitreal Aflibercept,
    Bevacizumab and Ranibizumab for Diabetic Macular Edema [Data file].
    Retrieved from https://public.jaeb.org/drcrnet. The analyses content and
    conclusions presented herein are solely the responsibility of the authors
    and have not been reviewed or approved by the DRCR Retina Network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from build_protocolT_table import load, pick_eye  # M0 helpers (same folder)

from equipoise import tracking
from equipoise.config import load_config
from equipoise.data.crosscheck import cross_check
from equipoise.data.schema import (
    ARM_SIZES,
    N_PATIENTS,
    VISITS_COLUMNS,
    numeric_casts,
    validate_analysis,
    validate_visits,
)

VISIT_KEY = "ParentLoginVisitID"


class BuildError(RuntimeError):
    """Raised when the raw tables do not join the way the build assumes."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise BuildError(message)


def spine(folder: Path) -> pd.DataFrame:
    """Patients with arm and study eye, as in M0."""
    trt = load(folder, "TTrtGroupUnmasked")[["PtID", "TTrtGroup", "RandEye"]]
    trt = trt.rename(columns={"PtID": "pt_id", "TTrtGroup": "treatment", "RandEye": "study_eye"})
    require(trt.pt_id.is_unique and len(trt) == N_PATIENTS, "expected 660 unique patients")
    require(trt.treatment.value_counts().to_dict() == ARM_SIZES, "arm sizes differ from trial")
    return trt


def visits_with_va(folder: Path, pts: pd.DataFrame) -> pd.DataFrame:
    """One row per completed visit with the study-eye VA and the visit date."""
    eye = pts.set_index("pt_id")["study_eye"]
    va = load(folder, "TVAReVATest").rename(columns={"PtID": "pt_id"})
    require(va[VISIT_KEY].is_unique, "TVAReVATest: duplicate visit keys")
    require(va.pt_id.isin(eye.index).all(), "TVAReVATest: unknown patients")
    va["va_letters"] = pick_eye(va, va.pt_id.map(eye), "Etdrs")

    vi = load(folder, "TVisitInfo")
    require(vi[VISIT_KEY].is_unique, "TVisitInfo: duplicate visit keys")
    v = va.merge(vi[[VISIT_KEY, "VisitDtDaysFromEnroll", "VisitMiss"]], on=VISIT_KEY,
                 how="left", validate="one_to_one", indicator=True)
    require((v._merge == "both").all(), "TVAReVATest rows without a TVisitInfo visit")
    require(v.VisitMiss.isna().all(), "VA recorded at a visit flagged as missed")
    require(v.VisitDtDaysFromEnroll.notna().all(), "visit without a date")
    v["days_from_rand"] = v.VisitDtDaysFromEnroll.astype(int)
    baseline = v[v.Visit == "Baseline"]
    require(len(baseline) == N_PATIENTS and baseline.pt_id.is_unique,
            "every patient needs exactly one baseline visit")
    require((baseline.days_from_rand == 0).all(), "baseline visit not on day 0")
    v = v.merge(pts[["pt_id", "treatment"]], on="pt_id", validate="many_to_one")
    return v[["pt_id", "treatment", VISIT_KEY, "days_from_rand", "Visit", "va_letters"]].rename(
        columns={"Visit": "visit_label"})


def attach_oct(folder: Path, v: pd.DataFrame, eye: pd.Series, window: int) -> pd.DataFrame:
    """Study-eye reading-centre CST (standard grades only), joined by visit key."""
    oc = load(folder, "RCOCTGradingResult").rename(columns={"PtID": "pt_id"})
    oc = oc[(oc.GradeType == "ST") & (oc.OCTScanEye == oc.pt_id.map(eye))]
    require(oc[VISIT_KEY].is_unique, "RCOCTGradingResult: several study-eye ST grades per visit")
    oc = oc[[VISIT_KEY, "OCTScanDtDaysFromEnroll", "OCTCentSubCalc", "OCTMachine"]]
    out = v.merge(oc, on=VISIT_KEY, how="left", validate="one_to_one")
    require(oc[VISIT_KEY].isin(v[VISIT_KEY]).all(), "OCT grade at a visit with no VA row")
    gap = (out.OCTScanDtDaysFromEnroll - out.days_from_rand).abs()
    require(not (gap > window).any(),
            f"{int((gap > window).sum())} OCT scans more than {window} days from their visit")
    return out.rename(columns={"OCTCentSubCalc": "cst_um", "OCTMachine": "oct_machine"}).drop(
        columns="OCTScanDtDaysFromEnroll")


def attach_treatments(folder: Path, v: pd.DataFrame, eye: pd.Series, window: int
                      ) -> pd.DataFrame:
    """Injection and laser flags, with the same study-eye / 'actually given' filters as M0."""
    ij = load(folder, "TIntraInj").rename(columns={"PtID": "pt_id"})
    ij = ij[ij.InjNotDone.isna() & (ij.TrtEye == ij.StudyEye)]
    require((ij.StudyEye == ij.pt_id.map(eye)).all(), "TIntraInj StudyEye differs from RandEye")
    la = load(folder, "TLaserTrt").rename(columns={"PtID": "pt_id"})
    la = la[la.LaserTrtNotDone.isna()]
    la = la[la.LasEyeTrt == la.pt_id.map(eye)]

    day = v.set_index(VISIT_KEY)["days_from_rand"]
    for name, df, key, date in [
        ("injected", ij, "ParentLoginVisitIDTrt", "InjectDtDaysFromEnroll"),
        ("laser", la, "ParentLogInVisitIDTrt", "LasTrtExamDtDaysFromEnroll"),
    ]:
        require(df[key].is_unique, f"{name}: several study-eye treatments at one visit")
        require(df[key].isin(day.index).all(), f"{name}: treatment at a visit with no VA row")
        lag = df[date] - df[key].map(day)
        require(lag.between(0, window).all(),
                f"{name}: {int((~lag.between(0, window)).sum())} treatments outside "
                f"0-{window} days after their visit")
        v[name] = v[VISIT_KEY].isin(df[key]).astype(int)
    return v


def main(folder: Path, out: Path) -> None:
    cfg = load_config()
    w = cfg.visits
    pts = spine(folder)
    eye = pts.set_index("pt_id")["study_eye"]
    v = visits_with_va(folder, pts)
    v = attach_oct(folder, v, eye, int(w["oct_window_days"]))
    v = attach_treatments(folder, v, eye, int(w["treatment_window_days"]))

    cols = [c.name for c in VISITS_COLUMNS]
    v = (v.sort_values(["pt_id", "days_from_rand", VISIT_KEY])[cols]
          .astype(numeric_casts(VISITS_COLUMNS)).reset_index(drop=True))

    # ------------------------------------------------------------- checks
    analysis = validate_analysis(pd.read_csv(cfg.paths.analysis_table, low_memory=False),
                                 real=True)
    validate_visits(v, real=True, analysis=analysis)
    require(v.pt_id.nunique() == N_PATIENTS, "visit table must cover all 660 patients")
    report = cross_check(v, analysis)
    print(report.to_string(index=False))
    tol = float(w["cross_check_tolerance_letters"])
    va52 = report.set_index("outcome").loc["Y_va_change_wk52"]
    require(va52.n_missing_mismatch == 0 and not (va52.max_abs_diff > tol),
            f"week-52 vision change disagrees with the analysis table:\n{va52}")
    if out.resolve() == cfg.paths.visits_table.resolve():  # not for test/scratch builds
        tracking.log_table(cfg, "build", report, "visits_crosscheck")

    out.parent.mkdir(parents=True, exist_ok=True)
    v.to_csv(out, index=False, lineterminator="\n")  # LF: same bytes on every OS
    per_pt = v.groupby("pt_id").size()
    print(f"wrote {out}  shape={v.shape}  visits/patient median={int(np.median(per_pt))} "
          f"range={per_pt.min()}-{per_pt.max()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
