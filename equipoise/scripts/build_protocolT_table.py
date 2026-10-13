"""
Build the Equipoise analysis table from the DRCR Protocol T public dataset.

Output: one row per patient (= one study eye; every patient has exactly one
randomized study eye), with three groups of columns:

  ID / TREATMENT  ->  pt_id, treatment, study_eye
  X_  (features)  ->  measured AT or BEFORE randomization only (safe to use)
  Y_  (outcomes)  ->  measured AFTER randomization (targets, never features)

Plus a few M_ (meta) columns for missing-data bookkeeping.

Usage:
    python build_protocolT_table.py "<path to 'Data Tables - Text files'>" <output.csv>

Data source attribution (required by the DRCR terms of use):
    The source of the data is the DRCR Retina Network (FAIN UG1EY014231).
    Protocol T: A Comparative Effectiveness Study of Intravitreal Aflibercept,
    Bevacizumab and Ranibizumab for Diabetic Macular Edema [Data file].
    Retrieved from https://public.jaeb.org/drcrnet. The analyses content and
    conclusions presented herein are solely the responsibility of the authors
    and have not been reviewed or approved by the DRCR Retina Network.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(folder: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(folder / f"{name}.txt", sep="|", low_memory=False, encoding="latin1")


def pick_eye(df: pd.DataFrame, eye: pd.Series, stem: str) -> pd.Series:
    """Columns come in OD (right) / OS (left) pairs; return the study-eye value."""
    return np.where(eye == "OD", df[f"{stem}OD"], df[f"{stem}OS"])


def yes(s: pd.Series) -> pd.Series:
    """Map Yes / 1 / Checked -> 1, everything else (No, blank) -> 0."""
    return s.isin(["Yes", 1, 1.0, "1", True]).astype(int)


def main(folder: Path, out: Path) -> None:
    # ------------------------------------------------------------------ spine
    trt = load(folder, "TTrtGroupUnmasked")[["PtID", "TTrtGroup", "RandEye"]]
    ros = load(folder, "PtRoster")[["PtID", "AgeAsOfEnrollDt"]]
    status = load(folder, "TPtRoster")[["PtID", "TPtStatus"]]
    df = (trt.rename(columns={"PtID": "pt_id", "TTrtGroup": "treatment", "RandEye": "study_eye"})
             .merge(ros.rename(columns={"PtID": "pt_id", "AgeAsOfEnrollDt": "X_age"}), on="pt_id")
             .merge(status.rename(columns={"PtID": "pt_id", "TPtStatus": "M_final_status"}), on="pt_id"))
    eye = lambda d: d["pt_id"].map(df.set_index("pt_id")["study_eye"])

    # ------------------------------------------------- demographics & history
    h = load(folder, "THxBaseline").rename(columns={"PtID": "pt_id"})
    h = h[h.pt_id.isin(df.pt_id)].copy()
    e = eye(h)
    hx = pd.DataFrame({"pt_id": h.pt_id})
    hx["X_female"] = (h.Gender == "F").astype(int)
    hx["X_race"] = h.Race.replace({"Unknown/not reported": np.nan})
    hx["X_hispanic"] = (h.Ethnicity == "Hispanic or Latino").astype(int)
    hx["X_diabetes_type"] = h.DiabetesType
    diab_age = pd.to_numeric(h.DiabAge.replace({"<1": 0, "90 or older": 90}), errors="coerce")
    hx["_diab_age"] = diab_age.values
    hx["X_insulin"] = yes(h.InsulinUsed)
    hx["X_prior_dme_trt"] = yes(pd.Series(pick_eye(h, e, "DMETrtHx"), index=h.index))
    hx["X_prior_antivegf"] = yes(pd.Series(pick_eye(h, e, "DMETrtVEGFHx"), index=h.index))
    hx["X_prior_focal_laser"] = yes(pd.Series(pick_eye(h, e, "DMETrtFocalLasHx"), index=h.index))
    hx["X_prior_prp"] = yes(pd.Series(pick_eye(h, e, "PRPHx"), index=h.index))
    df = df.merge(hx, on="pt_id", how="left")
    df["X_diabetes_duration_yrs"] = (df.X_age - df.pop("_diab_age")).clip(lower=0)

    # ------------------------------------------------------------- HbA1c lab
    lab = load(folder, "TLab").rename(columns={"PtID": "pt_id"})
    lab["X_hba1c"] = lab.HgbA1c.fillna(lab.HbA1cLabVal)
    df = df.merge(lab[["pt_id", "X_hba1c"]], on="pt_id", how="left")

    # ---------------------------------------- general health at randomization
    q = load(folder, "TRandExamQuestions").rename(columns={"PtID": "pt_id"})
    hq = pd.DataFrame({"pt_id": q.pt_id})
    hq["X_smoker_current"] = (q.SmkCurrent == "Yes").astype(int)
    bmi = 703 * q.Weight / q.Height ** 2          # form records inches and pounds
    hq["X_bmi"] = bmi.where(bmi.between(12, 80))
    for col, name in [("HypertensionCond", "hypertension"), ("MICond", "prior_mi"),
                      ("CADCond", "cad"), ("StrokeCond", "prior_stroke"),
                      ("RenalDiseaseCond", "renal_disease")]:
        hq[f"X_{name}"] = yes(q[col])
    df = df.merge(hq, on="pt_id", how="left")

    bp = load(folder, "TBloodPressure").rename(columns={"PtID": "pt_id"})
    bp = bp[bp.Visit == "Baseline"]
    bp["X_sbp"] = bp[["BldPrSys1", "BldPrSys2", "BldPrSys3"]].mean(axis=1)
    bp["X_dbp"] = bp[["BldPrDia1", "BldPrDia2", "BldPrDia3"]].mean(axis=1)
    df = df.merge(bp[["pt_id", "X_sbp", "X_dbp"]], on="pt_id", how="left")

    u = load(folder, "TUrine").rename(columns={"PtID": "pt_id"})
    u = u[u.Visit == "Baseline"][["pt_id", "AlbCreatRatio"]]
    u["X_log_uacr"] = np.log1p(u.pop("AlbCreatRatio"))
    df = df.merge(u, on="pt_id", how="left")

    # -------------------------------------------------- baseline eye exam
    o = load(folder, "TOcularExamBaseline").rename(columns={"PtID": "pt_id"})
    e = eye(o)
    oe = pd.DataFrame({"pt_id": o.pt_id})
    oe["X_pseudophakic"] = (pick_eye(o, e, "LensStatus") == "PC IOL").astype(int)
    oe["X_dr_severity_clinical"] = pd.Series(pick_eye(o, e, "DRSeverity"), index=o.index).replace(
        {"Cannot determine": np.nan})
    df = df.merge(oe, on="pt_id", how="left")

    iop = load(folder, "TIOP").rename(columns={"PtID": "pt_id"})
    iop = iop[iop.Visit == "Baseline"].copy()
    iop["X_iop"] = pick_eye(iop, eye(iop), "IOP")
    df = df.merge(iop[["pt_id", "X_iop"]], on="pt_id", how="left")

    # reading-centre DR severity from fundus photos (ETDRS scale recoded 1-10; 90 = ungradable)
    ph = load(folder, "RCPhotoGradingResult").rename(columns={"PtID": "pt_id"})
    ph = ph[(ph.Visit == "Baseline") & (ph.GradeType == "ST")]
    ph = ph[ph.RCPhotoEyes == eye(ph)].drop_duplicates("pt_id")
    ph["X_dr_severity_rc"] = ph.DRSevRecode.where(ph.DRSevRecode < 90)
    df = df.merge(ph[["pt_id", "X_dr_severity_rc"]], on="pt_id", how="left")

    # ------------------------------------------- visual acuity (all visits)
    va = load(folder, "TVAReVATest").rename(columns={"PtID": "pt_id"})
    va["eye"] = eye(va)
    va["va_study"] = pick_eye(va, va.eye, "Etdrs")
    va["va_fellow"] = np.where(va.eye == "OD", va.EtdrsOS, va.EtdrsOD)
    vb = va[va.Visit == "Baseline"][["pt_id", "va_study", "va_fellow"]]
    vb.columns = ["pt_id", "X_va_letters", "X_fellow_va_letters"]
    df = df.merge(vb, on="pt_id", how="left")
    for wk in (52, 104):
        v = va[va.Visit == f"{wk} week"][["pt_id", "va_study"]].rename(columns={"va_study": f"_va{wk}"})
        df = df.merge(v.drop_duplicates("pt_id"), on="pt_id", how="left")
        df[f"Y_va_change_wk{wk}"] = df.pop(f"_va{wk}") - df.X_va_letters

    # --------------------------------------- OCT thickness (reading centre)
    oc = load(folder, "RCOCTGradingResult").rename(columns={"PtID": "pt_id"})
    oc = oc[(oc.GradeType == "ST") & (oc.OCTScanEye == eye(oc))]
    ob = oc[oc.Visit == "Baseline"].drop_duplicates("pt_id")
    ob = pd.DataFrame({
        "pt_id": ob.pt_id,
        "X_cst_um": ob.OCTCentSubCalc,
        "X_oct_machine": ob.OCTMachine,
        "X_subretinal_fluid": yes(ob.SubretinalFluidCentSub),
        "X_epiretinal_membrane": yes(ob.EpiMembrane),
    })
    df = df.merge(ob, on="pt_id", how="left")
    for wk in (52, 104):
        c = oc[oc.Visit == f"{wk} week"].drop_duplicates("pt_id")[["pt_id", "OCTCentSubCalc"]]
        df = df.merge(c.rename(columns={"OCTCentSubCalc": f"_cst{wk}"}), on="pt_id", how="left")
        df[f"Y_cst_change_wk{wk}"] = df.pop(f"_cst{wk}") - df.X_cst_um

    # ----------------------------------------------- treatment burden outcomes
    vi = load(folder, "TVisitInfo").rename(columns={"PtID": "pt_id"})
    wk52_day = (vi[vi.Visit == "52 week"].drop_duplicates("pt_id")
                  .set_index("pt_id")["VisitDtDaysFromEnroll"])
    cut52 = df.pt_id.map(wk52_day).fillna(364)   # count treatments BEFORE the 52-week visit

    ij = load(folder, "TIntraInj").rename(columns={"PtID": "pt_id"})
    ij = ij[ij.InjNotDone.isna() & (ij.TrtEye == ij.StudyEye)]     # study-eye injections actually given
    day = ij.InjectDtDaysFromEnroll
    ij_cut = ij.pt_id.map(dict(zip(df.pt_id, cut52)))
    df["Y_n_inj_yr1"] = df.pt_id.map(ij[day < ij_cut].groupby("pt_id").size()).fillna(0).astype(int)
    df["Y_n_inj_total"] = df.pt_id.map(ij.groupby("pt_id").size()).fillna(0).astype(int)

    la = load(folder, "TLaserTrt").rename(columns={"PtID": "pt_id"})
    la = la[la.LaserTrtNotDone.isna()]
    la = la[la.LasEyeTrt == la.pt_id.map(df.set_index("pt_id")["study_eye"])]
    la_cut = la.pt_id.map(dict(zip(df.pt_id, cut52)))
    df["Y_laser_yr1"] = df.pt_id.isin(la[la.LasTrtExamDtDaysFromEnroll < la_cut].pt_id).astype(int)
    df["Y_laser_any"] = df.pt_id.isin(la.pt_id).astype(int)

    # ---------------------------------------------------------- bookkeeping
    df["M_has_wk52"] = df.Y_va_change_wk52.notna().astype(int)
    df["M_has_wk104"] = df.Y_va_change_wk104.notna().astype(int)
    df["X_va_below_69"] = (df.X_va_letters < 69).astype(int)   # 20/50 or worse: published subgroup

    order = (["pt_id", "treatment", "study_eye"]
             + sorted(c for c in df if c.startswith("X_"))
             + sorted(c for c in df if c.startswith("Y_"))
             + sorted(c for c in df if c.startswith("M_")))
    df = df[order].sort_values("pt_id")

    # ------------------------------------------------------------- checks
    assert df.pt_id.is_unique and len(df) == 660, "expected 660 unique patients"
    assert df.treatment.value_counts().to_dict() == {"Aflibercept": 224, "Bevacizumab": 218, "Ranibizumab": 218}
    assert df.X_va_letters.between(24, 78).all(), "baseline VA outside eligibility range 24-78"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  shape={df.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
